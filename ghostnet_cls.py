import os
import time
import pickle
import csv
import random
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

import tqdm
import wfdb
from scipy.signal import butter, filtfilt
from model.SGBNet import SGBModelCls


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def initialize_log_directory():
    if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
        os.mkdir(os.path.join(os.getcwd(), 'logs'))


def init_dir(save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(os.path.join(save_model_path, 'checkpoints')):
        os.mkdir(os.path.join(save_model_path, 'checkpoints'))


def initialization(seed=0):
    initialize_seed(seed)
    initialize_log_directory()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


class ECGDataset(Dataset):
    def __init__(self, dataset, waveform_dir):
        self.length = len(dataset)
        self.dataset = dataset
        self.waveform_dir = waveform_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw_signal, _ = wfdb.rdsamp(self.waveform_dir + '/' + self.dataset['EventID'][idx])
        raw_signal = raw_signal[:, self.dataset['Channel'][idx]]
        waveform = np.nan_to_num(raw_signal)

        if len(waveform) < 2500:
            waveform = np.pad(waveform, (0, 2500 - len(waveform)))
        elif len(waveform) > 2500:
            waveform = waveform[:2500]
        waveform = waveform[None, :]
        # waveform = butter_bandpass_filter(waveform, 0.1, 40, 250)
        label = 0 if 'sinus' in self.dataset['Label'][idx] else 1

        sample = {
            'waveform': torch.from_numpy(waveform.copy()).type(torch.FloatTensor),
            'label': int(label),
        }

        return sample


def get_loaders(params, batch_size):
    base_data_path = params['base_data_path']
    train_dataset_path = params['train_set_path']
    valid_dataset_path = params['valid_set_path']
    test_dataset_path = params['test_set_path']

    train_set = pd.read_csv(train_dataset_path)
    valid_set = pd.read_csv(valid_dataset_path)
    test_set = pd.read_csv(test_dataset_path)

    train_dataset = ECGDataset(dataset=train_set, waveform_dir=base_data_path)
    valid_dataset = ECGDataset(dataset=valid_set, waveform_dir=base_data_path)
    test_dataset = ECGDataset(dataset=test_set, waveform_dir=base_data_path)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=4,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=4)

    return train_loader, valid_loader, test_loader


def calculate_metrics(target, pred, threshold=0.5, return_confusion_matrix=False):
    auc = roc_auc_score(target, pred)

    y_pred = np.array(pred >= threshold, dtype=int)
    y_true = target.astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    all_metrics = {
        'acc': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    if return_confusion_matrix:
        return all_metrics, confusion_matrix(y_true, y_pred)
    else:
        return all_metrics, None


def train_one_epoch(data_loader, model, criterion, optimizer, device='cpu'):
    training_loss = 0
    training_acc = 0
    data_iterator = tqdm.tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    model.train()
    for i, samples in enumerate(data_iterator):
        # Get data
        waveform, labels = samples['waveform'], samples['label']

        # clear gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(waveform.to(device))
        p_outputs = torch.sigmoid(outputs)

        # calculate loss
        loss = criterion(outputs.to(device), labels.to(device))
        acc = (p_outputs.argmax(1).cpu().int() == labels).sum() / labels.shape[0]

        # Backprop and optimize
        loss.backward()
        optimizer.step()

        # update running metrics
        training_loss += loss.item()
        training_acc += acc

    training_loss /= len(data_loader)
    training_acc /= len(data_loader)

    return training_loss, training_acc


def evaluate(data_loader, model, criterion, device='cpu'):
    validation_loss = 0
    validation_acc = 0
    model_results = []
    targets = []
    data_iterator = tqdm.tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    model.eval()
    with torch.no_grad():
        for i, samples in enumerate(data_iterator):
            # Get data
            waveform, labels = samples['waveform'], samples['label']

            # Forward pass
            outputs = model(waveform.to(device))
            p_outputs = torch.sigmoid(outputs)

            # calculate loss
            loss = criterion(outputs.to(device), labels.to(device))
            acc = (p_outputs.argmax(1).cpu().int() == labels).sum() / labels.shape[0]

            # update running metrics
            validation_loss += loss.item()
            validation_acc += acc

            model_results.extend(p_outputs[:, 1].cpu().detach().numpy())
            targets.extend(labels.cpu().numpy())

    validation_loss /= len(data_loader)
    validation_acc /= len(data_loader)
    metrics, _ = calculate_metrics(np.array(targets), np.array(model_results))

    return validation_loss, validation_acc, metrics


def train(params, save_model_path):
    # get config
    hyperparameters = params['hyperparameters']

    # get device
    if params['aux']['device'] == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')
    else:
        device = 'cpu'
        print('Device: cpu')

    # get data
    train_loader, val_loader, _, = get_loaders(params['data'], hyperparameters['batch_size'])

    # define model
    model = SGBModelCls(num_classes=2).to(device)
    print('Number of model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get criterion, optimizer/scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])

    num_epochs = hyperparameters['epochs']
    best_loss = float('inf')
    best_score = 0
    train_loss_log = []
    validation_loss_log = []

    print('Start training...')
    print('Epochs:', num_epochs)
    print('Iterations per training epoch:', len(train_loader))
    print('Iterations per validation epoch:', len(val_loader))

    for epoch in range(num_epochs):
        print(f'####### Epoch [{epoch + 1}/{num_epochs}] #######')
        for param_group in optimizer.param_groups:
            print('LR: {:.6f}'.format(param_group['lr']))

        # training
        t0 = time.time()
        training_loss, training_acc = train_one_epoch(train_loader, model, criterion, optimizer, device)

        # validation
        validation_loss, validation_acc, metrics = evaluate(val_loader, model, criterion, device)

        # calculate global loss/metrics and log process
        print("Epoch[{}/{}] - Loss: {:.4f} - Accuracy: {:.4f} - ValLoss: {:.4f} - ValAccuracy: {:.4f}, ETA: {:.0f}s"
              .format(epoch + 1, num_epochs, training_loss, training_acc, validation_loss, validation_acc,
                      time.time() - t0))
        print('ACC: {:.4f} | AUC: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(
            metrics['acc'],
            metrics['auc'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ))

        train_loss_log.append(training_loss)
        validation_loss_log.append(validation_loss)

        # select and save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/checkpoint_epoch{epoch + 1}.pth'))

        if best_loss > validation_loss:
            best_loss = validation_loss
            print(f'Save best loss checkpoint at epoch {epoch + 1}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_loss_checkpoint.pth'))

        score = metrics['f1']
        if score > best_score:
            best_score = score
            print(f'Save best metrics checkpoint at epoch {epoch + 1}')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_metrics_checkpoint.pth'))

    loss_history = {
        'train': train_loss_log,
        'validation': validation_loss_log,
    }
    with open(f'{save_model_path}/history.pkl', 'wb') as f:
        pickle.dump(loss_history, f)


def test(params, save_model_path, mode='metric'):
    # get config
    hyperparameters = params['hyperparameters']

    # get device
    if params['aux']['device'] == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')
    else:
        device = 'cpu'
        print('Device: cpu')

    # get data
    _, _, test_loader = get_loaders(params['data'], hyperparameters['batch_size'])

    # load model
    model = SGBModelCls(num_classes=2).to(device)
    print('Number of model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if mode == 'metric':
        checkpoint = torch.load(os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_metrics_checkpoint.pth'),
                                map_location=device)
    else:
        checkpoint = torch.load(os.path.join(os.getcwd(), f'{save_model_path}/checkpoints/best_loss_checkpoint.pth'),
                                map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_acc = 0
    model_results = []
    targets = []
    test_iterator = tqdm.tqdm(test_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, samples in enumerate(test_iterator):
        # Get data
        waveform, labels = samples['waveform'], samples['label']

        # Forward pass
        outputs = model(waveform.to(device))
        p_outputs = torch.sigmoid(outputs)

        # calculate metrics
        acc = (p_outputs.argmax(1).cpu().int() == labels).sum() / labels.shape[0]
        test_acc += acc.item()

        model_results.extend(p_outputs[:, 1].cpu().detach().numpy())
        targets.extend(labels.cpu().numpy())

    test_acc /= len(test_iterator)
    metrics, all_confusion_matrix = calculate_metrics(np.array(targets), np.array(model_results),
                                                      return_confusion_matrix=True)

    with open(f'{save_model_path}/test_best_{mode}_checkpoint_prediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['TP', 'FN', 'FP', 'TN', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1-score'])
        tn, fp, fn, tp = all_confusion_matrix.ravel()
        writer.writerow([round(tp, 4), round(fn, 4), round(fp, 4), round(tn, 4),
                         round(metrics['acc'], 4), round(metrics['auc'], 4),
                         round(metrics['precision'], 4), round(metrics['recall'], 4), round(metrics['f1'], 4)])


if __name__ == '__main__':
    config = {
        'data': {
            'base_data_path': '/absolute/path/tp/mit_data',
            'train_set_path': 'absolute/path/to/mit_train.csv',
            'valid_set_path': 'absolute/path/to/mit_val.csv',
            'test_set_path': 'absolute/path/to/mit_test.csv',
        },
        'hyperparameters': {
            'epochs': 50,
            'batch_size': 128,
            'lr': 1e-4,
        },
        'aux': {
            'seed': 42,
            'device': 'cuda',
        }
    }

    # initialization
    initialization(config['aux']['seed'])
    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'SGB'
    save_model_path = f'logs/ghost/{current_time}_{model_name}'
    init_dir(save_model_path)

    # train and test
    train(config, save_model_path)
    test(config, save_model_path, 'metric')
    test(config, save_model_path, 'loss')
