import os
import csv
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from model.topic_transformer import TopicTransformer
from utils.transforms import ToTensor
from evaluate.eval import COCOEvalCap, evaluate_for_confusion
from dataset.dataset import ECGDataset, collate_fn

from config import *


def init_directory():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./logs/results'):
        os.mkdir('./logs/results')


def test():
    device = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([ToTensor()])

    predict_file = 'logs/results/predict_test.csv'
    predict_cf_file = 'logs/results/test_confusion_matrix.csv'

    model = TopicTransformer.load_from_checkpoint(checkpoint_path=CHECKPOINT_LOCATION).to(device)
    model.eval()

    test_df = pd.read_csv(DATA_CONFIG['test_labels_csv'])
    is_train, vocab = False, model.vocab
    test_set = ECGDataset(
        length=len(test_df),
        dataset=test_df,
        vocab=vocab,
        training=is_train,
        waveform_dir=DATA_CONFIG['data_dir'],
        transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=HYPER_PARAMETERS_CONFIG['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    event_ids = test_df['EventID']

    # For transformer
    sample_method = {'temp': None, 'k': None, 'p': None, 'greedy': True, 'm': None}
    max_length = 50

    verbose = False
    gts = {}
    res = {}

    predict_results = []
    confusion_matrix = np.zeros((4, 8))
    with open(predict_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['EventID', 'TechnicianComments', 'PredictComments', 'Channel'])
    for batch_idx, batch in enumerate(tqdm.tqdm(test_loader)):
        waveforms, specs, ids, targets, _, _, _ = batch
        waveforms = waveforms.to(device)
        specs = waveforms.to(device)
        words = model.sample(waveforms, specs, sample_method, max_length)
        generated = model.vocab.decode(words, skip_first=False)
        truth = model.vocab.decode(targets)
        for i in range(waveforms.shape[0]):
            res[ids[i]] = [generated[i]]
            gts[ids[i]] = [truth[i]]
            sub_confusion_matrix, check_predict = evaluate_for_confusion(gts[ids[i]][0], res[ids[i]][0])
            confusion_matrix += sub_confusion_matrix
            with open(predict_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([event_ids[ids[i]], truth[i], generated[i], test_df['Channel'][i]])
            predict_results.append(check_predict)
            if verbose:
                print('\n')
                print(f'ID: {event_ids[ids[i]]}')
                print(f'True: {gts[ids[i]][0]}')
                print(f'Pred: {res[ids[i]][0]}')

    # save confusion matrix
    with open(predict_cf_file, 'w') as f:
        writer = csv.writer(f)
        event_types = np.array(
            ['', 'AFIB', 'AVB2', 'AVB3', 'SINUS', 'SVT', 'VT', 'TACHY', 'BRADY'])
        writer.writerow(event_types)
        for i in range(4):
            if i == 0:
                row = np.insert(confusion_matrix[i].astype(str), 0, 'TP')
            elif i == 1:
                row = np.insert(confusion_matrix[i].astype(str), 0, 'FN')
            elif i == 2:
                row = np.insert(confusion_matrix[i].astype(str), 0, 'FP')
            else:
                row = np.insert(confusion_matrix[i].astype(str), 0, 'TN')
            writer.writerow(row)
        SE = np.nan_to_num(
            confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1]))
        P = np.nan_to_num(
            confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[2]))
        F1 = np.nan_to_num(2 * SE * P / (SE + P))
        SE = np.append(SE, np.mean(SE))
        P = np.append(P, np.mean(P))
        F1 = np.append(F1, np.mean(F1))
        writer.writerow(np.insert(SE.astype(str), 0, 'SE'))
        writer.writerow(np.insert(P.astype(str), 0, 'P+'))
        writer.writerow(np.insert(F1.astype(str), 0, 'F1'))

        writer.writerow(np.array(['']))
    f.close()

    print('Percent of wrong sentence is', (1 - np.sum(np.array(predict_results)) / len(predict_results)) * 100, ' %')

    COCOEval = COCOEvalCap()
    COCOEval.evaluate(gts, res)
    print(sample_method, COCOEval.eval)


if __name__ == '__main__':
    init_directory()
    test()
