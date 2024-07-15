import copy
import random
import csv

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from .ecgresnet import ECGResNet
from .network_topic import MLC, CoAttention
from .spectrogram_miniresnet import SpectrogramMiniResnet
from .transformer_topic import TopicTransformerModule
from .SGBNet import SGBModel

from utils.util import get_next_word
from evaluate.eval import evaluate_for_confusion
from config import *


class TopicTransformer(pl.LightningModule):
    def __init__(self, vocab, in_length, in_channels,
                 n_grps, N, num_classes, k,
                 dropout, first_width,
                 stride, dilation, num_layers, d_mode, nhead, **kwargs):
        super().__init__()
        self.vocab_length = len(vocab)
        self.vocab = vocab
        self.save_hyperparameters()

        if GHOSTNET:
            self.model = SGBModel()
            self.model.out_layer = AveragePool()
        else:
            self.model = ECGResNet(in_length, in_channels,
                                   n_grps, N, num_classes,
                                   dropout, first_width,
                                   stride, dilation)
            self.model.flatten = Identity()
            self.model.fc1 = AveragePool()
            self.model.fc2 = AveragePool()

        if SPECTRAL:
            self.custom_spectrum_model = SpectrogramMiniResnet(in_channels)

        self.pre_train = False
        if SPECTRAL:
            self.feature_embedding = nn.Linear(768, d_mode)
        else:
            self.feature_embedding = nn.Linear(256, d_mode)
        self.embed = nn.Embedding(len(vocab), 2 * d_mode)

        mlc = MLC(classes=num_classes, sementic_features_dim=d_mode, fc_in_features=256, k=k)
        attention = CoAttention(version='v6', embed_size=d_mode, hidden_size=d_mode, visual_size=256, k=k)

        self.transformer = TopicTransformerModule(d_mode, nhead, num_layers, mlc, attention)
        self.transformer.apply(init_weights)

        self.to_vocab = nn.Sequential(nn.Linear(2 * d_mode, len(vocab)))

        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.nlll_criterion = nn.NLLLoss(reduction="none", ignore_index=vocab.word2idx['<pad>'])
        self.mse_criterion = nn.MSELoss(reduction="none")

        self.val_confusion_matrix = np.zeros((4, 8))
        self.list_val_F1_score = []

    def load_pre_trained(self, pre_trained):
        self.transformer = pre_trained.transformer
        self.model = pre_trained.model
        self.embed = pre_trained.embed
        self.feature_embedding = pre_trained.feature_embedding
        self.to_vocab = pre_trained.to_vocab

        self.pre_train = True

    def sample(self, waveforms, specs, sample_method, max_length):
        if GHOSTNET:
            image_features, avg_feats = self.model(waveforms)
        else:
            _, (image_features, avg_feats) = self.model(waveforms)
        if SPECTRAL:
            spec_image_features = self.custom_spectrum_model(specs)
            spec_image_features = spec_image_features.reshape(spec_image_features.shape[0],
                                                              spec_image_features.shape[1],
                                                              spec_image_features.shape[2] * spec_image_features.shape[3])
            image_features = torch.cat([image_features, spec_image_features], dim=1)
        image_features = image_features.transpose(1, 2).transpose(1, 0)  # ( batch, feature, number)
        image_features = self.feature_embedding(image_features)

        start_tokens = torch.tensor([self.vocab('<start>')], device=image_features.device)
        nb_batch = waveforms.shape[0]
        start_tokens = start_tokens.repeat(nb_batch, 1)
        sent = self.embed(start_tokens).transpose(1, 0)

        attended_features = None

        tgt_mask = torch.zeros(sent.shape[1], sent.shape[0], device=image_features.device, dtype=bool)
        y_out = torch.zeros(nb_batch, max_length, device=image_features.device)

        for i in range(max_length):
            out, attended_features = self.transformer.forward_one_step(image_features, avg_feats, sent, tgt_mask,
                                                                       attended_features=attended_features)
            out = self.to_vocab(out[-1, :, :]).squeeze(0)
            s = sample_method
            word_idx, props = get_next_word(out, temp=s['temp'], k=s['k'], p=s['p'], greedy=s['greedy'], m=s['m'])
            y_out[:, i] = word_idx

            ended_mask = (tgt_mask[:, -1] | (word_idx == self.vocab('<end>'))).unsqueeze(1)
            tgt_mask = torch.cat((tgt_mask, ended_mask), dim=1)

            embedded = self.embed(word_idx).unsqueeze(0)
            sent = torch.cat((sent, embedded), dim=0)

            if ended_mask.sum() == nb_batch:
                break

        return y_out

    def forward(self, waveforms, specs, targets):
        if GHOSTNET:
            image_features, avg_feats = self.model(waveforms)
        else:
            _, (image_features, avg_feats) = self.model(waveforms)

        if SPECTRAL:
            spec_image_features = self.custom_spectrum_model(specs)
            spec_image_features = spec_image_features.reshape(spec_image_features.shape[0],
                                                              spec_image_features.shape[1],
                                                              spec_image_features.shape[2] * spec_image_features.shape[3])
            image_features = torch.cat([image_features, spec_image_features], dim=1)
        image_features = image_features.transpose(1, 2).transpose(1, 0)  # ( batch, feature, number)
        image_features = self.feature_embedding(image_features)
        tgt_key_padding_mask = targets == 0

        embedded = self.embed(targets).transpose(1, 0)
        out, tags = self.transformer(image_features, avg_feats, embedded, tgt_key_padding_mask)

        vocab_distribution = self.to_vocab(out)
        return vocab_distribution, tags

    def check_prediction(self, out, targets):
        nb_batch = out.shape[1]
        index = random.randint(0, nb_batch - 1)

        val, idx = out[:, index].max(dim=1)
        pred = ' '.join([self.vocab.idx2word[idxn.item()] for idxn in idx])
        truth = ' '.join([self.vocab.idx2word[word.item()] for word in targets[index]])

        print('\n True:', truth)
        print('Pred:' + pred)

    def loss_tags(self, tags, label):
        tag_loss = self.mse_criterion(tags, label).sum(dim=1)
        return tag_loss

    def loss(self, out, targets, weights, tags, topic, type_weight=None):
        out = F.log_softmax(out, dim=-1).reshape(-1, len(self.vocab))
        target = targets[:, 1:]
        batch_size, seq_length = target.shape
        target = target.transpose(1, 0).reshape(-1)
        loss = self.nlll_criterion(out, target)
        if type_weight:
            if type_weight == 's':
                alpha = torch.tensor(weights).reshape(batch_size, 1)
            else:
                alpha = copy.deepcopy(torch.tensor(self.vocab.weights)[target])

            pt = torch.exp(-loss)
            gamma = 2
            loss = (alpha.cuda() * (1 - pt) ** gamma * loss).mean()
        else:
            loss = loss.reshape(batch_size, seq_length).sum(dim=1).mean(dim=0)

        if TOPIC:
            target_loss = self.loss_tags(tags, topic)

            return (target_loss * 0.3 + loss * 0.7).mean()
        else:
            return loss.mean()

    def on_train_epoch_start(self):
        if self.pre_train:
            for param in self.transformer.mlc.parameters():
                param.requires_grad = False
            for param in self.transformer.attention.parameters():
                param.requires_grad = False
            for param in self.transformer.transformer_encoder.parameters():
                param.requires_grad = self.current_epoch > 5
            for param in self.model.parameters():
                param.requires_grad = self.current_epoch > 5

    def training_step(self, batch, batch_idx):
        waveforms, specs, ids, targets, lengths, weights, labels = batch
        vocab_distribution, tags = self.forward(waveforms, specs, targets)
        vocab_distribution = vocab_distribution[:-1, :, :]
        loss = self.loss(vocab_distribution, targets, weights, tags, labels, type_weight=None)

        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, specs, ids, targets, lengths, weights, labels = batch
        vocab_distribution, tags = self.forward(waveforms, specs, targets)
        vocab_distribution = vocab_distribution[:targets.shape[1] - 1, :, :]
        loss = self.loss(vocab_distribution, targets, weights, tags, labels, type_weight=None)

        sample_method = {'temp': None, 'k': None, 'p': None, 'greedy': True, 'm': None}
        max_length = 50
        words = self.sample(waveforms, specs, sample_method, max_length)
        generated = self.vocab.decode(words, skip_first=False)
        truth = self.vocab.decode(targets)
        gts = {}
        res = {}

        for i in range(waveforms.shape[0]):
            res[ids[i]] = [generated[i]]
            gts[ids[i]] = [truth[i]]
            sub_confusion_matrix, check_predict = evaluate_for_confusion(gts[ids[i]][0], res[ids[i]][0], ignore='sinus')
            self.val_confusion_matrix += sub_confusion_matrix

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=HYPER_PARAMETERS_CONFIG['learning_rate'])
        return optimizer

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            with open('./logs/results/log_val_cf.csv', 'w') as f:
                pass
            f.close()
        else:
            with open('./logs/results/log_val_cf.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(np.array(['Epochs', self.current_epoch]))
                event_types = np.array(
                    ['', 'AFIB', 'AVB2', 'AVB3', 'SINUS', 'SVT', 'VT', 'TACHY', 'BRADY', 'PAUSE', 'OTHER'])
                writer.writerow(event_types)
                for i in range(4):
                    if i == 0:
                        row = np.insert(self.val_confusion_matrix[i].astype(str), 0, 'TP')
                    elif i == 1:
                        row = np.insert(self.val_confusion_matrix[i].astype(str), 0, 'FN')
                    elif i == 2:
                        row = np.insert(self.val_confusion_matrix[i].astype(str), 0, 'FP')
                    else:
                        row = np.insert(self.val_confusion_matrix[i].astype(str), 0, 'TN')
                    writer.writerow(row)
                SE = np.nan_to_num(
                    self.val_confusion_matrix[0] / (self.val_confusion_matrix[0] + self.val_confusion_matrix[1]))
                P = np.nan_to_num(
                    self.val_confusion_matrix[0] / (self.val_confusion_matrix[0] + self.val_confusion_matrix[2]))
                F1 = np.nan_to_num(2 * SE * P / (SE + P))
                SE = np.append(SE, np.mean(SE))
                P = np.append(P, np.mean(P))
                F1 = np.append(F1, np.mean(F1))
                self.list_val_F1_score.append(np.mean(F1))
                writer.writerow(np.insert(SE.astype(str), 0, 'SE'))
                writer.writerow(np.insert(P.astype(str), 0, 'P+'))
                writer.writerow(np.insert(F1.astype(str), 0, 'F1'))

                writer.writerow(np.array(['']))
            f.close()

        self.val_confusion_matrix = np.zeros((4, 8))

    def on_train_end(self):
        max_F1 = max(self.list_val_F1_score)
        argmax_F1 = self.list_val_F1_score.index(max_F1)
        with open('./logs/results/log_val_cf.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.array(
                ['', '', '', '', '', '', '', '', '', '', '', '', '',
                 f'Best model at epoch {argmax_F1} with F1 score {max_F1}']))
        f.close()
        print(f'Best model at epoch {argmax_F1} with F1 score {max_F1}')


class AveragePool(nn.Module):
    def __init__(self, kernel_size=10):
        super(AveragePool, self).__init__()

    def forward(self, x):
        signal_size = x.shape[-1]
        kernel = torch.nn.AvgPool1d(signal_size)
        average_feature = kernel(x).squeeze(-1)
        return x, average_feature


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
