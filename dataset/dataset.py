import os
import json
import librosa
from operator import itemgetter
from nltk.tokenize import RegexpTokenizer
from collections import Counter

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from evaluate.eval import create_phrase_index
from utils.transforms import ToTensor

import numpy as np
import pandas as pd
import wfdb

from vocab import Vocabulary
from wfdb.processing import resample_sig
from config import *


class ECGDataset(Dataset):
    def __init__(self, length, dataset, vocab, training, waveform_dir, transform, label='Label'):
        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.length = length
        self.transform = transform
        self.label = label
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')
        self.training = training
        if training:
            self.vocab = self.setup_vocab(self.dataset['Label'])
            self.weights = self.setup_weights(self.dataset['Label'])
        else:
            self.vocab = vocab
            self.weights = None

    def setup_vocab(self, labels):
        corpus = labels.str.cat(sep=" ")

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        counter = counter.most_common()
        words = []
        cnts = []
        for i in range(0, len(counter)):
            words.append(counter[i][0])
            if words[-1] in ['sinus', 'arrhythmia', 'bradycardia', 'rhythm']:
                cnts.append(0.25)
            elif words[-1] in ['atrial', 'fibrillation/flutter', 'svt', 'tachycardia',
                               'vt', '1st', 'degree', '2nd', '3rd', 'block', 'av']:
                cnts.append(0.9)
            else:
                cnts.append(0.1)
        vocab = Vocabulary()
        vocab.add_word('<pad>', min(cnts))
        vocab.add_word('<start>', min(cnts))
        vocab.add_word('<end>', min(cnts))
        vocab.add_word('<unk>', min(cnts))
        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word, cnts[i])

        return vocab

    def setup_weights(self, labels):
        weights = np.zeros(len(labels))
        for i in range(0, len(labels)):
            if 'fibrillation/flutter' in labels[i] \
                    or 'svt' in labels[i] or 'vt' in labels[i] \
                    or '1st' in labels[i] or '2nd' in labels[i] or '3rd' in labels[i] \
                    or 'av' in labels[i]:
                weights[i] = 0.75
            elif 'sinus' in labels[i]:
                weights[i] = 0.25
            else:
                raise ValueError(f'Error at labels {labels[i]} {i}')

        return weights

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform, spec, sample_id = self.get_waveform(idx)

        sample = {
            'waveform': waveform,
            'spec': spec,
            'id': sample_id,
            'weights': self.weights[idx] if self.training else None
        }

        if self.label in self.dataset.columns.values:
            sentence = self.dataset[self.label].iloc[idx]
            try:
                tokens = self.tokenizer.tokenize(sentence)
            except:
                print(sentence)
                raise Exception()
            vocab = self.vocab
            caption = [vocab('<start>')]
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)

            sample['label'] = target

        sample['extra_label'] = torch.tensor(create_phrase_index(self.dataset['Label'][idx]), dtype=torch.float)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_waveform(self, idx):
        raw_signal, fields = wfdb.rdsamp(self.waveform_dir + '/' + self.dataset['EventID'][idx])
        fs = fields['fs']
        raw_signal = raw_signal[:, self.dataset['Channel'][idx]]
        waveform = np.nan_to_num(raw_signal)

        waveform = resample_sig(waveform, fs, 250)[0]
        if len(waveform) < 2500:
            waveform = np.pad(waveform, (0, 2500 - len(waveform)))
        elif len(waveform) > 2500:
            waveform = waveform[:2500]
        waveform = waveform[None, :]

        spec = np.squeeze(waveform)
        if SPECTRAL:
            spec = np.squeeze(waveform)
            spec = librosa.feature.melspectrogram(y=spec, sr=250, hop_length=128)
            spec = librosa.power_to_db(spec, ref=np.max)
            spec = librosa.util.normalize(spec)

            spec = np.expand_dims(spec, axis=0)

        return waveform, torch.from_numpy(spec).type(torch.FloatTensor), idx


def get_loaders(params):
    transform = transforms.Compose([ToTensor()])

    train_df = pd.read_csv(params['train_labels_csv'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_labels_csv'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    test_df = pd.read_csv(params['test_labels_csv'])

    is_train, vocab = True, None
    train_set = ECGDataset(
        length=len(train_df),
        dataset=train_df,
        vocab=vocab,
        training=is_train,
        waveform_dir=params['data_dir'],
        transform=transform
    )

    is_train, vocab = False, train_set.vocab
    val_set = ECGDataset(
        length=len(val_df),
        dataset=val_df,
        vocab=vocab,
        training=is_train,
        waveform_dir=params['data_dir'],
        transform=transform
    )

    test_set = ECGDataset(
        length=len(test_df),
        dataset=test_df,
        vocab=vocab,
        training=is_train,
        waveform_dir=params['data_dir'],
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=HYPER_PARAMETERS_CONFIG['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=HYPER_PARAMETERS_CONFIG['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(test_set, batch_size=HYPER_PARAMETERS_CONFIG['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, vocab


def collate_fn(data):

    captions = [d['label'] for d in data]
    lengths = [len(cap) for cap in captions]

    if len(lengths) == 1:
        return data[0]['waveform'].unsqueeze(0), data[0]['spec'].unsqueeze(0), np.array([data[0]['id']]), data[0][
            'label'].unsqueeze(0).long(), lengths, data[0]['weights'], data[0]['extra_label'].unsqueeze(0).long()

    ind = np.argsort(lengths)[::-1]

    lengths = list(itemgetter(*ind)(lengths))
    captions = list(itemgetter(*ind)(captions))

    waveforms = list(itemgetter(*ind)([d['waveform'] for d in data]))
    specs = list(itemgetter(*ind)([d['spec'] for d in data]))
    ids = list(itemgetter(*ind)([d['id'] for d in data]))
    weights = list(itemgetter(*ind)([d['weights'] for d in data]))

    # Merge images (from tuple of 3D tensor to 4D tensor).
    waveforms = torch.stack(waveforms, 0)
    specs = torch.stack(specs, 0)

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if 'extra_label' in data[0]:
        extra_label = list(itemgetter(*ind)([d['extra_label'] for d in data]))
        extra_label = torch.stack(extra_label, 0)
        return waveforms, specs, ids, targets, lengths, weights, extra_label

    return waveforms, specs, ids, targets, lengths, weights

