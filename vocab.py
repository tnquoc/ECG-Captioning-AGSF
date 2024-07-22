import numpy as np
import torch
from config import *


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.weights = torch.tensor([])
        self.idx = 0

    def add_word(self, word, weight):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.weights = torch.cat([self.weights, torch.tensor([weight])])
            self.idx += 1

    def get_word(self, idx):
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def decode(self, word_idxs, listfy=False, join_words=True, skip_first=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            if skip_first:
                wis = wis[1:]
            for wi in wis:
                word = self.idx2word[int(wi)]
                if word == '<end>':
                    break
                caption.append(word)

            if join_words:
                caption = ' '.join(caption)
            if listfy:
                caption = [caption]
            captions.append(caption)
        return captions
