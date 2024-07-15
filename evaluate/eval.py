import numpy as np

from .ptbtokenize import PTBTokenizer
from .metrics.bleu import Bleu
from .metrics.cider import Cider
from .metrics.meteor import Meteor
from .metrics.rouge import Rouge


class COCOEvalCap:
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self, gts, res):

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # print(gts)
        # =================================================
        # Set up scorers
        # =================================================
        # print('----- gts', gts)
        # print('----- res', res)
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if method == 'METEOR':
                continue
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                print("%s: %0.3f" % (method, score))

    def setEval(self, score, method):
        self.eval[method] = score


def create_phrase_index(corpus):
    list_index = np.zeros(8)
    if 'fibrillation' in corpus:
        list_index[0] = 1
    if '2nd' in corpus:
        list_index[1] = 1
    if '3rd' in corpus:
        list_index[2] = 1
    if 'sinus rhythm' in corpus or 'sinus arrhythmia' in corpus or '1st' in corpus:
        list_index[3] = 1
    if 'svt' in corpus:
        list_index[4] = 1
    if 'vt' in corpus.split():
        list_index[5] = 1
    if 'sinus tachycardia' in corpus:
        list_index[6] = 1
    if 'sinus bradycardia' in corpus:
        list_index[7] = 1

    return list_index


def evaluate_new(gts, res):
    list_index_gts = create_phrase_index(gts)
    list_index_res = create_phrase_index(res)
    result = 1
    for i in range(len(list_index_gts)):
        if list_index_gts[i] == 1 and list_index_res[i] == 0:
            result = 0
            break
    return result


def evaluate_for_confusion(gts, res, ignore=None):
    list_index_gts = create_phrase_index(gts)
    list_index_res = create_phrase_index(res)

    # define result matrix with 4 row and 10 columns
    # 4 rows corresponding to TP, FN, FP, TN
    # 8 columns corresponding to types
    sub_confusion_matrix = np.zeros((4, 8))
    sub_confusion_matrix[0, :] = np.multiply(list_index_gts, list_index_res)
    sub_confusion_matrix[1, :] = np.multiply(list_index_gts, np.ones(8) - list_index_res)
    sub_confusion_matrix[2, :] = np.multiply(list_index_res, np.ones(8) - list_index_gts)
    sub_confusion_matrix[3, :] = np.multiply(np.ones(8) - list_index_gts, np.ones(8) - list_index_res)

    if ignore == 'sinus':
        if 'sinus arrhythmia' in gts and 'sinus arrhythmia' in res:
            pass
        elif list_index_gts[4] == 1 and np.sum(list_index_gts) >= 2 and list_index_res[4] == 1\
                and np.sum(list_index_res) == 1:
            sub_confusion_matrix[0, 4] = 0
            sub_confusion_matrix[1, 4] = 1
    check_predict = 1
    for i in range(len(list_index_gts)):
        if list_index_gts[i] == 1 and list_index_res[i] == 0:
            check_predict = 0
            break
    return sub_confusion_matrix, check_predict
