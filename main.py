import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.dataset import get_loaders
from model.topic_transformer import TopicTransformer

from config import *


def init_directory():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./logs/results'):
        os.mkdir('./logs/results')


def cli_main():
    pl.seed_everything(1234)

    train_loader, val_loader, test_loader, vocab = get_loaders(DATA_CONFIG)

    model = TopicTransformer(vocab, **MODEL_CONFIG)

    checkpoint_callback = ModelCheckpoint(save_top_k=-1)

    trainer = pl.Trainer(max_epochs=HYPER_PARAMETERS_CONFIG['epochs'],
                         default_root_dir='logs/training/',
                         log_every_n_steps=5,
                         callbacks=[checkpoint_callback],
                         gpus=use_gpu)

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    use_gpu = 1 if USE_GPU and torch.cuda.is_available() else 0
    init_directory()
    cli_main()
