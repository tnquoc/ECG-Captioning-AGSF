USE_GPU = True

TOPIC = True
GHOSTNET = False
SPECTRAL = False
FNET = False

MODEL_CONFIG = {
    "in_length": 2500,
    "in_channels": 1,
    "n_grps": 3,
    "num_classes": 8,
    "N": 4,
    "k": 3,
    "dropout": 0.3,
    "first_width": 32,
    "stride": [2, 1, 2, 1],
    "num_layers": 4,
    "d_mode": 128,
    "nhead": 8,
    "dilation": 100,
}

DATA_CONFIG = {
    "data_dir": "/absolute/path/to/physionet.org/files/challenge-2021/1.0.3/training",
    "train_labels_csv": f"dataset/train_labels.csv",
    "val_labels_csv": "dataset/val_labels.csv",
    "test_labels_csv": "dataset/test_labels.csv",
}

HYPER_PARAMETERS_CONFIG = {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 60,
}

CHECKPOINT_LOCATION = 'checkpoints/pre_trained.ckpt'

LIST_MAIN_WORDS = ['atrial', 'fibrillation/flutter', '2nd', 'degree', 'av', 'block', '3rd',
                   'sinus', 'rhythm', 'svt', 'vt', 'tachycardia', 'bradycardia', 'at', 'unk', 'bpm', 'with',
                   'arrhythmia', '1st']
