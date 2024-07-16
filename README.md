# ECG-Captioning-AGSF
This is the official implementation of paper "ECG Captioning with Prior-Knowledge Transformer and Diffusion Probabilistic Model"

## Prerequisites
```
pip install -r requirements.txt
```
- Prepare the **wfdb** library
- Download standford corenlp via this link https://stanfordnlp.github.io/CoreNLP/index.html
- Create folder **checkpoints**
- Project structure:
```
├── checkpoints/
├── dataset/
├── evaluate/
├── logs/
├── model/
├── stanford-corenlp-4.4.0/
├── utils/
├── wfdb/
├── main.py
├── test.py
└── ...
```

## Prepare dataset

- Download the Physionet Challenge 2021 dataset (v1.0.3 | [link](https://physionet.org/content/challenge-2021/1.0.3/))
- We release the train/validation/test sets in folder **dataset**

## Usage

### Training

1. Data: 
   - Set DATA_CONFIG['data_dir'] in file **config.py** by the path to the downloaded Physionet Challenge 2021 dataset.
   - To maintain optimal performance during training, please augment the data using the method described in the provided [paper](https://dl.acm.org/doi/abs/10.1145/3591569.3591621). Additionally, ensure that the **train_labels.csv** file in folder **dataset** is recreated to accurately correspond with your augmented data.
2. Run ```python main.py``` to train model.
3. Checkpoints are saved in folder logs/training. 
4. Confusion matrix of each epoch is saved in file logs/results/log_val_cf.csv
5. Modify values in file **config.py** for different variants:
   - M1: TOPIC = False, SPECTRAL = False, FNET_USE = False, GHOSTNET = True 
   - M2: TOPIC = True, SPECTRAL = False, FNET_USE = False, GHOSTNET = True
   - M3: TOPIC = False, SPECTRAL = True, FNET_USE = False, GHOSTNET = True
   - M4: TOPIC = False, SPECTRAL = False, FNET_USE = True, GHOSTNET = True
   - M5: TOPIC = True, SPECTRAL = True, FNET_USE = True, GHOSTNET = True

### Testing

1. Download our released [checkpoint](https://drive.google.com/file/d/1TQnIc0oBQ3ld5Pj6t3H7084I8XWV_q30/view?usp=sharing) and place in folder **checkpoints**
2. Run ```python test.py``` to test model. 
3. Test results are saved in folder logs/results, which contains predicted captions and confusion matrix results. 
4. The given example checkpoint is in the mode TOPIC = True, SPECTRAL = False, FNET_USE = False, GHOSTNET = False
5. Current Metrics (NLP + CLS):

| BLEU1 | BLEU2 | BLEU3 | BLEU4 | ROUGE | CIDER | SE_Mean | P+_Mean | F1_Mean |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-------:|:-------:|
| 0.88  | 0.87  | 0.81  | 0.79  | 0.88  | 8.35  |  0.834  |  0.84   |  0.835  |

## References

- A Novel Approach for Long ECG Synthesis Utilize Diffusion Probabilistic Model ([paper](https://dl.acm.org/doi/abs/10.1145/3591569.3591621) | [code](https://github.com/tnquoc/ECG-Diffusion-DiffWave))
- Efficient ECG Classification with Light Weight Shuffle GhostNet Architecture ([paper](https://ieeexplore.ieee.org/document/10318918) | [code](https://github.com/tnquoc/SGBNet))

## Citation