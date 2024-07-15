import torch
import numpy as np
from scipy import interpolate


class ToTensor(object):
    """Converts ndarrays in sample to FloatTensors.
    """

    def __call__(self, sample):
        waveform = sample['waveform']
        if waveform.shape[0] == 12:
            # select only leads I, II and V1 until V6
            waveform = waveform[[0, 1, 6, 7, 8, 9, 10, 11], :]
            sample['waveform'] = torch.from_numpy(waveform.copy()).type(torch.float)
        else:
            sample['waveform'] = torch.from_numpy(waveform.copy()).type(torch.float)
        # else:
        #     sample['waveform'] = torch.from_numpy(waveform[:8, ].copy()).type(torch.float)
        #     print('Waveform with other no of leads than 8 or 12, please check!')

        if 'label' in sample:
            sample['label'] = torch.from_numpy(
                np.array(sample['label'])).type(torch.float)

        return sample


class ApplyGain(object):
    """Normalize ECG signal by multiplying by specified gain and converting to
    millivolts.
    """

    def __call__(self, sample):
        sample['waveform'] = sample['waveform'] * sample['gain'] * 0.001

        return sample


class Resample(object):
    """Convert 8 lead waveforms to their 12 lead equivalent using linear
    interpolation.
    """

    def __init__(self, sample_freq):
        """Initializes the resample transformation.

        Args:
            sample_freq (int): The required sampling frequency to resample to.
        """
        self.sample_freq = int(sample_freq)

    def __call__(self, sample):
        if sample['samplebase'] != None:
            samplebase = int(sample['samplebase'])
        else:
            if sample['waveform'].shape[1] in [300, 2500]:
                samplebase = 250
            elif sample['waveform'].shape[1] in [600, 5000]:
                samplebase = 500
            else:
                print("Unknown sample base")
                raise
        waveform = sample['waveform']

        if samplebase != self.sample_freq:
            length = int(waveform.shape[1])
            x = np.linspace(0, length / samplebase, num=length)
            f = interpolate.interp1d(x, waveform, axis=1)
            out_length = int((length / samplebase) * self.sample_freq)
            xnew = np.linspace(0, length / samplebase,
                               num=out_length)
            sample['waveform'] = f(xnew)
            sample['samplebase'] = self.sample_freq
        else:
            sample['samplebase'] = samplebase

        return sample