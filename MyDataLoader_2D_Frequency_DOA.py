import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import torchaudio.transforms as transforms
import numpy as np


def minmaxscaler_per_channel(data): # data(channels, samples)
    scaled_data = torch.zeros_like(data)
    for i in range(data.size(0)):
        channel_min = data[i].min()
        channel_max = data[i].max()
        scaled_data[i] = data[i] / (channel_max - channel_min)
    return scaled_data


class STFT_Feature:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=64, fmax=1000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.transformation = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)

    def __call__(self, signal):
        spectrogram = self.transformation(signal)  # (channels, fre_bins, time_frames)
        
        # Calculate frequency range
        freq_bins = self.n_fft // 2 + 1
        freq_resolution = self.sample_rate / self.n_fft
        max_bin = int(self.fmax/freq_resolution)
        
        # Crop frequency bins
        spectrogram = spectrogram[:, :max_bin, :]
        
        phase = torch.angle(spectrogram)
        magnitude = torch.abs(spectrogram)
        features = torch.cat((magnitude, phase), dim=0)
        
        return features


class My2DNoiseDataset_Frequency_DOA(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
        self.trasformation = STFT_Feature()
        
    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        frequency_label = self._get_audio_frequency_label(index)
        doa_label = self._get_audio_doa_label(index)
        signal,_ = torchaudio.load(os.path.join(self.folder, audio_sample_path)) # torch.Size([4, 8000])
        signal = minmaxscaler_per_channel(signal)
        spectorgram = self.trasformation(signal)
        return spectorgram, frequency_label, doa_label
    
    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_frequency_label(self, index):
        label = self.annotations_file.iloc[index,2]
        return label
    
    def _get_audio_doa_label(self, index):
        label = self.annotations_file.iloc[index,3]
        return label
    
    
class My2DNoiseDataset_Frequency_DOA1(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
        self.trasformation = STFT_Feature()
        
    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        frequency_label = self._get_audio_frequency_label(index)
        doa_label = self._get_audio_doa_label(index)
        signal,_ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler_per_channel(signal)
        spectorgram = self.trasformation(signal)
        return audio_sample_path, spectorgram, frequency_label, doa_label # !!!change
    
    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_frequency_label(self, index):
        label = self.annotations_file.iloc[index,2]
        return label
    
    def _get_audio_doa_label(self, index):
        label = self.annotations_file.iloc[index,3]
        return label