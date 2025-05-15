import numpy as np
from torch.utils.data import Dataset
import torch

def extract_valid_swing(data, sample_rate=85, energy_window_sec=0.5, energy_percentile=40):
    
    N_total = len(data)
    data = data[10:N_total-10] # remove annomly value at beginning
    N_total = len(data)

    ax, ay, az = data[:, 0], data[:, 1], data[:, 2]
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    
    energy_window_size = int(energy_window_sec * sample_rate)
    energy = np.convolve(acc_mag**2, np.ones(energy_window_size)/energy_window_size, mode='same')
    dynamic_energy_threshold = np.percentile(energy, energy_percentile)
    active = (energy > dynamic_energy_threshold)

    if np.any(active):
        start_idx = np.argmax(active)
        end_idx = len(active) - np.argmax(active[::-1])
    else:
        start_idx, end_idx = 0, N_total
    
    trimmed_data = data[start_idx:end_idx]
    
    return trimmed_data

class SwingDataset(Dataset):
    def __init__(self, X, 
                 train = True, y = None):
        self.X = X
        self.train = train
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

def to_numpy(p: torch.Tensor):
    if p.requires_grad:
        return p.detach().cpu().numpy()
    else:
        return p.cpu().numpy()
