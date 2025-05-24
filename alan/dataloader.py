import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import TRAIN_DATA_DIR, TRAIN_INFO, PREDICTING_FIELDS, POSSIBLE_VALUES, NUM_WORKERS, TEST_INFO, TEST_DATA_DIR
from helper.segment import Trim
from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    """
    每筆 item 回傳 (segment, meta)，皆為 torch.Tensor:
    - seg  : shape=(L, 6), dtype=torch.float32
    - meta : one-hot vector for [gender, hand, years, level], dtype=torch.float32

    Note:
    - dataframe: should contain columns `['unique_id', 'player_id'] + PREDICTING_FIELDS`
    """

    REQUIRED_FIELDS = ["unique_id", "player_id"] + PREDICTING_FIELDS

    def __init__(
        self,
        data_dir: Path,
        dataframe: pd.DataFrame,
        max_seq_len: int = 512,
        patch_size: int = 32,
        use_scaler: bool = False,
        train: bool = True,
        label: bool = False,
        scaler: StandardScaler = None,
        predicting_fields: List[str] = None,
        possible_values: List[List[int]] = None,
    ):
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.label = label
        self.use_scaler = use_scaler
        trim_method = Trim()

        if train:
            df_to_encode = dataframe[PREDICTING_FIELDS]
            encoder = OneHotEncoder(categories=POSSIBLE_VALUES, sparse_output=False)
            metas = torch.tensor(encoder.fit_transform(df_to_encode), dtype=torch.float32)
        else:
            # a same len meta but all zeros
            metas = torch.zeros((len(dataframe),), dtype=torch.float32)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        fpaths = [
            data_dir / f"{unique_id}.txt" for unique_id in dataframe["unique_id"].values
        ]
        
        all_features = []
        pbar = tqdm(zip(fpaths, metas), total=len(fpaths), desc="Loading data", leave=False)
        
        for fpath, meta in pbar:
            data = np.loadtxt(fpath)
            data = torch.tensor(detrend(data, axis=0), dtype=torch.float32)
            data = trim_method(data)[0].numpy()
            data = self.butter_lowpass_filter(data, cutoff=30, fs=85, order=4)
            data = torch.tensor(data.copy(), dtype=torch.float32)
            
            if use_scaler:
                all_features.append(data)
            else:
                mean = data.mean(axis=0)
                std = data.std(axis=0)
                data = (data - mean) / (std + 1e-8)
            
            if train:
                self.samples.append((data, meta))
            else:
                self.samples.append((data, None))
        
        if use_scaler and train:
            all_features = np.concatenate(all_features)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
        else:
            self.scaler = scaler
    
    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        seq = item[0]
        
        if seq.shape[0] <= self.max_seq_len + self.patch_size:
            seq = np.pad(seq, ((0, self.max_seq_len + self.patch_size - seq.shape[0]), (0, 0)), mode='constant', constant_values=0)
            if self.use_scaler:
                input = torch.tensor(self.scaler.transform(seq[:self.max_seq_len]), dtype=torch.float32)
                target = torch.tensor(self.scaler.transform(seq[self.patch_size:self.max_seq_len + self.patch_size]), dtype=torch.float32)
            else:
                input = torch.tensor(seq[:self.max_seq_len], dtype=torch.float32)
                target = torch.tensor(seq[self.patch_size:self.max_seq_len + self.patch_size], dtype=torch.float32)
            
            if self.label:
                return input, target, item[1]

            return input, target
        
        segment_start = torch.randint(0, seq.shape[0] - (self.max_seq_len + self.patch_size), (1,)).item()
        if self.use_scaler:
            input = torch.tensor(self.scaler.transform(seq[segment_start:segment_start + self.max_seq_len]), dtype=torch.float32)
            target = torch.tensor(self.scaler.transform(seq[segment_start + self.patch_size:segment_start + self.max_seq_len + self.patch_size]), dtype=torch.float32)
        else:
            input = seq[segment_start:segment_start + self.max_seq_len].clone().detach()
            target = seq[segment_start + self.patch_size:segment_start + self.max_seq_len + self.patch_size].clone().detach()
        
        if self.label:
            return input, target, item[1]
        
        return input, target
