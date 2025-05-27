import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from helper.segment import Trim
from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

class TrajectoryDataset(Dataset):

    # REQUIRED_FIELDS = ["unique_id", "player_id"] + PREDICTING_FIELDS

    def __init__(
        self,
        data_dir: Path,
        dataframe: pd.DataFrame,
        max_seq_len: int = 512,
        patch_size: int = 32,
        use_scaler: bool = False,
        train: bool = True,
        label: bool = False,
        sample_weights: bool = True,
        detrend_and_filter: bool = True,
        scaler: StandardScaler = None,
        encoder: List[LabelEncoder] = None,
        predicting_fields: List[str] = None,
    ):
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.label = label
        self.use_scaler = use_scaler
        self.encoders = encoder
        trim_method = Trim()

        if train:
            df_to_encode = dataframe[predicting_fields]
            if self.encoders is None:
                self.encoders = [LabelEncoder().fit(df_to_encode[col]) for col in df_to_encode.columns]
            
            encoded = [enc.transform(df_to_encode[col])[:, None] for enc, col in zip(self.encoders, df_to_encode.columns)]
            encoded = np.concatenate(encoded, axis=1)
            metas = torch.tensor(encoded, dtype=torch.long)
            
            if sample_weights:
                cal_metas = metas[:, 1:]  # Exclude the first column (unique_id)
                unique_combinations, counts = np.unique(cal_metas.numpy(), axis=0, return_counts=True)
                weights = max(counts) / counts
                # weights = weights ** 0.5  # soften the weights by taking the square root
                weights = np.round(weights).astype(int)
                comb2weight = {tuple(comb): w for comb, w in zip(unique_combinations, weights)}
                sample_weights = np.array([comb2weight[tuple(meta.tolist())] for meta in cal_metas])
            else:
                sample_weights = np.ones((len(dataframe),), dtype=int)
        else:
            # a same len meta but all zeros
            metas = torch.zeros((len(dataframe),), dtype=torch.float32)
            sample_weights = np.ones((len(dataframe),), dtype=int)

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        fpaths = [
            data_dir / f"{unique_id}.txt" for unique_id in dataframe["unique_id"].values
        ]
        self.fpaths = fpaths
        
        all_features = []
        pbar = tqdm(zip(fpaths, metas, sample_weights), total=len(fpaths), desc="Loading data", leave=False)
        
        for fpath, meta, sample_weight in pbar:
            data = np.loadtxt(fpath)
            if detrend_and_filter:
                data = torch.tensor(detrend(data, axis=0), dtype=torch.float32) # trim method will not work with numpy arrays
            else:
                data = torch.tensor(data, dtype=torch.float32)
            data = trim_method(data)[0].numpy()
            if detrend_and_filter:
            # Apply low-pass filter
                data = self.butter_lowpass_filter(data, cutoff=30, fs=85, order=4)
            data = data.astype(np.float32)
            
            if use_scaler:
                all_features.append(data)
            else:
                mean = data.mean(axis=0)
                std = data.std(axis=0)
                data = (data - mean) / (std + 1e-8)
            
            if train:
                for _ in range(sample_weight):
                    self.samples.append((data, meta))
            else:
                self.samples.append((data, None))
        
        if use_scaler and scaler is None:
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

        seq_len_needed = self.max_seq_len + self.patch_size

        if seq.shape[0] <= seq_len_needed:
            seq = np.pad(seq, ((0, seq_len_needed - seq.shape[0]), (0, 0)), mode='constant', constant_values=0)

            if self.use_scaler:
                seq = self.scaler.transform(seq).astype(np.float32)

            if self.label:
                return seq[:self.max_seq_len], seq[self.patch_size:self.max_seq_len + self.patch_size], item[1]

            return seq[:self.max_seq_len], seq[self.patch_size:self.max_seq_len + self.patch_size]

        # Only call randint if the range is valid
        max_start = seq.shape[0] - seq_len_needed
        if max_start > 0:
            segment_start = torch.randint(0, max_start, (1,)).item()
        else:
            segment_start = 0

        if self.use_scaler:
            seq = self.scaler.transform(seq).astype(np.float32)

        if self.label:
            return seq[segment_start:segment_start + self.max_seq_len], seq[segment_start + self.patch_size:segment_start + self.max_seq_len + self.patch_size], item[1]

        return seq[segment_start:segment_start + self.max_seq_len], seq[segment_start + self.patch_size:segment_start + self.max_seq_len + self.patch_size]
