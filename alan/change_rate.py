import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TRAIN_DATA_DIR, TRAIN_INFO, NUM_WORKERS, TEST_INFO, TEST_DATA_DIR
from helper.segment import Trim, Segment
from scipy.signal import detrend
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from tqdm import tqdm
from alan.gpt import GPT, GPTConfig
import matplotlib.pyplot as plt
import os
import wandb
from copy import deepcopy
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


device = "cuda:0"
model_args = GPTConfig(
    max_seq_len = 1024,
    in_chans = 6,
    n_layer = 8,
    n_head = 8,
    n_embd = 128,
    patch_size = 8,
    dropout = 0.2,
    bias = False
)

data_dir = "outs/out23"


PREDICTING_FIELDS = [
    # "gender",
    # "hold racket handed",
    # "play years",
    "level",
]

POSSIBLE_VALUES = [
    # [1, 2],
    # [1, 2],
    # [0, 1, 2],
    [2, 3, 4, 5],
]

print(f"preparing dataset...")

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
    ):
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.label = label
        self.use_scaler = use_scaler
        self.scaler = scaler
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
        
        if use_scaler and self.scaler is None:
            all_features = np.concatenate(all_features)
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
    
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

df = pd.read_csv(TRAIN_INFO)[TrajectoryDataset.REQUIRED_FIELDS]
unique_player = df.drop_duplicates(subset=["player_id"])

train_player_ids, valid_player_ids = train_test_split(
    unique_player["player_id"].to_numpy(),
    test_size=0.2,
    random_state=42,
    # stratify=unique_player[PREDICTING_FIELDS[0]].to_numpy(),
)

train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(train_player_ids)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=True,
    label=True,
)

valid_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(valid_player_ids)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=True,
    scaler=train_dataset.scaler,
    label=True,
)

model = GPT(model_args)
model.load_state_dict(torch.load(f'{data_dir}/model.pth'))
model = model.to(device)
model.eval()

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import joblib


svm: SVC = joblib.load(f"{data_dir}/svm_model.joblib")
for i in range(len(valid_dataset)):
    data, meta = valid_dataset.samples[i]
    if meta[1] == 1:  # only use level 2
        break
print(meta)
# data, meta = valid_dataset.samples[250]
data = train_dataset.scaler.transform(data)
# data = torch.tensor(data, dtype=torch.float32)
print(f"data shape: {data.shape}, meta shape: {meta.shape}")

all_embedding = []
for start in range(0, data.shape[0] - model_args.max_seq_len):
    segment = torch.tensor(data[start:start + model_args.max_seq_len], dtype=torch.float32)
    # if segment.shape[0] < model_args.max_seq_len:
    #     continue
    segment = segment.unsqueeze(0).to(device)
    embedding = model(segment)
    embedding = torch.mean(embedding, dim=1).detach().cpu().numpy()
    all_embedding.append(embedding)
    
all_embedding = np.concatenate(all_embedding)
print(f"all_embedding shape: {all_embedding.shape}")
label = meta.argmax().item()
print(f"label: {label}")
probs = svm.predict_proba(all_embedding)

plt.figure(figsize=(12, 6))
for i in range(probs.shape[1]):
    plt.plot(probs[:, i], label=f'Class {i}')
plt.xlabel('Segment (T)')
plt.ylabel('Predicted Probability')
plt.title('SVM Predicted Probabilities, ground Truth: ' + str(label))
plt.legend()
plt.tight_layout()
plt.savefig("segment_probs.png")
plt.close()

