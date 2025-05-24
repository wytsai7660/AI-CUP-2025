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
from torchvision.transforms import Compose
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from tqdm import tqdm
from alan.gpt import GPT, GPTConfig
import matplotlib.pyplot as plt
import os
import wandb


MAX_SEQ_LEN = 512
PATCH_SIZE = 16
device = "cuda:1"

PREDICTING_FIELDS = [
    # "gender",
    "hold racket handed",
    # "play years",
    # "level",
]

POSSIBLE_VALUES = [
    # [1, 2],
    [1, 2],
    # [0, 1, 2],
    # [2, 3, 4, 5],
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
        train: bool = True,
        label: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.label = label
        trim_method = Trim()
        # self.rng = torch.Generator()
        # self.rng.manual_seed(42)

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
        
        # all_features = []
        pbar = tqdm(zip(fpaths, metas), total=len(fpaths), desc="Loading data", leave=False)
        
        for fpath, meta in pbar:
            data = np.loadtxt(fpath)
            data = torch.tensor(detrend(data, axis=0), dtype=torch.float32)
            data = trim_method(data)[0].numpy()
            data = self.butter_lowpass_filter(data, cutoff=30, fs=85, order=4)
            data = torch.tensor(data.copy(), dtype=torch.float32)
            # normalize along accelerometer and gyroscope axes
            # do it per example to better capture var instead of absolute values
            acc_data_mean = data[:, 0:3].mean(axis=0)
            acc_data_std = data[:, 0:3].std(axis=0)
            gyro_data_mean = data[:, 3:6].mean(axis=0)
            gyro_data_std = data[:, 3:6].std(axis=0)
            data[:, 0:3] = (data[:, 0:3] - acc_data_mean) / (acc_data_std + 1e-8)
            data[:, 3:6] = (data[:, 3:6] - gyro_data_mean) / (gyro_data_std + 1e-8)
            
            if train:
                self.samples.append((data, meta))
            else:
                self.samples.append((data, None))
            # all_features.append(data)
        
        # all_features = np.concatenate(all_features)
        # self.scaler = StandardScaler()
        # self.scaler.fit(all_features)
    
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
        
        if len(seq) < self.max_seq_len + PATCH_SIZE:
            seq = np.pad(seq, ((0, self.max_seq_len + PATCH_SIZE - len(seq)), (0, 0)), mode='constant', constant_values=0)
            # input = torch.tensor(self.scaler.transform(seq[:self.max_seq_len]), dtype=torch.float32)
            # target = torch.tensor(self.scaler.transform(seq[PATCH_SIZE:self.max_seq_len + PATCH_SIZE]), dtype=torch.float32)
            input = torch.tensor(seq[:self.max_seq_len], dtype=torch.float32)
            target = torch.tensor(seq[PATCH_SIZE:self.max_seq_len + PATCH_SIZE], dtype=torch.float32)
            
            if self.label:
                return input, target, item[1]

            return input, target
        
        segment_start = torch.randint(0, seq.shape[0] - (MAX_SEQ_LEN + PATCH_SIZE), (1,)).item()
        input = seq[segment_start:segment_start + MAX_SEQ_LEN].clone().detach()
        target = seq[segment_start + PATCH_SIZE:segment_start + MAX_SEQ_LEN + PATCH_SIZE].clone().detach()
        # input = torch.tensor(self.scaler.transform(seq[segment_start:segment_start + MAX_SEQ_LEN]), dtype=torch.float32)
        # target = torch.tensor(self.scaler.transform(seq[segment_start + PATCH_SIZE:segment_start + MAX_SEQ_LEN + PATCH_SIZE]), dtype=torch.float32)
        if self.label:
            return input, target, item[1]
        
        return input, target


df = pd.read_csv(TRAIN_INFO)[TrajectoryDataset.REQUIRED_FIELDS]
unique_player = df.drop_duplicates(subset=["player_id"])

train_player_ids, valid_player_ids = train_test_split(
    unique_player["player_id"].to_numpy(),
    test_size=0.2,
    random_state=42,
    stratify=unique_player['hold racket handed'].to_numpy(),
)


wandb_config = {
    "batch_size": 32,
    "learning_rate": 5e-3,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 128,
    "patch_size": 16,
    "dropout": 0.2,
    "bias": False,
    "block_size": (512 // 16),
    "in_chans": 6,
    "n_epoch": 500,
    "detrending": True,
    "lowpass_cutoff": 30,
    "lowpass_order": 4,
}
out_dir = "out6"
use_wandb = True

if use_wandb:
    wandb.init(project="imugpt-experiments", config=wandb_config)

train_df = pd.read_csv(TRAIN_INFO)[TrajectoryDataset.REQUIRED_FIELDS]
test_df = pd.read_csv(TEST_INFO)[["unique_id"]]

train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(train_player_ids)],
    train=True,
)
valid2_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(valid_player_ids)],
    train=True,
)
valid_dataset = TrajectoryDataset(
    TEST_DATA_DIR,
    test_df,
    train=False,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
valid2_dataloader = DataLoader(
    valid2_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


input, target = next(iter(valid_dataloader))

model = GPT(
    GPTConfig(
        block_size = (512 // 16),
        in_chans = 6,
        n_layer = 8,
        n_head = 8,
        n_embd = 128,
        patch_size = 16,
        dropout = 0.2,
        bias = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
)

n_epoch = 500
optimizer = GPT.configure_optimizers(model, learning_rate=5e-3, weight_decay=0.01, betas=(0.9, 0.95), device_type=device)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
model = model.to(device)

os.makedirs(out_dir, exist_ok=True)

for epoch in range(n_epoch):
    total_train_loss = 0.0
    train_seen_items = 0
    total_valid_loss = 0.0
    val_seen_items = 0
    
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", leave=False)
    for i, (input, target) in enumerate(pbar):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        logits, loss = model(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()
        train_seen_items += input.size(0)
        if i % 50 == 0 and use_wandb:
            wandb.log({
                "train_loss": loss.item(),
            })

    
    # scheduler.step()
    
    print(f"Epoch {epoch+1}/{n_epoch} - Train Loss: {total_train_loss / len(train_dataloader):.4f}")

    model.eval()
    pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", leave=False)
    with torch.no_grad():
        for i, (input, target) in enumerate(pbar):
            input, target = input.to(device), target.to(device)
            logits, loss = model(input, target)
            total_valid_loss += loss.item()
            val_seen_items += input.size(0)
            # pbar.set_postfix(loss=loss.item())
            if i % 50 == 0 and use_wandb:
                wandb.log({
                    "valid_loss": loss.item(),
                })

        
        if epoch % 50 == 0 or epoch == n_epoch - 1:
            plt.figure(figsize=(12, 8))
            
            print(logits.shape, target.shape)
            # logits: (B, num_patches, patch_size * in_chans)
            B, num_patches, patch_dim = logits.shape
            logits = logits.view(B, num_patches, 6, PATCH_SIZE)  # (B, num_patches, C, patch_size)
            logits = logits.permute(0, 2, 1, 3)  # (B, C, num_patches, patch_size)
            logits = logits.contiguous().view(B, 6, -1)  # (B, C, T) where T = num_patches * patch_size
            logits = logits.permute(0, 2, 1)  # (B, T, C)
            logits = logits.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            print(logits.shape, target.shape)
            
            for i in range(6):
                plt.subplot(3, 2, i+1)
                plt.plot(target[0, :, i], 'b-', label='Ground Truth')
                plt.plot(logits[0, :, i], 'r-', label='Prediction')
                plt.title(f'Channel {i+1}')
                if i == 0:
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{out_dir}/prediction_epoch_{epoch+1}.png')
            plt.close()
    
    print(f"Epoch {epoch+1}/{n_epoch} - Valid Loss: {total_valid_loss / len(valid_dataloader):.4f}")
    
    total_valid_loss = 0.0
    pbar = tqdm(valid2_dataloader, desc=f"Epoch {epoch+1}/{n_epoch}", leave=False)
    with torch.no_grad():
        for i, (input, target) in enumerate(pbar):
            input, target = input.to(device), target.to(device)
            logits, loss = model(input, target)
            total_valid_loss += loss.item()
            # pbar.set_postfix(loss=loss.item())
            if i % 50 == 0 and use_wandb:
                wandb.log({
                    "valid_level_loss": loss.item(),
                })
                
    print(f"Epoch {epoch+1}/{n_epoch} - Valid2 Loss: {total_valid_loss / len(valid2_dataloader):.4f}")

wandb.finish()
torch.save(model.state_dict(), f"{out_dir}/model.pth")