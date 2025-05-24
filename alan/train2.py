import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TRAIN_DATA_DIR, TRAIN_INFO, PREDICTING_FIELDS, POSSIBLE_VALUES, NUM_WORKERS, TEST_INFO, TEST_DATA_DIR
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

device = "cuda:0"
model_args = GPTConfig(
    max_seq_len = 200,
    in_chans = 6,
    n_layer = 8,
    n_head = 8,
    n_embd = 64,
    patch_size = 16,
    dropout = 0.2,
    bias = False
)

batch_size = 32
learning_rate = 5e-3
weight_decay = 0.01
betas = (0.9, 0.95)
num_epochs = 500

use_scaler = True
out_dir = "outs/out28"
use_wandb = False

# play years macro f1 0.41
# level macro f1 0.50
# gender macro f1 0.69

PREDICTING_FIELDS = [
    # "gender",
    # "hold racket handed",
    "play years",
    # "level",
]

POSSIBLE_VALUES = [
    # [1, 2],
    # [1, 2],
    [0, 1, 2],
    # [2, 3, 4, 5],
]


wandb_configs = {
    "n_embd": model_args.n_embd,
    "n_layer": model_args.n_layer,
    "n_head": model_args.n_head,
    "in_chans": model_args.in_chans,
    "patch_size": model_args.patch_size,
    "max_seq_len": model_args.max_seq_len,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "betas": betas,
    "num_epochs": num_epochs,
    "use_scaler": use_scaler,
    "out_dir": out_dir,
}

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
        
        # Calculate class weights for the first predicting field (assumes single field classification)
        labels = dataframe[PREDICTING_FIELDS[0]].values
        class_counts = np.array([(labels == v).sum() for v in POSSIBLE_VALUES[0]])
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(POSSIBLE_VALUES[0])
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
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
        
        if seq.shape[0] <= self.max_seq_len:
            seq = np.pad(seq, ((0, self.max_seq_len - seq.shape[0]), (0, 0)), mode='constant', constant_values=0)
            if self.use_scaler:
                input = torch.tensor(self.scaler.transform(seq[:self.max_seq_len]), dtype=torch.float32)
            else:
                input = torch.tensor(seq[:self.max_seq_len], dtype=torch.float32)
            
            return input, item[1]
        
        segment_start = torch.randint(0, seq.shape[0] - (self.max_seq_len), (1,)).item()
        if self.use_scaler:
            input = torch.tensor(self.scaler.transform(seq[segment_start:segment_start + self.max_seq_len]), dtype=torch.float32)
        else:
            input = seq[segment_start:segment_start + self.max_seq_len].clone().detach()
        
        return input, item[1]


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
    max_seq_len=model_args.max_seq_len,
    use_scaler=use_scaler,
)
valid_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(valid_player_ids)],
    max_seq_len=model_args.max_seq_len,
    scaler=train_dataset.scaler if use_scaler else None,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

input, target = next(iter(valid_dataloader))
print(f"input shape: {input.shape}")
print(f"target shape: {target.shape}")
# exit()


from alan.rnn_model import SwingGRU
model = SwingGRU(
    input_dim=6,
    d_model=32,
    output_dim=len(POSSIBLE_VALUES[0]),
    dropout=0.2,
    weight=train_dataset.class_weights.to(device),
)

optimizer = model.configure_optimizers(learning_rate=learning_rate, weight_decay=weight_decay, betas=betas)
model = model.to(device)

os.makedirs(out_dir, exist_ok=True)

best_score = 0.0
best_model = deepcopy(model.state_dict())

for epoch in range(num_epochs):
    total_train_loss = 0.0
    train_seen_items = 0
    total_valid_loss = 0.0
    val_seen_items = 0
    
    model.train()
    all_logits = []
    all_targets = []
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for i, (input, target) in enumerate(pbar):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        logits, loss = model(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()
        train_seen_items += input.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())
        # if i % 50 == 0 and use_wandb:
        #     wandb.log({
        #         "train_loss": loss.item(),
        #     })
    
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    
    all_logits = all_logits.argmax(axis=1)
    all_targets = all_targets.argmax(axis=1)
    report = classification_report(
        all_targets,
        all_logits,
        target_names=[str(v) for v in POSSIBLE_VALUES[0]],
    )
    # print(report)
    acc_score = accuracy_score(all_targets, all_logits)
    f1 = f1_score(all_targets, all_logits, average='macro')
    # print(f"Accuracy: {acc_score:.4f}, F1 Score: {f1:.4f}")
        
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {total_train_loss / len(train_dataloader):.4f}")

    model.eval()
    all_logits = []
    all_targets = []
    pbar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    with torch.no_grad():
        for i, (input, target) in enumerate(pbar):
            input, target = input.to(device), target.to(device)
            logits, loss = model(input, target)
            total_valid_loss += loss.item()
            val_seen_items += input.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    
    
    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    
    roc_auc = roc_auc_score(all_targets, all_logits, multi_class='ovr', average='micro')
    
    all_logits = all_logits.argmax(axis=1)
    all_targets = all_targets.argmax(axis=1)

    report = classification_report(
        all_targets,
        all_logits,
        target_names=[str(v) for v in POSSIBLE_VALUES[0]],
    )
    # print(report)
    acc_score = accuracy_score(all_targets, all_logits)
    f1 = f1_score(all_targets, all_logits, average='macro')
    print(f"Accuracy: {acc_score:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    val_loss = total_valid_loss / len(valid_dataloader)

    if f1 > best_score:
        best_score = f1
        best_model = deepcopy(model.state_dict())
        print(f"Best model saved with score: {f1:.4f}, loss: {total_valid_loss / len(valid_dataloader):.4f}")
        torch.save(model.state_dict(), f"{out_dir}/model_{PREDICTING_FIELDS[0]}_f1{f1:.4f}_roc{roc_auc:.4f}_loss{val_loss:.3f}.pth")

    print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {val_loss:.4f}")

if use_wandb:
    wandb.finish()
print(f"Best model saved with score: {best_score:.4f}")
torch.save(best_model, f"{out_dir}/model_{PREDICTING_FIELDS[0]}_f1{f1:.4f}_roc{roc_auc:.4f}_loss{val_loss:.3f}.pth")
# torch.save(model.state_dict(), f"{out_dir}/model.pth")
