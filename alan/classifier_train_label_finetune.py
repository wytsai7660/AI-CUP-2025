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
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score



device = "cuda:1"
model_args = GPTConfig(
    max_seq_len = 1024,
    in_chans = 6,
    n_layer = 8,
    n_head = 8,
    n_embd = 64,
    patch_size = 16,
    dropout = 0.2,
    bias = False
)

batch_size = 32
learning_rate = 5e-5
weight_decay = 5e-5
betas = (0.9, 0.95)
num_epochs = 500

use_scaler = True
data_dir = "outs/out12"
use_wandb = False



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
                
        labels = dataframe[PREDICTING_FIELDS[0]].values
        class_counts = np.array([(labels == v).sum() for v in POSSIBLE_VALUES[0]])
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(POSSIBLE_VALUES[0])
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        
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
    stratify=unique_player[PREDICTING_FIELDS[0]].to_numpy(),
)

train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(train_player_ids)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    label=True,
)

valid_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(valid_player_ids)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=use_scaler,
    scaler=train_dataset.scaler if use_scaler else None,
    label=True,
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

# def plot():
#     pca = PCA(n_components=2)
#     embedding_2d = pca.fit_transform(all_embedding)


#     import matplotlib.colors as mcolors
#     bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
#     norm = mcolors.BoundaryNorm(bounds, plt.cm.tab10.N)

#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(
#         embedding_2d[:, 0], embedding_2d[:, 1],
#         c=all_label, cmap='tab10', alpha=0.7, norm=norm # Apply the norm here
#     )
#     plt.xlabel('PCA 1')
#     plt.ylabel('PCA 2')
#     plt.title('PCA of Embeddings')

#     # Create the colorbar using the same norm and explicitly set the ticks to the actual label values
#     cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])
#     cbar.set_label('Class')
#     # No need to set ticks again if they are passed directly to plt.colorbar

#     plt.tight_layout()
#     plt.savefig("emb_pca.png")
#     plt.close()



class GPTClassifier(nn.Module):
    def __init__(self, model_args: GPTConfig, output_dim: int = 4, weight=None):
        super().__init__()
        self.model = GPT(model_args)
        self.model.load_state_dict(torch.load(f'{data_dir}/model.pth'))
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        for block in self.model.transformer.h[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True


        # self.attention_net = nn.Sequential(
        #     nn.Linear(model_args.n_embd, model_args.n_embd // 2),
        #     nn.GELU(),
        #     nn.Linear(model_args.n_embd // 2, 1),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(model_args.n_embd, model_args.n_embd // 2),
            nn.LayerNorm(model_args.n_embd // 2),
            nn.GELU(),
            nn.Dropout(model_args.dropout),
            nn.Linear(model_args.n_embd // 2, output_dim)
        )
        
        self.criterion = nn.CrossEntropyLoss(weight = weight)
        
    def forward(self, x, target=None):
        x = self.model(x) # (batch_size, seq_len, n_embd)
        # attention_scores = self.attention_net(x)
        # attention_weights = F.softmax(attention_scores, dim=1)
        # x = torch.sum(x * attention_weights, dim=1) # (batch_size, n_embd)
        x = x[:, 8:, :].mean(dim=1) # (batch_size, n_embd)
        x = self.classifier(x)
        if target is None:
            return x, None
        else:
            loss = self.criterion(x, target)
            return x, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

model = GPTClassifier(model_args, output_dim=len(POSSIBLE_VALUES[0]), weight=train_dataset.class_weights.to(device))
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
model.to(device)
# exit(0)

best_score = 0.0
best_model = None
for epoch in range(num_epochs):
    total_train_loss = 0.0
    train_seen_items = 0
    total_valid_loss = 0.0
    val_seen_items = 0
    
    model.train()
    all_logits = []
    all_targets = []
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for i, (input, _, target) in enumerate(pbar):
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
        for i, (input, _, target) in enumerate(pbar):
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
    print(report)
    acc_score = accuracy_score(all_targets, all_logits)
    f1 = f1_score(all_targets, all_logits, average='macro')
    print(f"Accuracy: {acc_score:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    val_loss = total_valid_loss / len(valid_dataloader)

    if f1 > best_score:
        best_score = f1
        best_model = deepcopy(model.state_dict())
        print(f"Best model saved with score: {f1:.4f}, loss: {total_valid_loss / len(valid_dataloader):.4f}")
        torch.save(model.state_dict(), f"{data_dir}/model_{PREDICTING_FIELDS[0]}_f1{f1:.4f}_roc{roc_auc:.4f}_loss{val_loss:.3f}.pth")

    print(f"Epoch {epoch+1}/{num_epochs} - Valid Loss: {val_loss:.4f}")
