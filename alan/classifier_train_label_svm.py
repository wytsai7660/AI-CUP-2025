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

batch_size = 32
learning_rate = 2e-3
weight_decay = 0.001
betas = (0.9, 0.95)
num_epochs = 500

use_scaler = True
data_dir = "outs/out23"
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


# df = pd.read_csv(TRAIN_INFO)[TrajectoryDataset.REQUIRED_FIELDS]
# unique_player = df.drop_duplicates(subset=["player_id"])


# train_dataset = TrajectoryDataset(
#     TRAIN_DATA_DIR,
#     df[df["player_id"].isin(train_player_ids)],
# )
# valid_dataset = TrajectoryDataset(
#     TRAIN_DATA_DIR,
#     df[df["player_id"].isin(valid_player_ids)],
# )

# if use_wandb:
#     wandb.init(project="imugpt-experiments", config=wandb_configs)

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

def plot():
    pca = PCA(n_components=2)
    embedding_2d = pca.fit_transform(all_embedding)


    import matplotlib.colors as mcolors
    bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    norm = mcolors.BoundaryNorm(bounds, plt.cm.tab10.N)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=all_label, cmap='tab10', alpha=0.7, norm=norm # Apply the norm here
    )
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA of Embeddings')

    # Create the colorbar using the same norm and explicitly set the ticks to the actual label values
    cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])
    cbar.set_label('Class')
    # No need to set ticks again if they are passed directly to plt.colorbar

    plt.tight_layout()
    plt.savefig("emb_pca.png")
    plt.close()


model = GPT(model_args)
model.load_state_dict(torch.load(f'{data_dir}/model.pth'))
model = model.to(device)

model.eval()



all_embedding = []
all_label = []
pbar = tqdm(train_dataloader, desc=f"make trainset", leave=False)
with torch.no_grad():
    # for _ in range(5):
    for i, (input, target, label) in enumerate(pbar):
        input = input.to(device)
        embedding = model(input)
        embedding = torch.mean(embedding, dim=1)
        embedding = embedding.detach().cpu().numpy()
        all_embedding.append(embedding)
        label_indices = torch.argmax(label, dim=1).cpu().numpy()
        all_label.append(label_indices)

all_embedding = np.concatenate(all_embedding)
all_label = np.concatenate(all_label)
all_label = all_label.astype(int)
print(all_embedding.shape)
print(all_label.shape)

all_test_embedding = []
all_test_label = []
pbar = tqdm(valid_dataloader, desc=f"make trainset", leave=False)
with torch.no_grad():
    for i, (input, target, label) in enumerate(pbar):
        input = input.to(device)
        embedding = model(input)
        embedding = torch.mean(embedding, dim=1)
        embedding = embedding.detach().cpu().numpy()
        all_test_embedding.append(embedding)
        label_indices = torch.argmax(label, dim=1).cpu().numpy()
        all_test_label.append(label_indices)

all_test_embedding = np.concatenate(all_test_embedding)
all_test_label = np.concatenate(all_test_label)
all_test_label = all_test_label.astype(int)
print(all_test_embedding.shape)
print(all_test_label.shape)



from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import joblib

svm = SVC(C=1.0, random_state=42, class_weight='balanced', probability=True)
svm.fit(all_embedding, all_label)
y_pred = svm.predict(all_test_embedding)
y_pred_proba = svm.predict_proba(all_test_embedding)
print(y_pred_proba.shape, all_test_label.shape)
print("SVM Classification Report:")
print(classification_report(all_test_label, y_pred))
print("SVM Confusion Matrix:")
print(confusion_matrix(all_test_label, y_pred))
print("SVM Accuracy:", accuracy_score(all_test_label, y_pred))
print("SVM F1 Score:", f1_score(all_test_label, y_pred, average='macro'))
print("SVM ROC AUC Score:", roc_auc_score(all_test_label, y_pred_proba, multi_class='ovr', average='micro'))

# Save the trained SVM model
svm_model_path = os.path.join(data_dir, "svm_model.joblib")
joblib.dump(svm, svm_model_path)
print(f"SVM model saved to {svm_model_path}")

# data, meta = train_dataset.samples[0]
# if use_scaler:
#     data = train_dataset.scaler.transform(data)
#     # data = torch.tensor(data, dtype=torch.float32)
# print(f"data shape: {data.shape}, meta shape: {meta.shape}")

# all_embedding = []
# for start in range(0, data.shape[0] - model_args.max_seq_len):
#     segment = torch.tensor(data[start:start + model_args.max_seq_len], dtype=torch.float32)
#     # if segment.shape[0] < model_args.max_seq_len:
#     #     continue
#     segment = segment.unsqueeze(0).to(device)
#     embedding = model(segment)
#     embedding = torch.mean(embedding, dim=1).detach().cpu().numpy()
#     all_embedding.append(embedding)
    
# all_embedding = np.concatenate(all_embedding)
# print(f"all_embedding shape: {all_embedding.shape}")
# label = meta.argmax().item()
# print(f"label: {label}")
# probs = svm.predict_proba(all_embedding)

# plt.figure(figsize=(12, 6))
# for i in range(probs.shape[1]):
#     plt.plot(probs[:, i], label=f'Class {i}')
# plt.xlabel('Segment (T)')
# plt.ylabel('Predicted Probability')
# plt.title('SVM Predicted Probabilities, ground Truth: ' + str(label))
# plt.legend()
# plt.tight_layout()
# plt.savefig("segment_probs.png")
# plt.close()

# exit(0)
