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
from alan.dataloader import TrajectoryDataset


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

data_dir = "outs/out30"


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

df = pd.read_csv(TRAIN_INFO)[TrajectoryDataset.REQUIRED_FIELDS]
unique_player = df.drop_duplicates(subset=["player_id"])

player_id = 1
train_player_ids = [pid for pid in unique_player if pid != player_id]
# Get all player_ids except the current one
print(f"Validating on player_id: {player_id}")

train_dataset = TrajectoryDataset(
    TRAIN_DATA_DIR,
    df[df["player_id"].isin(train_player_ids)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=True,
    predicting_fields=PREDICTING_FIELDS,
    possible_values=POSSIBLE_VALUES,
)

valid_dataset = TrajectoryDataset(
    TEST_DATA_DIR,
    df[df["player_id"].isin(player_id)],
    train=True,
    max_seq_len=model_args.max_seq_len,
    patch_size=model_args.patch_size,
    use_scaler=True,
    scaler=train_dataset.scaler,
    predicting_fields=PREDICTING_FIELDS,
    possible_values=POSSIBLE_VALUES,

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

