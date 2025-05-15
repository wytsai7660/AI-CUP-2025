import warnings
warnings.simplefilter('ignore')
import math
import pandas as pd
import numpy as np
import sys
import time
import datetime
from contextlib import contextmanager
import logging
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold,StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import LayerNorm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import lr_scheduler, AdamW
from sklearn.preprocessing import StandardScaler, RobustScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import gc
import random
import os
from types import SimpleNamespace
import logging
import wandb
import joblib
import csv
from copy import deepcopy
import argparse
import yaml
from utils import *
from models import SwingGRU, EncoderOnlyClassifier

pd.set_option('display.max_columns', 300)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str, required=True,
                    help='Path to the YAML file')
parser.add_argument('--wandb', action='store_true',
                    help='Whether to use wandb')
parser.add_argument('--device', type=str, default="cuda:0",
                    help='Cuda device to use')

cmd_args = parser.parse_args()
config_path = cmd_args.config
use_wandb = cmd_args.wandb
device = cmd_args.device

if not os.path.exists(config_path):
    print(f"Configuration file not found at {config_path}")
    exit(1)

try:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        args = SimpleNamespace(**config_data)
except yaml.YAMLError as e:
    print(f"Error decoding YAML: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)

if use_wandb:
    wandb_config = vars(args)
    wandb.init(project="table-tennis-experiments", name=args.name, config=wandb_config)

# ========================= preparing dataset ============================
print(f"preparing dataset...")
df = pd.read_csv('../data/train/train_info.csv')

selected_columns_df = df[['player_id', 'gender', 'hold racket handed', 'play years', 'level']]
unique_players_info = selected_columns_df.drop_duplicates(subset=['player_id'], keep='first')
sorted_unique_players_info = unique_players_info.sort_values(by='player_id', ascending=True)

train_indices, val_indices = train_test_split(
    sorted_unique_players_info['player_id'].to_numpy(),
    test_size=0.2,
    random_state=42,
    stratify=sorted_unique_players_info[args.split_type].to_numpy()
)

all_features = []
for index, row in df.iterrows():
    unique_id = row['unique_id']
    player_id = row['player_id']
    mode = row['mode'] - 1
    gender = row['gender'] - 1
    hand = row['hold racket handed'] - 1
    year = row['play years']
    level = row['level'] - 2
    if player_id in val_indices:
        continue
    
    filepath = f'../data/train/train_data/{unique_id}.txt'
    imu_data = np.loadtxt(filepath)
    imu_data = extract_valid_swing(imu_data)
    batch = len(imu_data) // args.shift
    
    if args.use_aug_features == "none":
        feature_arr_ = np.zeros((batch, args.seq_len, 6))
        all_features.append(imu_data)
    elif args.use_aug_features == "basic":
        feature_arr_ = np.zeros((batch, args.seq_len, 24))
        diff_prev = np.diff(imu_data, axis=0, prepend=imu_data[:1])
        diff_next = np.diff(imu_data, axis=0, append=imu_data[-1:])
        cumsum = np.cumsum(imu_data, axis=0)
        all_feature = np.concatenate((imu_data, diff_prev, diff_next, cumsum), axis=1)
        all_features.append(all_feature)


all_features = np.concatenate(all_features)
print(all_features.shape)
sc = RobustScaler()
sc.fit(all_features)

partial_train_feature = []
partial_train_target = []
partial_val_feature = []
partial_val_target = []

for index, row in df.iterrows():
    unique_id = row['unique_id']
    player_id = row['player_id']
    mode = row['mode'] - 1
    gender = row['gender'] - 1
    hand = row['hold racket handed'] - 1
    year = row['play years']
    level = row['level'] - 2
    
    filepath = f'../data/train/train_data/{unique_id}.txt'
    imu_data = np.loadtxt(filepath)
    imu_data = extract_valid_swing(imu_data)
    
    batch = len(imu_data) // args.shift

    if args.use_aug_features == "none":
        feature_arr_ = np.zeros((batch, args.seq_len, 6))
        all_feature = sc.transform(imu_data)

    elif args.use_aug_features == "basic":
        feature_arr_ = np.zeros((batch, args.seq_len, 24))
        diff_prev = np.diff(imu_data, axis=0, prepend=imu_data[:1])
        diff_next = np.diff(imu_data, axis=0, append=imu_data[-1:])
        cumsum = np.cumsum(imu_data, axis=0)
        all_feature = np.concatenate((imu_data, diff_prev, diff_next, cumsum), axis=1)
        all_feature = sc.transform(all_feature)
    
    if args.use_mode_as_target:
        target_arr_ = np.zeros((batch, 21), dtype=int)
    else:
        target_arr_ = np.zeros((batch, 11), dtype=int)
    
    gender_one_hot = np.eye(2)[gender]
    hand_one_hot = np.eye(2)[hand]
    year_one_hot = np.eye(3)[year]
    level_one_hot = np.eye(4)[level]
    mode_one_hot = np.eye(10)[mode]
    
    if args.use_mode_as_target:
        target = np.concatenate((gender_one_hot, hand_one_hot, year_one_hot, level_one_hot, mode_one_hot))
    else:
        target = np.concatenate((gender_one_hot, hand_one_hot, year_one_hot, level_one_hot))
    
    for i in range(batch):
        if i == batch - 1:
            feature = all_feature[i*args.shift:]
            feature_arr_[i, :len(feature), :] = feature
            target_arr_[i, :len(feature)] = target
        else:
            feature_arr_[i, :, :] = all_feature[i*args.shift:i*args.shift+args.seq_len]
            target_arr_[i, :] = target
    
    if player_id in train_indices:
        partial_train_feature.append(feature_arr_)
        partial_train_target.append(target_arr_)
    else:
        partial_val_feature.append(feature_arr_)
        partial_val_target.append(target_arr_)
        
train_feature = np.concatenate(partial_train_feature, axis=0)
train_target = np.concatenate(partial_train_target, axis=0)
val_feature = np.concatenate(partial_val_feature, axis=0)
val_target = np.concatenate(partial_val_target, axis=0)

train_feature = train_feature.astype(np.float32)
val_feature = val_feature.astype(np.float32)
train_target = train_target.astype(np.float32)
val_target = val_target.astype(np.float32)

# ========================= preparing dataset ============================

# ========================= define classes ============================

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

# ========================= define classes ============================
# ========================= define funcs ============================

def metric_report(y_batch: torch.Tensor, out_batch: torch.Tensor, final=False):
    cut = [0, 2, 4, 7, 11]
    classes = ['gender', 'hand', 'year', 'level']
    metrics = {}
    for start, end, cls in zip(cut, cut[1:], classes):
        y_pred = out_batch[:, start:end]
        y_true = y_batch[:, start:end]
        
        micro_roc_score = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
        macro_roc_score = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        acc_score = accuracy_score(y_true, y_pred)
        micro_f1_score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        macro_f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics[f'{cls[0]}_micro_roc'] = micro_roc_score
        metrics[f'{cls[0]}_macro_roc'] = macro_roc_score
        metrics[f'{cls[0]}_acc'] = acc_score
        metrics[f'{cls[0]}_micro_f1'] = micro_f1_score
        metrics[f'{cls[0]}_macro_f1'] = macro_f1_score
        print(f"{cls} micro roc {micro_roc_score:.4f} macro roc: {macro_roc_score:.4f} acc: {acc_score:.4f} micro f1: {micro_f1_score:.4f} macro f1: {macro_f1_score:.4f}")

    if use_wandb and final:
        wandb.log(metrics)


def get_things(train_target, train_loader, args):
    if args.model_type == 'gru':
        model = SwingGRU(args)
    elif args.model_type == 'encoder':
        model = EncoderOnlyClassifier(args)
    else:
        raise ValueError(f"no such model type {args.model_type}")

    model = model.to(device)
    
    train_weights = 1 / np.sum(train_target, axis=0)
    
    criterions = []
    if args.use_mode_as_target:
        cut = [0, 2, 4, 7, 11, 21]
        classes = ['gender', 'hand', 'year', 'level', 'mode']
    else:
        cut = [0, 2, 4, 7, 11]
        classes = ['gender', 'hand', 'year', 'level']

    for start, end in zip(cut, cut[1:]):
        class_weights = train_weights[start:end] / np.sum(train_weights[start:end])
        partial_weights = torch.tensor(class_weights).to(device)
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=partial_weights)
        # elif args.loss_type == 'FL':
        #     criterion = FocalLoss(weight=partial_weights, gamma=args.FL_gamma)
        # elif loss_type == 'BCE':
        #     criterion = nn.BCEWithLogitsLoss(weight=partial_weights)
        else:
            raise ValueError(f"no such loss {args.loss_type}")
        criterions.append(criterion)
        
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        )
    num_train_optimization_steps = int(len(train_loader) * args.n_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    
    return model, criterions, optimizer, scheduler

# ========================= define funcs ============================

# ================================== train =================================
print(f"start training...")
set_seed(42)

train_ds = SwingDataset(train_feature, train=True, y = train_target)
val_ds = SwingDataset(val_feature, train=True, y = val_target)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=4)

model, criterions, optimizer, scheduler = get_things(train_target, train_loader, args)

X, y = next(iter(train_loader))
print(X.shape, y.shape)

best_loss = 9999.9
best_model = deepcopy(model.state_dict())
early_stop_cnt = 0

for epoch in range(args.n_epochs):
    train_out_batch = []
    val_out_batch = []
    train_y_batch = []
    val_y_batch = []
    train_loss = 0.0
    train_main_loss = 0.0
    train_aux_loss = 0.0
    val_loss = 0.0
    val_main_loss = 0.0
    val_aux_loss = 0.0
    
    model.train() 
    pbar = tqdm(train_loader, total=len(train_loader), leave = False)
    for i, d in enumerate(pbar):
        X, y = d
        b, t, c = X.size()
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        out = model(X)
        
        aux_loss = 0
        main_loss = 0
        aux_loss += criterions[0](out[:, 0:2], y[:, 0:2])
        aux_loss += criterions[1](out[:, 2:4], y[:, 2:4])
        aux_loss += criterions[2](out[:, 4:7], y[:, 4:7])
        main_loss += criterions[3](out[:, 7:11], y[:, 7:11])
        if args.use_mode_as_target:
            aux_loss += criterions[4](out[:, 11:21], y[:, 11:21])
        loss = (1 - args.main_loss_weight) * aux_loss + args.main_loss_weight * main_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            out[:, :2] = F.softmax(out[:, :2])
            out[:, 2:4] = F.softmax(out[:, 2:4])
            out[:, 4:7] = F.softmax(out[:, 4:7])
            out[:, 7:11] = F.softmax(out[:, 7:11])
        
        train_out_batch.append(to_numpy(out))
        train_y_batch.append(to_numpy(y))
        train_loss += loss.item()
        train_main_loss += main_loss.item()
        train_aux_loss += aux_loss.item()
        
        if i % 10 == 0 and use_wandb:
            wandb.log({
                # "epoch": epoch,
                "train_loss": loss.item(),
                "train_main_loss": main_loss.item(),
                "train_aux_loss": aux_loss.item(),
            })

    
    train_out_batch = np.concatenate(train_out_batch)
    train_y_batch = np.concatenate(train_y_batch)
    
    train_loss = train_loss/len(train_loader)
    train_main_loss = train_main_loss/len(train_loader)
    train_aux_loss = train_aux_loss/len(train_loader)
    print(f"Train loss: {train_loss:.4f} Aux loss: {train_aux_loss:.4f} Main loss: {train_main_loss:.4f}")
    
    model.eval()
    pbar = tqdm(val_loader, total=len(val_loader), leave = False)
    with torch.no_grad():
        for d in pbar:
            X, y = d
            X, y = X.to(device), y.to(device)
            
            out = model(X)
            
            aux_loss = 0
            main_loss = 0
            aux_loss += criterions[0](out[:, 0:2], y[:, 0:2])
            aux_loss += criterions[1](out[:, 2:4], y[:, 2:4])
            aux_loss += criterions[2](out[:, 4:7], y[:, 4:7])
            main_loss += criterions[3](out[:, 7:11], y[:, 7:11])
            if args.use_mode_as_target:
                aux_loss += criterions[4](out[:, 11:21], y[:, 11:21])
            loss = (1 - args.main_loss_weight) * aux_loss + args.main_loss_weight * main_loss

            with torch.no_grad():
                out[:, 0:2] = F.softmax(out[:, 0:2])
                out[:, 2:4] = F.softmax(out[:, 2:4])
                out[:, 4:7] = F.softmax(out[:, 4:7])
                out[:, 7:11] = F.softmax(out[:, 7:11])
            
            val_out_batch.append(to_numpy(out))
            val_y_batch.append(to_numpy(y))
            val_loss += loss.item()
            val_main_loss += main_loss.item()
            val_aux_loss += aux_loss.item()
            
            if i % 10 == 0 and use_wandb:
                wandb.log({
                    # "epoch": epoch,
                    "val_loss": loss.item(),
                    "val_main_loss": main_loss.item(),
                    "val_aux_loss": aux_loss.item(),
                })



    val_out_batch = np.concatenate(val_out_batch)
    val_y_batch = np.concatenate(val_y_batch)

    val_loss = val_loss/len(val_loader)
    val_main_loss = val_main_loss/len(val_loader)
    val_aux_loss = val_aux_loss/len(val_loader)
    print(f"Val loss: {val_loss:.4f} Aux loss: {val_aux_loss:.4f} Main loss: {val_main_loss:.4f}")
    metric_report(val_y_batch, val_out_batch)
    
    if val_main_loss < best_loss:
        best_loss = val_main_loss
        best_model = deepcopy(model.state_dict())
        print(f"✨ best Val loss at epoch {epoch}, loss {val_main_loss:.4f}")
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1
    
    if early_stop_cnt == 4:
        print(f"No imporvement for {early_stop_cnt} Epochs, stopping")
        break

# ================================== val ======================================

model.load_state_dict(best_model)
model.eval()
df = pd.read_csv('../data/train/train_info.csv')

y_true_batch = []
y_pred_batch = []

for index, row in df.iterrows():
    unique_id = row['unique_id']
    player_id = row['player_id']
    mode = row['mode'] - 1
    gender = row['gender'] - 1
    hand = row['hold racket handed'] - 1
    year = row['play years']
    level = row['level'] - 2
    
    if not (player_id in val_indices):
        continue
    
    filepath = f'../data/train/train_data/{unique_id}.txt'
    imu_data = np.loadtxt(filepath)
    imu_data = extract_valid_swing(imu_data)
    
    batch = len(imu_data) // args.shift
    if args.use_aug_features == "none":
        feature_arr_ = np.zeros((batch, args.seq_len, 6))
        all_feature = sc.transform(imu_data)

    elif args.use_aug_features == "basic":
        feature_arr_ = np.zeros((batch, args.seq_len, 24))
        diff_prev = np.diff(imu_data, axis=0, prepend=imu_data[:1])
        diff_next = np.diff(imu_data, axis=0, append=imu_data[-1:])
        cumsum = np.cumsum(imu_data, axis=0)
        all_feature = np.concatenate((imu_data, diff_prev, diff_next, cumsum), axis=1)
        all_feature = sc.transform(all_feature)
    
    if args.use_mode_as_target:
        target_arr_ = np.zeros((batch, 21), dtype=int)
    else:
        target_arr_ = np.zeros((batch, 11), dtype=int)
    
    gender_one_hot = np.eye(2)[gender]
    hand_one_hot = np.eye(2)[hand]
    year_one_hot = np.eye(3)[year]
    level_one_hot = np.eye(4)[level]
    mode_one_hot = np.eye(10)[mode]
    
    if args.use_mode_as_target:
        target = np.concatenate((gender_one_hot, hand_one_hot, year_one_hot, level_one_hot, mode_one_hot))
    else:
        target = np.concatenate((gender_one_hot, hand_one_hot, year_one_hot, level_one_hot))
    
    for i in range(batch):
        if i == batch - 1:
            feature = all_feature[i*args.shift:]
            feature_arr_[i, :len(feature), :] = feature
            target_arr_[i, :len(feature)] = target
        else:
            feature_arr_[i, :, :] = all_feature[i*args.shift:i*args.shift+args.seq_len]
            target_arr_[i, :] = target
    
    feature_arr_ = torch.tensor(feature_arr_, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(feature_arr_)
        
        out[:, 0:2] = F.softmax(out[:, 0:2])
        out[:, 2:4] = F.softmax(out[:, 2:4])
        out[:, 4:7] = F.softmax(out[:, 4:7])
        out[:, 7:11] = F.softmax(out[:, 7:11])
        
        out = torch.mean(out, dim=0)
        y_pred_batch.append(to_numpy(out))
        
        
    y_true_batch.append(target)

y_pred_batch = np.stack(y_pred_batch)
y_true_batch = np.stack(y_true_batch)
y_pred_batch = np.nan_to_num(y_pred_batch, nan=0.5)

print(f"regular batch")
metric_report(y_true_batch, y_pred_batch, final=True)

if use_wandb:
    wandb.finish()
