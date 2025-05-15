#%%
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
import gc
import random
import os
from types import SimpleNamespace
import logging
import wandb
import joblib
import csv
from copy import deepcopy

pd.set_option('display.max_columns', 300)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

import argparse
import yaml
parser = argparse.ArgumentParser(description='Your main training script loading config from YAML.')
parser.add_argument('--config', type=str, required=True,
                    help='Path to the YAML configuration file')

cmd_args = parser.parse_args()
config_path = cmd_args.config

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
use_wandb = True

debug = False


exp = "003"
if not os.path.exists(f"../out/exp/exp{exp}"):
    os.makedirs(f"../out/exp/exp{exp}")
    os.makedirs(f"../out/exp/exp{exp}/exp{exp}_model")
logger_path = f"../out/exp/exp{exp}/exp_{exp}.txt"
model_path =f"../out/exp/exp{exp}/exp{exp}_model/exp{exp}.pth"
LOGGER = logging.getLogger(__name__)
file_handler = logging.FileHandler(logger_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

device = "cuda"

fe = "005"

train_feature_path = f"../out/fe/fe{fe}/train_feature.npy"
train_target_path = f"../out/fe/fe{fe}/train_target.npy"
val_feature_path = f"../out/fe/fe{fe}/val_feature.npy"
val_target_path = f"../out/fe/fe{fe}/val_target.npy"
val_indices_path = f"../out/fe/fe{fe}/val_indices.npy"
sc_path = f"../out/fe/fe{fe}/robust_scaler.joblib"

train_feature = np.load(train_feature_path)
train_target = np.load(train_target_path)
val_feature = np.load(val_feature_path)
val_target = np.load(val_target_path)
val_indices = np.load(val_indices_path)

train_feature = train_feature.astype(np.float32)
val_feature = val_feature.astype(np.float32)
train_target = train_target.astype(np.float32)
val_target = val_target.astype(np.float32)

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
        
class SwingGRU(nn.Module):
    def __init__(self, args):
        # dropout=0.2,
        # input_dim = 24,
        # hidden_dim = 64,
        # model_dim = 128,
        # out_size = 11
        super(SwingGRU, self).__init__()
        self.args = args
        self.numerical_linear  = nn.Sequential(
                nn.Linear(args.input_dim, args.hidden_dim),
                nn.LayerNorm(args.hidden_dim)
            )
        
        self.rnn = nn.GRU(args.hidden_dim, args.d_model,
                            num_layers = 2, 
                            batch_first=True,
                            bidirectional=True)
                
        self.linear_out  = nn.Sequential(
                nn.Linear(args.d_model * 2, 
                          args.d_model),
                nn.LayerNorm(args.d_model),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.d_model, 11),
                # nn.Sigmoid(),
        )
        self._reinitialize()
        
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
    
    def forward(self, numerical_array):
        
        numerical_embedding = self.numerical_linear(numerical_array)
        output,_ = self.rnn(numerical_embedding)
        if self.args.mean_pooling:
            last = torch.mean(output, dim=1)
        else:
            last = output[:, -1, :]
        output = self.linear_out(last)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        # Create sinusoidal positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class EncoderOnlyClassifier(nn.Module):
    def __init__(self, args):
        # input_dim=24, n_enc=2, nhead=8, d_model=64, max_seq_len=1000
        super().__init__()
        self.args = args
        # Initialize Transformer model
        self.input_proj = nn.Sequential(
            nn.Linear(args.input_dim, args.d_model),
        )
        
        # Positional encoding
        if args.pos_emb:
            self.pos_encoder = PositionalEncoding(args.d_model, args.max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.d_model * 4,
            batch_first=True,
            norm_first=True,
            # activation=F.gelu(approximate='tanh')
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.n_enc)
        self.classifier = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.GELU(approximate='tanh'),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, 11),
        )

    def forward(self, src):
        # Project input to d_model dimension
        x = self.input_proj(src)  # -> (batch_size, seq_len, d_model)
        
        # Add positional encoding
        if self.args.pos_emb:
            x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        memory = self.encoder(x)
        
        # Use the last time-step from encoder output
        # last = torch.mean(memory, dim=1)
        if self.args.mean_pooling:
            last = torch.mean(memory, dim=1)
        else:
            last = memory[:, -1, :]  # shape: (batch_size, d_model)
        logits = self.classifier(last)  # shape: (batch_size, 11)
        
        return logits

def to_numpy(p: torch.Tensor):
    if p.requires_grad:
        return p.detach().cpu().numpy()
    else:
        return p.cpu().numpy()

def metric_report(y_batch, out_batch, log='none'):
    cut = [0, 2, 4, 7, 11]
    classes = ['gender', 'hand', 'year', 'level']
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
        # micro_presicion_score = average_precision_score(y_batch[:, start:end], out_batch[:, start:end], average='micro')
        # macro_presicion_score = average_precision_score(y_batch[:, start:end], out_batch[:, start:end], average='macro')
        print(f"micro roc {micro_roc_score:.4f} macro roc: {macro_roc_score:.4f} acc: {acc_score:.4f} micro f1: {micro_f1_score:.4f} macro f1: {macro_f1_score:.4f}")
        # print(f"{cls} micro roc: {micro_roc_score:.4f}, macro roc: {macro_roc_score:.4f}, micro presci: {micro_presicion_score:.4f}, macro presci: {macro_presicion_score:.4f}")
        
        if cls == 'level' and use_wandb:
            if log == 'progress':
                wandb.log({
                    # "epoch": epoch,
                    "micro_roc_auc": micro_roc_score,
                    "macro_roc_auc": macro_roc_score,
                    "acc_score": acc_score,
                    "micro_f1_score": micro_f1_score,
                    "macro_f1_score": macro_f1_score,
                    # "micro_presicion": micro_presicion_score,
                    # "macro_presicion": macro_presicion_score,
                })

            elif log == 'full':
                wandb.log({
                    # "epoch": epoch,
                    "full_micro_roc_auc": micro_roc_score,
                    "full_macro_roc_auc": macro_roc_score,
                    "full_acc_score": acc_score,
                    "full_micro_f1_score": micro_f1_score,
                    "full_macro_f1_score": macro_f1_score,

                    # "full_micro_presicion": micro_presicion_score,
                    # "full_macro_presicion": macro_presicion_score,
                })
        
            elif log == 'finish':
                wandb.log({
                    # "epoch": epoch,
                    "final_micro_roc_auc": micro_roc_score,
                    "final_macro_roc_auc": macro_roc_score,
                    "final_acc_score": acc_score,
                    "final_micro_f1_score": micro_f1_score,
                    "final_macro_f1_score": macro_f1_score,
                    # "final_micro_presicion": micro_presicion_score,
                    # "final_macro_presicion": macro_presicion_score,
                })

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        :param class_weight: tensor of shape [num_classes]，類似 nn.CrossEntropyLoss 的 weight 參數
        :param gamma: focal loss 的聚焦參數
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: logits tensor，形狀為 [N, C]
        :param targets: one-hot tensor，形狀為 [N, C]
        """
        assert inputs.shape == targets.shape, "Inputs and one-hot targets must have the same shape"

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        weight_tensor = self.weight * targets

        loss = -weight_tensor * focal_weight * log_probs * targets
        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss(args, class_weights):
    criterions = []
    cut = [0, 2, 4, 7, 11]
    classes = ['gender', 'hand', 'year', 'level']

    for start, end in zip(cut, cut[1:]):
        partial_weights = torch.tensor(class_weights[start:end]).to(device)
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=partial_weights)
        elif args.loss_type == 'FL':
            criterion = FocalLoss(weight=partial_weights, gamma=args.FL_gamma)
        # elif loss_type == 'BCE':
        #     criterion = nn.BCEWithLogitsLoss(weight=partial_weights)
        else:
            raise ValueError(f"no such loss {args.loss_type}")
        criterions.append(criterion)
    
    return criterions


# ablations: encoder vs gru
# learning rate, batch size, weight decay
# main loss weight
# loss type
# mean pooling
# 

#====================== config ======================

args = SimpleNamespace(
    batch_size = 32,
    n_epochs = 10,
    lr = 1e-5,
    weight_decay = 0.05,
    num_warmup_steps = 10,
    main_loss_weight = 0.6,
    loss_type = 'BCE',
    model_type = 'gru',
    seq_len = 500,
    shift = 250,
    
    # input_dim = 24,
    # n_enc = 2,
    # nhead = 8,
    # d_model = 64,
    # max_seq_len = 500,
    # dropout = 0.2,
    # mean_pooling = True,
    # pos_emb = False,
    
    input_dim = 24,
    hidden_dim = 64,
    d_model = 128,
    dropout = 0.2,
    mean_pooling = True,
)

#==================override with file==============

if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at {config_path}")
    exit(1)

try:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        args = SimpleNamespace(**config_data)
except yaml.YAMLError as e:
    print(f"Error decoding YAML from {config_path}: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading or processing config {config_path}: {e}")
    exit(1)

if use_wandb:
    wandb_config = vars(args)
    wandb.init(project="table-tennis-002", name=f"{timestamp}", config=wandb_config)

#====================================================


set_seed(42)
cut = [0, 2, 4, 7, 11]
classes = ['gender', 'hand', 'year', 'level']

train_weights = 1 / np.sum(train_target, axis=0)
final_weights = np.zeros((11,))
for start, end, cls in zip(cut, cut[1:], classes):
    class_weights = train_weights[start:end] / np.sum(train_weights[start:end])
    final_weights[start:end] = class_weights
    print(f"{cls} weights: {class_weights}")

train_ds = SwingDataset(train_feature, train=True, y = train_target)
val_ds = SwingDataset(val_feature, train=True, y = val_target)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=4)

if args.model_type == 'gru':
    model = SwingGRU(args)
elif args.model_type == 'encoder':
    model = EncoderOnlyClassifier(args)
else:
    raise ValueError(f"no such model type {args.model_type}")

model = model.to(device)
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

criterions = get_loss(args, final_weights)

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
        aux_loss += criterions[0](out[:, :2], y[:, :2])
        aux_loss += criterions[1](out[:, 2:4], y[:, 2:4])
        aux_loss += criterions[2](out[:, 4:7], y[:, 4:7])
        main_loss += criterions[3](out[:, 7:], y[:, 7:])
        loss = (1 - args.main_loss_weight) * aux_loss + args.main_loss_weight * main_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            out[:, :2] = F.softmax(out[:, :2])
            out[:, 2:4] = F.softmax(out[:, 2:4])
            out[:, 4:7] = F.softmax(out[:, 4:7])
            out[:, 7:] = F.softmax(out[:, 7:])
        
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
            aux_loss += criterions[0](out[:, :2], y[:, :2])
            aux_loss += criterions[1](out[:, 2:4], y[:, 2:4])
            aux_loss += criterions[2](out[:, 4:7], y[:, 4:7])
            main_loss += criterions[3](out[:, 7:], y[:, 7:])
            loss = (1 - args.main_loss_weight) * aux_loss + args.main_loss_weight * main_loss

            with torch.no_grad():
                out[:, :2] = F.softmax(out[:, :2])
                out[:, 2:4] = F.softmax(out[:, 2:4])
                out[:, 4:7] = F.softmax(out[:, 4:7])
                out[:, 7:] = F.softmax(out[:, 7:])
            
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
    metric_report(val_y_batch, val_out_batch, log='progress')
    
    if val_main_loss < best_loss:
        best_loss = val_main_loss
        best_model = deepcopy(model.state_dict())
        print(f"✨ best Val loss at epoch {epoch}, loss {val_main_loss:.4f}")
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1
    
    if early_stop_cnt == 3:
        print(f"No imporvement for {early_stop_cnt} Epochs, stopping")
        break

#%%
val_indices
#%%
# ================ validation pharse =================

model.load_state_dict(best_model)
model.eval()
sc: RobustScaler = joblib.load(sc_path)

df = pd.read_csv('../data/train/train_info.csv')

seq_len = args.seq_len
shift = args.shift

y_batch = []
out_batch = []
full_out_batch = []

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
    
    batch = len(imu_data) // shift
    feature_arr_ = np.zeros((batch, seq_len, 24))
    target_arr_ = np.zeros((batch, 11), dtype=int)
    
    diff_prev = np.diff(imu_data, axis=0, prepend=imu_data[:1])
    diff_next = np.diff(imu_data, axis=0, append=imu_data[-1:])
    cumsum = np.cumsum(imu_data, axis=0)
    all_feature = np.concatenate((imu_data, diff_prev, diff_next, cumsum), axis=1)
    all_feature = sc.transform(all_feature)
    gender_one_hot = np.eye(2)[gender]
    hand_one_hot = np.eye(2)[hand]
    year_one_hot = np.eye(3)[year]
    level_one_hot = np.eye(4)[level]
    target = np.concatenate((gender_one_hot, hand_one_hot, year_one_hot, level_one_hot))
    
    for i in range(batch):
        if i == batch - 1:
            feature = all_feature[i*shift:]
            feature_arr_[i, :len(feature), :] = feature
            target_arr_[i, :len(feature)] = target
            # mask_arr_[i, :len(fe)] = 1
            # target_mask_arr_[i, offset:len(fe)] = 1
        else:
            feature_arr_[i, :, :] = all_feature[i*shift:i*shift+seq_len]
            target_arr_[i, :] = target
            # mask_arr_[i, :] = 1
            # target_mask_arr_[i, offset:] = 1
    
    feature_arr_ = torch.tensor(feature_arr_, dtype=torch.float32).to(device)
    # target_arr_ = torch.tensor(target_arr_, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(feature_arr_)
        
        out[:, :2] = F.softmax(out[:, :2])
        out[:, 2:4] = F.softmax(out[:, 2:4])
        out[:, 4:7] = F.softmax(out[:, 4:7])
        out[:, 7:] = F.softmax(out[:, 7:])
        
        out = torch.mean(out, dim=0)
        out_batch.append(to_numpy(out))
        
    # full len
    if args.model_type == 'gru' or (not args.pos_emb):
        all_feature = torch.tensor(all_feature[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(all_feature)
            
            out[:, :2] = F.softmax(out[:, :2])
            out[:, 2:4] = F.softmax(out[:, 2:4])
            out[:, 4:7] = F.softmax(out[:, 4:7])
            out[:, 7:] = F.softmax(out[:, 7:])
            
            out = out.squeeze()
            full_out_batch.append(to_numpy(out))
        
    y_batch.append(target)

#%%

out_batch = np.stack(out_batch)
y_batch = np.stack(y_batch)
out_batch = np.nan_to_num(out_batch, nan=0.5)

print(f"regular batch")
metric_report(y_batch, out_batch, log='finish')

if args.model_type == 'gru' or (not args.pos_emb):
    full_out_batch = np.nan_to_num(full_out_batch, nan=0.5)
    full_out_batch = np.stack(full_out_batch)

    print(f"full batch")
    metric_report(y_batch, full_out_batch, log='full')

if use_wandb:
    wandb.finish()
    
#%%

#=============================== test phrase ==========================
df = pd.read_csv('../data/test/test_info.csv')
os.makedirs("submissions", exist_ok=True)


model.eval()
results = []

for index, row in df.iterrows():
    unique_id = row['unique_id']
    # mode = row['mode'] - 1
    
    filepath = f'../data/test/test_data/{unique_id}.txt'
    imu_data = np.loadtxt(filepath)
    imu_data = extract_valid_swing(imu_data)
    
    batch = len(imu_data) // shift
    feature_arr_ = np.zeros((batch, seq_len, 24))
    
    diff_prev = np.diff(imu_data, axis=0, prepend=imu_data[:1])
    diff_next = np.diff(imu_data, axis=0, append=imu_data[-1:])
    cumsum = np.cumsum(imu_data, axis=0)
    all_feature = np.concatenate((imu_data, diff_prev, diff_next, cumsum), axis=1)
    all_feature = sc.transform(all_feature)
    
    for i in range(batch):
        if i == batch - 1:
            feature = all_feature[i*shift:]
            feature_arr_[i, :len(feature), :] = feature
        else:
            feature_arr_[i, :, :] = all_feature[i*shift:i*shift+seq_len]

    feature_arr_ = torch.tensor(feature_arr_, dtype=torch.float32).to(device)
    # all_feature = torch.tensor(all_feature[None, :], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(feature_arr_)
        # full_out = model(all_feature).squeeze()
        
        out[:, :2] = F.softmax(out[:, :2])
        out[:, 2:4] = F.softmax(out[:, 2:4])
        out[:, 4:7] = F.softmax(out[:, 4:7])
        out[:, 7:] = F.softmax(out[:, 7:])
        
        out = torch.mean(out, dim=0)
    
    out = to_numpy(out)
    out = np.nan_to_num(out, nan=0.5)
    row = [unique_id, *(f"{0.5:.4f}" for _ in range(9 - 4)), *(f"{v:.4f}" for v in out[7:])]
    results.append(row)
    # full_out_batch.append(to_numpy(full_out))

header = [
    "unique_id",
    "gender",
    "hold racket handed",
    "play years_0","play years_1","play years_2",
    "level_2","level_3","level_4","level_5"
]
with open(f"submissions/{timestamp}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print(f"Saved results to {timestamp}.csv")
