import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.simplefilter('ignore')


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.patch_len = args.patch_len
        self.d_model = args.d_model
        self.max_num_patchs = args.seq_len // args.patch_len
        
        self.patch_embedding = nn.Linear(args.patch_len, args.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.max_num_patchs, args.d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model, 
            nhead=args.nhead, 
            dim_feedforward=args.d_model * 4, 
            dropout=args.dropout,
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.n_enc)
        
        self.classification_head = nn.Sequential(
            nn.Linear(args.n_channels * args.d_model, args.d_model // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, args.output_dim)
        )
        
        self.dropout = nn.Dropout(args.dropout)
        
        # input (B, T, C)
    
    
    def forward(self, x):
        B, T, C = x.size()
        
        num_patchs = T // self.patch_len
        
        x = x.permute(0, 2, 1)
        x = x.reshape(B * C, T)
        # (B*C, num_patches, patch_len)
        x_patched = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # Patch Embedding: (B*C, num_patches, d_model)
        x_embedded = self.patch_embedding(x_patched)
        x_embedded = x_embedded + self.pos_encoder[:, :num_patchs]
        x_embedded = self.dropout(x_embedded)
        
        # Input: (B*C, num_patches, d_model)
        # Output: (B*C, num_patches, d_model)
        transformer_out = self.transformer_encoder(x_embedded)
        
        # (B*C, d_model)
        aggregated_out = transformer_out.mean(dim=1)
        
        # (B, C, d_model)
        channel_combined_out = aggregated_out.view(B, C, self.d_model)
        
        # (B, C * d_model)
        final_representation = channel_combined_out.reshape(B, C * self.d_model)

        # (B, num_classes)
        output = self.classification_head(final_representation)

        return output

import types

args = types.SimpleNamespace(
    n_channels = 6,
    d_model = 64,
    patch_len = 256,
    seq_len = 4096,
    dropout = 0.2,
    output_dim = 11,
    nhead = 8,
    n_enc = 2,
)

model = PatchTST(args)
n = sum([p.numel() for p in model.parameters()])
print(n)
X = torch.randn((2, 128 * 10, 6))
print(X.shape)
y = model(X)
print(y.shape)

from sklearn.model_selection import train_test_split
import pandas as pd

# df = pd.read_csv('../data/train/train_info.csv')

# selected_columns_df = df[['player_id', 'gender', 'hold racket handed', 'play years', 'level']]
# unique_players_info = selected_columns_df.drop_duplicates(subset=['player_id'], keep='first')
# sorted_unique_players_info = unique_players_info.sort_values(by='player_id', ascending=True)

# train_indices, val_indices = train_test_split(
#     sorted_unique_players_info['player_id'].to_numpy(),
#     test_size=0.2,
#     random_state=42,
#     stratify=sorted_unique_players_info['level'].to_numpy()
# )


import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

imu_data = np.loadtxt('/home/alanliang/workspace/AI-CUP-2025/alan/data/train/train_data/1.txt')

sc = RobustScaler()
imu_data = sc.fit_transform(imu_data)
imu_data = torch.tensor(imu_data[None, :], dtype=torch.float32)
print(imu_data)
print(imu_data.shape)

y = model(imu_data)
print(y.shape)