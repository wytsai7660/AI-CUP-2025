import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.simplefilter('ignore')


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.patch_len = args.patch_len
        self.d_model = args.d_model
        if args.seq_len % args.patch_len != 0:
            print(f"Warning: seq_len ({args.seq_len}) is not perfectly divisible by patch_len ({args.patch_len}).")
            print(f"Max_num_patchs will be floor({args.seq_len} / {args.patch_len}).")

        self.max_num_patchs = args.seq_len // args.patch_len
        
        self.patch_embedding = nn.Linear(args.patch_len, args.d_model)
        pos_encoding = self._generate_sinusoidal_embedding(self.max_num_patchs, self.d_model)
        self.register_buffer('pos_encoder', pos_encoding) # Not a learnable parameter

        # self.pos_encoder = nn.Parameter(torch.randn(1, self.max_num_patchs, args.d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=args.d_model, 
            nhead=args.nhead, 
            dim_feedforward=args.d_model * 4, 
            dropout=args.dropout,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.n_enc)
        
        self.classification_head = nn.Sequential(
            nn.Linear(args.n_channels * args.d_model, args.d_model // 2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model // 2, args.output_dim)
        )
        
        self.dropout = nn.Dropout(args.dropout)
        
        
    def _generate_sinusoidal_embedding(self, max_len, d_model):
        """
        Generates sinusoidal positional embeddings.
        Args:
            max_len: Maximum sequence length (number of patches).
            d_model: Embedding dimension.
        Returns:
            Tensor of shape (1, max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0) # Add batch dimension (1, max_len, d_model)

    
    
    def forward(self, x):
        B, T, C = x.size()
        
        num_patchs = T // self.patch_len
        if T % self.patch_len != 0:
            print(f"Warning: Input sequence length T ({T}) is not perfectly divisible by patch_len ({self.patch_len}).")
            print(f"Using {num_patchs} patches.")
        
        x = x.permute(0, 2, 1)
        x = x.reshape(B * C, T)
        # (B*C, num_patches, patch_len)
        x_patched = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        # Patch Embedding: (B*C, num_patches, d_model)
        x_embedded = self.patch_embedding(x_patched)
        
        if num_patchs > self.max_num_patchs:
            raise ValueError(f"Input sequence yields {num_patchs} patches, but max_num_patchs is {self.max_num_patchs}. Check seq_len and patch_len.")

        x_embedded = x_embedded + self.pos_encoder[:, :num_patchs, :] 
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


class PatchGRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_channels = args.n_channels
        self.patch_len = args.patch_len
        self.d_model = args.d_model
        self.max_num_patchs = args.seq_len // args.patch_len
        
        self.patch_embedding = nn.Linear(args.patch_len, args.d_model)
        
        self.rnn = nn.GRU(args.d_model, args.d_model,
                            num_layers = 2, 
                            batch_first=True,
                            bidirectional=True)
                
        self.classification_head = nn.Sequential(
            nn.Linear(args.n_channels * args.d_model * 2, args.d_model // 2),
            nn.GELU(),
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
        x_embedded = self.dropout(x_embedded)
        
        # Input: (B*C, num_patches, d_model)
        # Output: (B*C, num_patches, d_model)
        transformer_out, _ = self.rnn(x_embedded)
        print(transformer_out.shape)
        # (B*C, d_model)
        aggregated_out = transformer_out.mean(dim=1)
        print(aggregated_out.shape)
        
        # (B, C, d_model)
        channel_combined_out = aggregated_out.view(B, C, self.d_model * 2)
        
        # (B, C * d_model)
        final_representation = channel_combined_out.reshape(B, C * self.d_model * 2)

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

model = PatchGRU(args)
n = sum([p.numel() for p in model.parameters()])
print(n)
X = torch.randn((2, 128 * 10, 6))
print(X.shape)
y = model(X)
print(y.shape)

# from sklearn.model_selection import train_test_split
# import pandas as pd

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