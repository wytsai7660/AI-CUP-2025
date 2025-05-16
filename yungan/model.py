import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)                  # [max_len,1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)                                             # 偶数维
        pe[:, 1::2] = torch.cos(pos * div)                                             # 奇数维
        pe = pe.unsqueeze(1)  # -> [max_len, 1, d_model]
        self.register_buffer('pe', pe)   # 存为 buffer，不参与梯度
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch_size, d_model]
        return: x + pe[:seq_len]
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

class EncoderOnlyClassifier(nn.Module):
    def __init__(self, d_model=6, n_enc = 9,nhead = 8, dim_ff=256, max_len=500):
        super().__init__()
        # 初始化 Transformer 模型
        self.input_proj = nn.Linear(d_model, 64)
        self.pos_encoder = PositionalEncoding(d_model=64, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=False       # 我們用 (seq, batch, dim) 格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc)
        self.classifier = nn.Linear(64, 11)

    def forward(self, src):
        # src, tgt: (seq_len, batch_size, d_model)
        # 把最後一個維度從 in_dim=6 -> proj_dim=64
        x = self.input_proj(src)    # -> (seq_len, batch_size, 64)
        x = self.pos_encoder(x)
        memory = self.encoder(x) 
        # Use the last time-step from decoder output
        last = memory[-1]   # shape: (batch_size, d_model)
        logits = self.classifier(last)  # shape: (batch_size, 9)
        return logits