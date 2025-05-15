import torch
import torch.nn as nn

class EncoderOnlyClassifier(nn.Module):
    def __init__(self, d_model=6, n_enc=6,nhead = 8, dim_ff=512, max_len=5000):
        super().__init__()
        # 初始化 Transformer 模型
        self.pos_embed = nn.Embedding(max_len, 64)
        self.input_proj = nn.Linear(d_model, 64)
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
        # ★ 生成位置索引，并把它转换成向量，加到 x 上
        seq_len, batch_size, _ = x.size()
        # positions: [0,1,...,seq_len-1] 形状 (seq_len, 1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(1)
        # pos_emb: (seq_len, 1, 64) → 会广播到 (seq_len, batch_size, 64)
        pos_emb = self.pos_embed(pos)
        x = x + pos_emb
        memory = self.encoder(x) 
        # Use the last time-step from decoder output
        last = memory[-1]   # shape: (batch_size, d_model)
        logits = self.classifier(last)  # shape: (batch_size, 9)
        return logits