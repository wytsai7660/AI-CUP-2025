import math

import torch
import torch.nn as nn

from config import CHANNELS


class PositionalEncoding(nn.Module):
    """
    標準的 Transformer 位置編碼
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2)

        pe = torch.zeros(max_len, 1, d_model)  # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe
        )  # 註冊為 buffer，這樣模型保存時會一同保存，但不會被認為是模型參數

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True for encoder
               or [seq_len, batch_size, embedding_dim] if batch_first=False
        """
        # 假設 x 的 shape 是 [batch_size, seq_len, embedding_dim]
        # TransformerEncoderLayer 預設 batch_first=False, 即 (seq_len, batch, feature)
        # 如果 TransformerEncoderLayer 的 batch_first=True, 則 x 是 (batch, seq_len, feature)
        # 我們的 pe 是 (max_len, 1, d_model)，需要調整以匹配 x

        # 如果 x 是 (batch_size, seq_len, embedding_dim)
        if x.dim() == 3 and self.pe.size(1) == 1:  # batch_first=True
            # self.pe[:x.size(1)] -> (seq_len, 1, d_model) -> transpose(0,1) -> (1, seq_len, d_model)
            x = x + self.pe[: x.size(1)].transpose(0, 1)
        # 如果 x 是 (seq_len, batch_size, embedding_dim)
        elif x.dim() == 3 and self.pe.size(1) == 1:  # batch_first=False
            x = x + self.pe[: x.size(0)]
        else:
            raise ValueError(
                f"Input x shape {x.shape} not supported by PositionalEncoding."
            )

        return self.dropout(x)


class MultiTaskTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,  # Transformer 模型的隱藏維度
        nhead: int = 4,  # 多頭自註意力的頭數 (需能整除 d_model)
        num_encoder_layers: int = 3,  # Transformer Encoder 層數
        dim_feedforward: int = 256,  # FeedForward 層的隱藏維度
        dropout: float = 0.1,
        num_classes_list: list = [2, 2, 3, 4],  # 各任務的類別數
        max_len: int = 5000,
    ):  # 預期的最大序列長度，用於位置編碼
        super().__init__()

        self.d_model = d_model

        # 1. 輸入嵌入
        self.input_projection = nn.Linear(len(CHANNELS), d_model)

        # 2. 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 3. Transformer 編碼器
        # TransformerEncoderLayer 預設 batch_first=False. 為了方便處理 (batch, length, feature) 的輸入，設為 True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # *** 重要 ***
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers
        )

        # 4. 多任務分類頭
        self.task_heads = nn.ModuleList()
        for num_classes in num_classes_list:
            self.task_heads.append(nn.Linear(d_model, num_classes))

        self.init_weights()

    def init_weights(self):
        # 初始化權重 (可選，但通常是個好習慣)
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        for head in self.task_heads:
            head.weight.data.uniform_(-initrange, initrange)
            head.bias.data.zero_()

    def _generate_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """
        生成 padding mask.
        Args:
            lengths: Tensor, shape (batch_size), 每個序列的實際長度
            max_len: int, 當前 batch 中最長的序列長度
        Returns:
            Tensor, shape (batch_size, max_len), True 代表是 padding 的位置
        """
        batch_size = lengths.size(0)
        # 創建一個 (batch_size, max_len) 的 mask，初始為 False
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> list:
        """
        Args:
            inputs: Tensor, shape (batch, length, 6)
            lengths: Tensor, shape (batch)
        Returns:
            list of Tensors, 每個 Tensor 是對應任務的 logits (batch, num_classes_for_task)
        """
        batch_size, seq_len, _ = inputs.shape

        # 1. 生成 padding mask
        # TransformerEncoder 需要的 src_key_padding_mask: (N, S) 其中 N=batch_size, S=seq_len
        # 如果一個 S 位置的值為 True，則該 S 位置將被忽略 (masked out)。
        padding_mask = self._generate_padding_mask(lengths, seq_len)  # (batch, seq_len)

        # 2. 輸入嵌入 + 位置編碼
        embedded_inputs = self.input_projection(inputs) * math.sqrt(
            self.d_model
        )  # (batch, seq_len, d_model)
        positioned_inputs = self.pos_encoder(
            embedded_inputs
        )  # (batch, seq_len, d_model)

        # 3. Transformer 編碼
        # src_key_padding_mask: (batch, seq_len)
        transformer_output = self.transformer_encoder(
            src=positioned_inputs, src_key_padding_mask=padding_mask
        )  # (batch, seq_len, d_model)

        # 4. 序列表示聚合 (使用 mean pooling)
        # 我們需要確保只對非 padding 部分進行平均
        # 首先，將 padding 部分的輸出設為0，以便不影響求和
        # padding_mask 是 True for padded, 我們需要 False for padded for multiplication
        # (batch, seq_len, 1)
        mask_for_pooling = (~padding_mask).unsqueeze(-1).float()

        summed_output = (transformer_output * mask_for_pooling).sum(
            dim=1
        )  # (batch, d_model)

        # 計算每個序列的實際 token 數量 (避免除以0)
        # lengths 是 (batch), reshape to (batch, 1) for broadcasting
        actual_lengths = lengths.unsqueeze(1).float()
        actual_lengths = torch.clamp(
            actual_lengths, min=1.0
        )  # 避免除以0 (雖然理論上長度至少為1)

        pooled_output = summed_output / actual_lengths  # (batch, d_model)

        # 也可以簡單地取第一個 token 的輸出，如果任務允許（例如，類似 BERT 的 CLS token）
        # pooled_output = transformer_output[:, 0, :] # (batch, d_model)
        # 注意：如果使用這種方法，要確保 padding_mask 不會 mask 掉第一個 token (通常不會)
        # 且第一個 token 的表示需要能代表整個序列。對於一般訊號，mean pooling 更通用。

        # 5. 多任務分類頭
        task_logits = []
        for head in self.task_heads:
            task_logits.append(head(pooled_output))

        return task_logits
