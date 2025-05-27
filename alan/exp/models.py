import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
                nn.Linear(args.d_model, args.output_dim),
                # nn.Sigmoid(),
        )
        # self._reinitialize()
        
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
            nn.Linear(args.d_model, args.output_dim),
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
            # nn.LayerNorm(args.n_channels * args.d_model),
            nn.Linear(args.n_channels * args.d_model, args.d_model // 2),
            # nn.LayerNorm(args.d_model // 2),
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: 可選 class weighting，shape: (num_classes,)
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits, shape (B, C)
        targets: one-hot labels, shape (B, C)
        """
        # softmax + log
        log_probs = F.log_softmax(inputs, dim=1)      # shape (B, C)
        probs = torch.exp(log_probs)                  # shape (B, C)

        # focal loss term
        focal_term = (1 - probs) ** self.gamma

        # if alpha is provided, apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)      # shape (C,)
            alpha_factor = targets * alpha.unsqueeze(0)  # shape (B, C)
        else:
            alpha_factor = 1.0

        # element-wise focal loss
        loss = -alpha_factor * focal_term * targets * log_probs  # shape (B, C)

        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
