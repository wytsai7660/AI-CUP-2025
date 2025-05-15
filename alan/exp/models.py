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
