import torch
import torch.nn as nn
import math

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2000, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # Create a vector of shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Create the division term for the frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        # Register 'pe' as a buffer so it's part of the model state but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        Args:
            x: Input tensor of shape (batch, max_seq_len, d_model)
        Returns:
            Tensor with added positional encoding, shape (batch, max_seq_len, d_model)
        """
        # Add positional encoding up to the length of the input sequence
        # .requires_grad_(False) ensures the positional encoding itself is not updated during training
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

# --- Layer Normalization ---
class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization.
    """
    def __init__(self, features: int, eps: float = 1e-6): # Added 'features' argument
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.alpha = nn.Parameter(torch.ones(features)) # Changed size to 'features'
        # Learnable shift parameter (beta)
        self.bias = nn.Parameter(torch.zeros(features)) # Changed size to 'features'

    def forward(self, x):
        """
        Applies layer normalization.
        Args:
            x: Input tensor, typically shape (batch, max_seq_len, features)
        Returns:
            Normalized tensor, same shape as input.
        """
        # Calculate mean and standard deviation along the feature dimension (-1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Normalize, scale, and shift
        return self.alpha * (x - mean) / (torch.sqrt(std**2 + self.eps)) + self.bias # Added sqrt for stability

# --- Feed Forward Network ---
class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward network used in Transformer blocks.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # First linear layer expands dimensionality
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # Second linear layer contracts dimensionality back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        Args:
            x: Input tensor of shape (batch, max_seq_len, d_model)
        Returns:
            Output tensor of shape (batch, max_seq_len, d_model)
        """
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        x = torch.relu(self.linear1(x)) # Apply ReLU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        # Ensure d_model is divisible by the number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Dimensionality of each head's key/query/value vectors
        self.d_k = d_model // n_heads

        # Linear layers for Query, Key, Value transformations
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        # Final linear layer to combine head outputs
        self.w_out = nn.Linear(d_model, d_model) # Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Static method to compute scaled dot-product attention.
        Args:
            query: Query tensor (Batch, n_heads, Seq_q, d_k)
            key: Key tensor (Batch, n_heads, Seq_k, d_k)
            value: Value tensor (Batch, n_heads, Seq_v, d_k) (Seq_k == Seq_v)
            mask: Optional mask tensor
            dropout: Dropout layer
        Returns:
            Tuple of (output tensor, attention scores)
        """
        d_k = query.size(-1) # Dimensionality of keys/queries

        # Calculate attention scores: (Q * K^T) / sqrt(d_k)
        # (Batch, n_heads, Seq_q, d_k) @ (Batch, n_heads, d_k, Seq_k) -> (Batch, n_heads, Seq_q, Seq_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided (prevents attention to certain positions)
        if mask is not None:
            # Fill masked positions with a very small number before softmax
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # -1e9 or float('-inf')

        # Apply softmax to get attention probabilities along the key sequence dimension
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, n_heads, Seq_q, Seq_k)

        # Apply dropout to attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply attention scores by Value: (Attention * V)
        # (Batch, n_heads, Seq_q, Seq_k) @ (Batch, n_heads, Seq_v, d_k) -> (Batch, n_heads, Seq_q, d_k)
        # Note: Seq_k and Seq_v must be the same for this matrix multiplication.
        output = attention_scores @ value
        return output, attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for Multi-Head Attention.
        Args:
            q: Query input tensor (Batch, Seq_q, d_model)
            k: Key input tensor (Batch, Seq_k, d_model)
            v: Value input tensor (Batch, Seq_v, d_model) (Seq_k == Seq_v)
            mask: Optional mask tensor
        Returns:
            Output tensor of shape (Batch, Seq_q, d_model)
        """
        query = self.w_q(q) # (Batch, Seq_q, d_model) -> (Batch, Seq_q, d_model)
        key = self.w_k(k)   # (Batch, Seq_k, d_model) -> (Batch, Seq_k, d_model)
        value = self.w_v(v) # (Batch, Seq_v, d_model) -> (Batch, Seq_v, d_model)

        # Reshape and transpose for multi-head calculation:
        # (Batch, Seq, d_model) -> (Batch, Seq, n_heads, d_k) -> (Batch, n_heads, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention
        # x shape: (Batch, n_heads, Seq_q, d_k)
        # self.attention_scores shape: (Batch, n_heads, Seq_q, Seq_k)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Combine heads:
        # (Batch, n_heads, Seq_q, d_k) -> (Batch, Seq_q, n_heads, d_k) -> (Batch, Seq_q, d_model)
        # .contiguous() is needed after transpose before .view()
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k) # self.n_heads * self.d_k = d_model

        # Pass through the final output linear layer
        # (Batch, Seq_q, d_model) -> (Batch, Seq_q, d_model)
        return self.w_out(x)

# --- Residual Connection ---
class ResidualConnection(nn.Module):
    """
    Implements the residual connection followed by layer normalization.
    """
    def __init__(self, features: int, dropout: float = 0.1) -> None: # Added 'features'
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features) # Pass features to LayerNorm

    def forward(self, x, sublayer):
        """
        Applies the residual connection.
        Args:
            x: Input tensor
            sublayer: The layer (e.g., MultiHeadAttention or FeedForward) to apply.
        Returns:
            Output tensor after residual connection and normalization.
        """
        # Apply normalization to the input, pass through sublayer, apply dropout, then add original input
        return x + self.dropout(sublayer(self.norm(x)))

# --- Encoder Layer ---
class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Consists of self-attention and feed-forward sublayers.
    """
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float = 0.1): # Added d_model
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections needed: one after attention, one after feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)]) # Pass d_model

    def forward(self, x, src_mask):
        """
        Forward pass through the encoder layer.
        Args:
            x: Input tensor (Batch, Seq, d_model)
            src_mask: Source mask for self-attention
        Returns:
            Output tensor (Batch, Seq, d_model)
        """
        # Apply self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Apply feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# --- Encoder ---
class Encoder(nn.Module):
    """
    The Transformer Encoder, composed of multiple EncoderLayers.
    """
    def __init__(self, features: int, layers: nn.ModuleList): # Added features
        super().__init__()
        self.layers = layers
        # Final normalization layer after all encoder layers
        self.norm = LayerNormalization(features) # Pass features

    def forward(self, x, mask):
        """
        Forward pass through the entire encoder stack.
        Args:
            x: Input tensor (Batch, Seq, d_model)
            mask: Source mask
        Returns:
            Encoded output tensor (Batch, Seq, d_model)
        """
        # Pass input through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final normalization
        return self.norm(x)

# --- Decoder Layer ---
class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.
    Consists of self-attention, cross-attention, and feed-forward sublayers.
    """
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float = 0.1) -> None: # Added d_model
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Three residual connections needed: self-attn, cross-attn, feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)]) # Pass d_model

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the decoder layer.
        Args:
            x: Input tensor to the decoder (Batch, Seq_tgt, d_model)
            encoder_output: Output from the encoder (Batch, Seq_src, d_model)
            src_mask: Source mask (for cross-attention)
            tgt_mask: Target mask (for self-attention)
        Returns:
            Output tensor (Batch, Seq_tgt, d_model)
        """
        # Masked self-attention (on decoder input) with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Cross-attention (decoder input Q, encoder output K, V) with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Feed-forward with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# --- Decoder ---
class Decoder(nn.Module):
    """
    The Transformer Decoder, composed of multiple DecoderLayers.
    """
    def __init__(self, features: int, layers: nn.ModuleList): # Added features
        super().__init__()
        self.layers = layers
        # Final normalization layer after all decoder layers
        self.norm = LayerNormalization(features) # Pass features

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass through the entire decoder stack.
        Args:
            x: Target input tensor (Batch, Seq_tgt, d_model)
            encoder_output: Output from the encoder (Batch, Seq_src, d_model)
            src_mask: Source mask
            tgt_mask: Target mask
        Returns:
            Decoded output tensor (Batch, Seq_tgt, d_model)
        """
        # Pass input through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply final normalization
        return self.norm(x)

# --- Final MLP for Regression Output ---
class FinalMLP(nn.Module):
    """
    Projects the final decoder output to the desired regression output dimension.
    """
    def __init__(self, d_model: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Linear layer to map d_model to the final output dimension
        self.linear = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout) # Added dropout for regularization

    def forward(self, x):
        """
        Forward pass for the final projection.
        Args:
            x: Input tensor, expected shape (Batch, d_model) after sequence aggregation
               or (Batch, Seq, d_model) if applied per token.
        Returns:
            Output tensor, shape (Batch, output_dim) or (Batch, Seq, output_dim)
        """
        x = self.linear(x)
        # x = self.dropout(x) # Optional: Apply dropout before the final output
        return x

# --- Transformer Model ---
class Transformer(nn.Module):
    """
    The complete Transformer model architecture.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, final_mlp: FinalMLP, d_input: int = 6, d_model: int = 32) -> None: # Changed d_input default
        super().__init__()
        # Linear layer to project input features (d_input) to d_model
        self.input_projection = nn.Linear(d_input, d_model)
        # Linear layer to project target features (assuming same as input for now) to d_model
        # If target features are different, you might need a separate projection layer
        self.target_projection = nn.Linear(d_input, d_model) # Assuming target also has d_input features initially

        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos # Positional encoding for source sequence
        self.tgt_pos = tgt_pos # Positional encoding for target sequence
        self.final_mlp = final_mlp # Final projection layer

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.
        Args:
            src: Source input tensor (Batch, Seq_src, d_input)
            src_mask: Source mask
        Returns:
            Encoder output tensor (Batch, Seq_src, d_model)
        """
        # Project input features to d_model, then add positional encoding
        src = self.input_projection(src) # (Batch, Seq_src, d_input) -> (Batch, Seq_src, d_model)
        src = self.src_pos(src)
        # Pass through the encoder stack
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decodes the target sequence using encoder output.
        Args:
            encoder_output: Output from the encoder (Batch, Seq_src, d_model)
            src_mask: Source mask
            tgt: Target input tensor (Batch, Seq_tgt, d_input) - Assuming target features match input
            tgt_mask: Target mask
        Returns:
            Decoder output tensor (Batch, Seq_tgt, d_model)
        """
        # Project target features to d_model, then add positional encoding
        tgt = self.target_projection(tgt) # (Batch, Seq_tgt, d_input) -> (Batch, Seq_tgt, d_model)
        tgt = self.tgt_pos(tgt)
        # Pass through the decoder stack
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Applies the final projection layer.
        Args:
            x: Input tensor from the decoder (Batch, Seq_tgt, d_model)
        Returns:
            Projected output tensor (Batch, Seq_tgt, output_dim)
        """
        return self.final_mlp(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Full forward pass of the Transformer.
        Args:
            src: Source input tensor (Batch, Seq_src, d_input)
            tgt: Target input tensor (Batch, Seq_tgt, d_input)
            src_mask: Source mask
            tgt_mask: Target mask
        Returns:
            Final output tensor for regression (Batch, output_dim)
        """
        # Encode the source sequence
        encoder_output = self.encode(src, src_mask) # (Batch, Seq_src, d_model)
        # Decode the target sequence using encoder output
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask) # (Batch, Seq_tgt, d_model)

        # --- MODIFICATION: Select the last token's output for regression ---
        # Take the output corresponding to the last token in the target sequence
        last_token_output = decoder_output[:, -1, :] # (Batch, d_model)

        # Apply the final MLP to the last token's output
        final_output = self.project(last_token_output) # (Batch, output_dim)

        return final_output


# --- Builder Function ---
def build_transformer(N=4, d_model=32, d_ff=128, h=4, dropout=0.1, d_input=6, output_dim=4, max_seq_len=2000): # Added d_input, output_dim, max_seq_len defaults
    """
    Builds the Transformer model with specified hyperparameters.
    Args:
        N: Number of encoder/decoder layers
        d_model: Model dimension (embedding size)
        d_ff: Dimension of the feed-forward layer
        h: Number of attention heads
        dropout: Dropout rate
        d_input: Dimension of the input features (CHANGED to 6)
        output_dim: Dimension of the final regression output (CHANGED to 4)
        max_seq_len: Maximum sequence length for positional encoding
    Returns:
        An instance of the Transformer model.
    """
    # --- Input Checks ---
    assert d_model % h == 0, "d_model must be divisible by n_heads (h)"

    # --- Create Components ---
    # Positional Encoding
    src_pos = PositionalEncoding(d_model, max_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, max_seq_len, dropout)

    # Attention Blocks (shared between encoder and decoder for simplicity here)
    self_attn = MultiHeadAttention(d_model, h, dropout)
    cross_attn = MultiHeadAttention(d_model, h, dropout) # Separate instance for cross-attention clarity

    # Feed Forward Block
    feed_forward = FeedForward(d_model, d_ff, dropout)

    # Encoder Layers
    encoder_layers = nn.ModuleList(
        [EncoderLayer(d_model, self_attn, feed_forward, dropout) for _ in range(N)] # Pass d_model
    )
    # Decoder Layers
    decoder_layers = nn.ModuleList(
        [DecoderLayer(d_model, self_attn, cross_attn, feed_forward, dropout) for _ in range(N)] # Pass d_model
    )

    # Encoder and Decoder Stacks
    encoder = Encoder(d_model, encoder_layers) # Pass d_model
    decoder = Decoder(d_model, decoder_layers) # Pass d_model

    # Final Projection Layer
    final_mlp = FinalMLP(d_model, output_dim, dropout) # Use output_dim=4

    # --- Assemble Transformer ---
    model = Transformer(
        encoder,
        decoder,
        src_pos,
        tgt_pos,
        final_mlp,
        d_input=d_input, # Use d_input=6
        d_model=d_model
    )

    # --- Initialize Parameters (Optional but recommended) ---
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

# --- Example Usage ---
if __name__ == '__main__':
    # Hyperparameters
    batch_size = 8
    src_max_seq_len = 50 # Example source sequence length
    tgt_max_seq_len = 10 # Example target sequence length (can be different)
    input_features = 6 # Your input feature dimension
    output_features = 9 # Your desired output dimension
    model_dim = 32
    num_layers = 4
    num_heads = 4
    ff_dim = 128
    dropout_rate = 0.1

    # Build the model
    transformer_model = build_transformer(
        N=num_layers,
        d_model=model_dim,
        d_ff=ff_dim,
        h=num_heads,
        dropout=dropout_rate,
        d_input=input_features,
        output_dim=output_features,
        max_seq_len=max(src_max_seq_len, tgt_max_seq_len) + 10 # Set max_seq_len for PE buffer
    )

    # Create dummy input data
    # Source input: (batch, max_seq_len, features)
    src_data = torch.randn(batch_size, src_max_seq_len, input_features)
    # Target input (for teacher forcing during training): (batch, max_seq_len, features)
    # For regression, the target input 'tgt' might be different.
    # Often, for sequence-to-value regression, you might only need the source 'src'.
    # The 'tgt' input to the decoder during training usually starts with a special token
    # and uses the shifted true output sequence.
    # For inference, 'tgt' is generated token by token.
    # If you only need ONE output vector (4,) per sequence, the decoder structure might
    # need further adaptation depending on your exact task setup.
    # This example assumes a standard seq-to-seq setup where we take the last output.
    tgt_data = torch.randn(batch_size, tgt_max_seq_len, input_features) # Using input_features for tgt for simplicity

    # Create dummy masks (no masking in this simple example)
    # Real applications require proper masking (padding mask, look-ahead mask)
    src_mask = None # Or torch.ones(batch_size, 1, src_max_seq_len) etc.
    tgt_mask = None # Or generate causal mask torch.tril(torch.ones(tgt_max_seq_len, tgt_max_seq_len))... etc.

    # Forward pass
    output = transformer_model(src_data, tgt_data, src_mask, tgt_mask)

    # Print output shape
    print("Input Source Shape:", src_data.shape)
    print("Input Target Shape:", tgt_data.shape)
    print("Output Shape:", output.shape) # Should be (batch_size, output_features) = (8, 4)

    # Check number of parameters
    total_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
