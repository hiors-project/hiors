from typing import Callable, Optional, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str) -> Callable:
    """Get activation function from string name."""
    activation_map = {
        'relu': F.relu,
        'gelu': F.gelu,
        'silu': F.silu,
        'elu': F.elu,
    }
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}. Supported: {list(activation_map.keys())}")
    return activation_map[activation]

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Create query, key, value projections
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        
        # Output projection
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.query, self.key, self.value, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute query, key, value
        q = self.query(x)  # [batch, seq_len, hidden_dim]
        k = self.key(x)    # [batch, seq_len, hidden_dim]
        v = self.value(x)  # [batch, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (for autoregressive generation)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Transpose back and reshape
        out = out.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        out = out.contiguous().view(batch_size, seq_len, hidden_dim)
        
        # Final projection
        out = self.proj(out)
        
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * hidden_dim
            
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.activation = get_activation_fn(activation)
        
        # Feed-forward layers
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
            
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        
        # Layer normalization
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        else:
            self.ln1 = None
            self.ln2 = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (like GPT)
        if self.ln1 is not None:
            attn_input = self.ln1(x)
        else:
            attn_input = x
            
        # Self-attention with residual connection
        attn_output = self.attention(attn_input, mask=mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        if self.ln2 is not None:
            ff_input = self.ln2(x)
        else:
            ff_input = x
            
        ff_output = self.feed_forward(ff_input)
        x = x + ff_output
        
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 544,  # Default from launcher config
        num_layers: int = 8,
        num_heads: int = 8,
        hidden_dims: Sequence[int] = [512],
        out_dim: int = 512,  # Now explicitly required for output dimension
        ff_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        activations: str = "silu",  # Changed to string
        max_seq_len: int = 1024,
        pooling_method: str = "mean",  # Options: "mean", "max", "last", "first"
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        hidden_dims = hidden_dims[-1]
        out_dim = hidden_dims
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.activations = activations
        self.max_seq_len = max_seq_len
        self.pooling_method = pooling_method
        
        if ff_dim is None:
            ff_dim = 4 * hidden_dims
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims)
        
        # Position embeddings (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dims)
        
        # Input dropout
        if dropout_rate > 0:
            self.input_dropout = nn.Dropout(dropout_rate)
        else:
            self.input_dropout = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dims,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                use_layer_norm=use_layer_norm,
                activation=activations,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        if use_layer_norm:
            self.ln_f = nn.LayerNorm(hidden_dims)
        else:
            self.ln_f = None
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims, out_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Input and output projections
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Position embeddings
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def _pool_sequence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool the sequence dimension to get a single vector per batch item.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask of shape [batch_size, seq_len] where True indicates valid positions
            
        Returns:
            Pooled tensor of shape [batch_size, hidden_dim]
        """
        if self.pooling_method == "mean":
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [batch_size, seq_len, hidden_dim]
                masked_x = x * mask_expanded
                sum_x = masked_x.sum(dim=1)  # [batch_size, hidden_dim]
                valid_lengths = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
                return sum_x / (valid_lengths + 1e-8)  # Avoid division by zero
            else:
                # Simple mean pooling
                return x.mean(dim=1)  # [batch_size, hidden_dim]
        
        elif self.pooling_method == "max":
            if mask is not None:
                # Masked max pooling
                masked_x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                return masked_x.max(dim=1)[0]  # [batch_size, hidden_dim]
            else:
                return x.max(dim=1)[0]  # [batch_size, hidden_dim]
        
        elif self.pooling_method == "last":
            if mask is not None:
                # Get the last valid position for each sequence
                lengths = mask.sum(dim=1) - 1  # [batch_size]
                batch_indices = torch.arange(x.size(0), device=x.device)
                return x[batch_indices, lengths]  # [batch_size, hidden_dim]
            else:
                return x[:, -1]  # [batch_size, hidden_dim]
        
        elif self.pooling_method == "first":
            return x[:, 0]  # [batch_size, hidden_dim]
        
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, out_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, hidden_dim]
        x = x + pos_emb.unsqueeze(0)  # Broadcast over batch dimension
        
        # Input dropout
        if self.input_dropout is not None:
            x = self.input_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final layer norm
        if self.ln_f is not None:
            x = self.ln_f(x)
        
        # Pool the sequence dimension
        x = self._pool_sequence(x, mask=mask)  # [batch_size, hidden_dim]
        
        # Output projection
        x = self.output_proj(x)  # [batch_size, out_dim]
        
        return x
