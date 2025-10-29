from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        input_dim: int,
        activations: str = "silu",
        activate_final: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.activations = activations
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        
        # Create layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        current_size = input_dim
        for i, size in enumerate(self.hidden_dims):
            self.layers.append(nn.Linear(current_size, size))
            current_size = size
            
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    self.dropouts.append(nn.Dropout(p=self.dropout_rate))
                else:
                    self.dropouts.append(nn.Identity())
                    
                if self.use_layer_norm:
                    self.layer_norms.append(nn.LayerNorm(size))
                else:
                    self.layer_norms.append(nn.Identity())
            else:
                self.dropouts.append(nn.Identity())
                self.layer_norms.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = getattr(F, self.activations)

        for i, size in enumerate(self.hidden_dims):
            x = self.layers[i](x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.dropouts[i](x)
                x = self.layer_norms[i](x)
                x = activations(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(
        self,
        features: int,
        act: Callable,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Create layers
        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()
            
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)
        else:
            self.layer_norm = nn.Identity()
            
        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)

    def forward(self, x):
        residual = x
        
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        # For MLPResNetBlock, input and output should have the same dimensions
        # since we're using features for both input and output
        return residual + x


class MLPResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 2,
        out_dim: int = 256,
        dropout_rate: float = None,
        use_layer_norm: bool = False,
        hidden_dim: int = 256,
        activations: Callable = F.silu,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations
        
        # Create layers
        self.input_dense = nn.Linear(input_dim, self.hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.blocks.append(MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            ))
            
        self.output_dense = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        """
        x = self.input_dense(x)    # input_size -> hidden_dim
        
        for block in self.blocks:
            x = block(x)

        x = self.activations(x)
        x = self.output_dense(x)
        return x


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.value
