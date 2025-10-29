from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


class EncodingWrapper(nn.Module):
    """
    Encodes observations into a single flat encoding, adding additional
    functionality for adding proprioception and stopping the gradient.

    Args:
        encoder: The encoder network.
        use_proprio: Whether to concatenate proprioception (after encoding).
        state_dim: The dimension of the proprioceptive state.
        proprio_latent_dim: The dimension of the proprioceptive latent space.
        enable_stacking: Whether to enable stacking of observations.
        image_keys: The keys in the observation dict that correspond to images.
    """

    def __init__(
        self,
        encoder: Dict[str, nn.Module],
        use_proprio: bool,
        state_dim: int = 14,
        proprio_latent_dim: int = 64,
        enable_stacking: bool = False,
        image_keys: Iterable[str] = ("image",),
    ):
        super().__init__()
        self.encoder = nn.ModuleDict(encoder)
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.enable_stacking = enable_stacking
        self.image_keys = image_keys
        
        if use_proprio:
            self.state_dense = nn.Linear(
                state_dim,
                proprio_latent_dim
            )
            # Initialize weights using Xavier uniform
            nn.init.xavier_uniform_(self.state_dense.weight)
            
            self.state_layer_norm = nn.LayerNorm(proprio_latent_dim)

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        stop_gradient=False,
        is_encoded=False,
        train=False,
    ) -> torch.Tensor:
        """
        observations:
            image: (batch, seq_len, H, W, C)
            state: (batch, seq_len, state_dim)
        Returns:
            encoded: (batch, seq_len, latent_dim)
        """
        assert self.image_keys or self.use_proprio, "No image keys provided and use_proprio=False. Must have either images or proprioception."

        # encode images with encoder
        encoded = []
        for image_key in self.image_keys:
            image = observations[image_key]
            if not is_encoded:
                if self.enable_stacking:
                    # Combine stacking and channels into a single dimension
                    if len(image.shape) == 4:
                        pass
                    elif len(image.shape) == 5:
                        image = rearrange(image, "B T C H W -> B (T C) H W")
                    else:
                        raise ValueError(f"Image shape must be (T, C, H, W) or (B, T, C, H, W), got {image.shape}")

            image = self.encoder[image_key](image, encode=not is_encoded)

            # if stop_gradient:
            #     image = image.detach()

            encoded.append(image)

        # Concatenate image encodings only if there are any images
        if encoded:
            encoded = torch.cat(encoded, dim=-1)
        else:
            encoded = None
        if self.use_proprio:
            # project state to embeddings as well
            state = observations["state"]
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(state.shape) == 2:
                    state = rearrange(state, "T C -> (T C)")
                    if encoded is not None:
                        encoded = encoded.reshape(-1)
                if len(state.shape) == 3:
                    state = rearrange(state, "B T C -> B (T C)")
            state = self.state_dense(state)
            state = self.state_layer_norm(state)
            state = torch.tanh(state)
            
            if encoded is not None:
                encoded = torch.cat([encoded, state], dim=-1)
            else:
                encoded = state

        # encoded = nn.Dropout(rate=0.25)(encoded, deterministic=not train)

        return encoded
