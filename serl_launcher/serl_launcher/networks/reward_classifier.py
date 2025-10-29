import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import shutil
from pathlib import Path
import time
from typing import Callable, Dict, List
from safetensors.torch import load_file
from termcolor import cprint

from serl_launcher.vision.resnet_v1 import resnetv1_configs
from serl_launcher.common.encoding import EncodingWrapper


class BinaryClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, embed_dim: int = 512):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Create layers
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = self.dense1(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = self.forward(x)
        y = torch.sigmoid(x)
        return y

class NWayClassifier(nn.Module):
    def __init__(self, encoder_def: nn.Module, hidden_dim: int = 256, n_way: int = 12, embed_dim: int = 512):
        super().__init__()
        self.encoder_def = encoder_def
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.embed_dim = embed_dim
        
        # Create layers
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, n_way)

    def forward(self, x, train=False):
        x = self.encoder_def(x, train=train)
        x = self.dense1(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        x = self.forward(x)
        y = F.softmax(x, dim=-1)
        return y

def create_classifier(
    image_keys: List[str],
    use_proprio: bool = False,
    state_dim: int = 14,
    n_way: int = 2,
):
    """
    Create a reward classifier.
    Args:
        image_keys: List of image keys for the encoder
        use_proprio: Whether to use proprioceptive input
        state_dim: Dimension of the proprioceptive input
        n_way: Number of classes for the classifier
    Returns:
        A classifier model
    """
    encoders = {}
    for image_key in image_keys:
        pretrained_encoder = resnetv1_configs["pretrained-resnet18"]()
        encoders[image_key] = pretrained_encoder
    encoder_def = EncodingWrapper(
        encoder=encoders,
        use_proprio=use_proprio,
        state_dim=state_dim,
        enable_stacking=True,
        image_keys=image_keys,
    )

    print(f"{n_way=}")

    embed_dim = 512 * len(image_keys) + (state_dim if use_proprio else 0)

    if n_way <= 2:
        classifier = BinaryClassifier(encoder_def=encoder_def, embed_dim=embed_dim)
    else:
        classifier = NWayClassifier(encoder_def=encoder_def, n_way=n_way, embed_dim=embed_dim)

    return classifier


def load_classifier_func(
    image_keys: List[str],
    checkpoint_path: str,
    use_proprio: bool = False,
    state_dim: int = 14,
    n_way: int = 2,
):
    """
    Load a trained classifier from checkpoint.
    Args:
        image_keys: List of image keys for the encoder
        checkpoint_path: Path to the saved checkpoint
        use_proprio: Whether to use proprioceptive input
        state_dim: Dimension of the proprioceptive input
        n_way: Number of classes for the classifier

    Returns:
        Restored Model
    """
    # Create the model structure
    model = create_classifier(
        image_keys, 
        use_proprio=use_proprio, 
        state_dim=state_dim,
        n_way=n_way
    )

    checkpoint_file = Path(checkpoint_path) / "model.safetensors"
    state_dict = load_file(checkpoint_file)
    model.load_state_dict(state_dict, strict=True)

    cprint(f"Classifier loaded from {checkpoint_file}", "green")

    return model


def outcome_reward_func(obs, action, classifier, env_config):
    """Predict binary outcome reward using classifier."""
    processed_obs = obs.copy()
    for key in obs.keys():
        if key in env_config.classifier_keys:
            # (T, H, W, C) -> (T, C, H, W)
            tensor = torch.from_numpy(obs[key]).permute(0, 3, 1, 2).float().cuda()
            
            # Apply classifier image cropping if configured
            if key in env_config.classifier_image_crop:
                x, y, width, height = env_config.classifier_image_crop[key]
                tensor = tensor[:, :, y:y+height, x:x+width]
            
            processed_obs[key] = tensor
        elif env_config.classifier_use_proprio:
            processed_obs['state'] = torch.from_numpy(obs['state']).float().cuda()

    p = classifier.predict(processed_obs).item()
    r = int(p > 0.75)
    cprint(f"[reward] p={p:.3f}, r={r}", "green" if r == 1 else "white")
    return r

def process_classifier_func(obs, action, classifier, env_config):
    """Predict subtask reward using n-way classifier."""
    processed_obs = obs.copy()
    for key in obs.keys():
        if key in env_config.classifier_keys:
            # (T, H, W, C) -> (T, C, H, W)
            tensor = torch.from_numpy(obs[key]).permute(0, 3, 1, 2).float().cuda()
            
            # Apply classifier image cropping if configured
            if key in env_config.classifier_image_crop:
                x, y, width, height = env_config.classifier_image_crop[key]
                tensor = tensor[:, :, y:y+height, x:x+width]
            
            processed_obs[key] = tensor
        elif env_config.classifier_use_proprio:
            processed_obs['state'] = torch.from_numpy(obs['state']).float().cuda()

    subtask_probs = classifier.predict(processed_obs)
    subtask_labels = subtask_probs.argmax(dim=-1).squeeze()
    subtask_labels = subtask_labels.cpu().numpy()
    subtask_labels = int(subtask_labels)    # get scalar
    return subtask_labels
