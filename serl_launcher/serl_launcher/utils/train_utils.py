import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

import imageio
import torch
import torch.nn as nn
import numpy as np
import wandb


def load_recorded_video(video_path: str):
    """
    Load a recorded video and return as wandb.Video format.
    
    :param video_path: Path to the video file
    :return: wandb.Video object
    """
    with open(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """
    batch = dict(batch) if not isinstance(batch, dict) else batch
    
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels_tensor = torch.as_tensor(batch["observations"][pixel_key])
            obs_pixels = obs_pixels_tensor[:, :-1, ...]
            next_obs_pixels = obs_pixels_tensor[:, 1:, ...]

            # Update observations
            obs = batch["observations"].copy()
            obs[pixel_key] = obs_pixels
            
            next_obs = batch["next_observations"].copy()
            next_obs[pixel_key] = next_obs_pixels
            
            batch["observations"] = obs
            batch["next_observations"] = next_obs

    return batch


def compute_param_norm(model):
    """Compute parameter norm for logging
    
    Args:
        model: The model to compute parameter norm for
    """
    total_norm = 0.0
    for p in model.parameters():
        # if p.requires_grad:
        param_norm = p.data.norm(dtype=torch.float32)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def compute_grad_norm(model):
    """Compute gradient norm for logging
    
    Args:
        model: The model to compute gradient norm for
    """
    total_norm = 0.0
    for p in model.parameters():
        print(p.grad)
        if p.grad is not None:
            param_norm = p.grad.data.norm(dtype=torch.float32)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def count_params(model):
    """
    Count the total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())

def count_params_trainable(model):
    """
    Count the total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)