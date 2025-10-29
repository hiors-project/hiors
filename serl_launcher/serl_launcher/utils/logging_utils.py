import logging
import wandb
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict
import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import cloudpickle as pickle
import time
from collections import deque
from collections import OrderedDict, defaultdict
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional

import numpy as np
import torch
import gymnasium as gym

import draccus
from lerobot.configs.types import FeatureType, NormalizationMode

# ----------------------------------------------------------------------------
# Debug utilities
def print_dict_mean(d, prefix="", important_keys=None):
    """
    Print dictionary values, showing means for most keys but full values for important keys.
    
    Args:
        d: Dictionary to print
        prefix: String prefix for nested keys
        important_keys: List of key names that should print full values instead of means
    """
    if important_keys is None:
        important_keys = []
    
    for key in d:
        full_key = f"{prefix}.{key}" if prefix else key
        value = d[key]
        
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            print_dict_mean(value, full_key, important_keys)
        else:
            # Check if this key should print full information
            should_print_full = (
                key in important_keys or 
                full_key in important_keys or
                any(important_key in full_key for important_key in important_keys)
            )
            
            if should_print_full:
                # Print full value for important keys
                print(f"{full_key}: {value}")
            else:
                # Print mean for regular keys
                # if hasattr(value, 'mean'):
                #     mean_val = value.mean()
                #     print(f"{full_key}: {mean_val}")
                # elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                #     # Handle iterables that don't have mean method
                #     try:
                #         mean_val = np.mean(value)
                #     except:
                #         mean_val = torch.mean(torch.tensor(value, dtype=torch.float32))
                #     print(f"{full_key}: {mean_val}")
                # else:
                # For non-iterable values, just print the value
                print(f"{full_key}: {value}")

def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")

# ----------------------------------------------------------------------------
# Quality of life utilities
def format_value(value):
    if isinstance(value, float):
        if abs(value) < 1e-2:
            return f"{value:.2e}"
        return f"{value:.2f}"
    return str(value)

def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    # Sort metrics by key name for consistent display
    for key in sorted(metrics.keys()):
        value = metrics[key]
        formatted_value = format_value(value)
        table.add_row(key, formatted_value)

    # Create a panel with the table
    panel = Panel(
        table,
        title="Metrics",
        expand=False,
        border_style="bold green",
    )

    # Print the panel
    rprint(panel)
# ----------------------------------------------------------------------------


"""Wrapper that tracks the cumulative rewards and episode lengths."""
class RecordEpisodeStatistics(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since beginning of episode>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     "final_observation": "<array of length num-envs>",
        ...     "_final_observation": "<boolean array of length num-envs>",
        ...     "final_info": "<array of length num-envs>",
        ...     "_final_info": "<boolean array of length num-envs>",
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since beginning of episode>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        super().__init__(env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.episode_count = 0
        self.episode_start_times: np.ndarray = None
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = self.env.reset(**kwargs)
        self.episode_start_times = np.full(
            self.num_envs, time.perf_counter(), dtype=np.float32
        )
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                }
                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
            self.return_queue.extend(self.episode_returns[dones])
            self.length_queue.extend(self.episode_lengths[dones])
            self.episode_count += num_dones
            self.episode_lengths[dones] = 0
            self.episode_returns[dones] = 0
            self.episode_start_times[dones] = time.perf_counter()
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

def register_features_types():
    draccus.decode.register(FeatureType, lambda x: FeatureType[x])
    draccus.encode.register(FeatureType, lambda x: x.name)

    draccus.decode.register(NormalizationMode, lambda x: NormalizationMode[x])
    draccus.encode.register(NormalizationMode, lambda x: x.name)
