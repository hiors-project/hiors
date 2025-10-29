#!/usr/bin/env python

# A modified version of the replay buffer from:
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import time
from typing import Callable, Optional, Sequence, TypedDict
from termcolor import cprint
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset
import numpy as np
# import cv2
from tqdm import tqdm, trange
# import queue
# import threading
# from concurrent.futures import ThreadPoolExecutor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from serl_launcher.networks.reward_classifier import naive_reward_func

class Transition(TypedDict):
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: float
    next_observations: dict[str, torch.Tensor]
    dones: bool
    truncateds: bool
    complementary_info: dict[str, torch.Tensor | float | int] | None = None

class BatchTransition(TypedDict):
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: dict[str, torch.Tensor]
    masks: torch.Tensor
    dones: torch.Tensor
    # truncateds: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str=None):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """
    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and (shape[0] == 3 or shape[-1] == 3):   # (C,H,W) or (H,W,C)
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["observations"] = {
        key: torch.cat(
            [left_batch_transitions["observations"][key], right_batch_transition["observations"][key]],
            dim=0,
        )
        for key in left_batch_transitions["observations"]
    }

    # Concatenate basic fields
    left_batch_transitions["actions"] = torch.cat(
        [left_batch_transitions["actions"], right_batch_transition["actions"]], dim=0
    )
    left_batch_transitions["rewards"] = torch.cat(
        [left_batch_transitions["rewards"], right_batch_transition["rewards"]], dim=0
    )

    # Concatenate next_observations fields
    left_batch_transitions["next_observations"] = {
        key: torch.cat(
            [left_batch_transitions["next_observations"][key], right_batch_transition["next_observations"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_observations"]
    }

    # Concatenate masks and dones fields
    left_batch_transitions["masks"] = torch.cat(
        [left_batch_transitions["masks"], right_batch_transition["masks"]], dim=0
    )
    left_batch_transitions["dones"] = torch.cat(
        [left_batch_transitions["dones"], right_batch_transition["dones"]], dim=0
    )

    # Handle complementary_info
    # TODO: we ignore complementary_info for now, because of state chunk
    left_batch_transitions["complementary_info"] = None
    # left_info = left_batch_transitions.get("complementary_info")
    # right_info = right_batch_transition.get("complementary_info")
    # if right_info is not None and left_info is not None:
    #     for key in right_info:
    #         print(f"{key=}")
    #         if key in left_info:
    #             left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
    #         else:
    #             left_info[key] = right_info[key]

    return left_batch_transitions

# Resize without padding
def resize_no_pad(img, width, height, mode="bilinear"):
    """
    Resize a batch of images to the target width and height without padding.
    The image will be stretched to fit the target size, possibly changing the aspect ratio.
    Args:
        img: torch.Tensor of shape (B, C, H, W)
        width: target width
        height: target height
        mode: interpolation mode (default: 'bilinear')
    Returns:
        torch.Tensor of shape (B, C, height, width)
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    interpolate_params = {
        'size': (height, width),
        'mode': mode,
    }
    if mode != "nearest":
        interpolate_params['align_corners'] = False
    return F.interpolate(img, **interpolate_params)

class ReplayBuffer(Dataset):
    """
    The online replay buffer is used to sample from the online data.
    We want to avoid using torch.utils.data.DataLoader sampling to handle variable length of transitions.
    """
    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        observation_keys: Sequence[str] | None = None,
        trans_observation_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        success_threshold: float = 0.5,
        has_done_key: bool = False,
        has_reward_key: bool = False,
        action_horizon: int = 50,
        action_dim: int = 16,
        image_size: Optional[tuple[int, int]] = (224, 224),
    ):
        """
        Initialize the online replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            device: Device for tensor operations during sampling
            observation_keys: Keys for observations
            trans_observation_keys: Keys to return when sampling (subset of observation_keys)
            image_size: Size to resize images to (width, height)
            image_augmentation_function: Function for image augmentation
            use_drq: Whether to use DrQ image augmentation
            storage_device: Device for storing transitions
            optimize_memory: Whether to optimize memory usage
            success_threshold: Threshold for determining episode success based on rewards
            action_horizon: Length of action chunks for sampling
            action_dim: Dimension of the action space
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory
        self.success_threshold = success_threshold
        
        # Set up observation keys
        self.observation_keys = observation_keys
        self.trans_observation_keys = trans_observation_keys if trans_observation_keys is not None else observation_keys
        
        # Create mapping from observation keys to transformed keys
        if observation_keys is not None and trans_observation_keys is not None:
            assert len(self.observation_keys) == len(self.trans_observation_keys), \
                "observation_keys and trans_observation_keys must have the same length"
            
            self.obs_to_trans_obs_map = {}
            for i, key in enumerate(self.observation_keys):
                self.obs_to_trans_obs_map[key] = self.trans_observation_keys[i]

            self.trans_obs_to_obs_map = {v: k for k, v in self.obs_to_trans_obs_map.items()}
        
        self.image_size = image_size
        # Set up image augmentation
        self.image_augmentation_function = image_augmentation_function
        # if image_augmentation_function is None:
        #     base_function = functools.partial(random_shift, pad=4)
        #     self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq
        
        # Initialize complementary info tracking
        self.has_done_key = has_done_key
        self.has_reward_key = has_reward_key
        # Dynamic complementary info keys based on actual data
        self.complementary_info_keys = ["action_is_pad"]
        self.has_complementary_info = False  # Will be set based on actual data
        
        self.episode_starts = []
        self.episode_success = []
        self.current_episode_start = 0
        self.current_episode_reward_sum = 0.0
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def __len__(self):
        return self.size

    def move_transition_to_device(self, transition: Transition) -> Transition:
        device = torch.device(self.storage_device)
        non_blocking = device.type == "cuda"

        def to_tensor_if_needed(val):
            if isinstance(val, np.ndarray):
                return torch.from_numpy(val).to(device, non_blocking=non_blocking)
            elif isinstance(val, (int, float)):
                return torch.tensor(val).to(device, non_blocking=non_blocking)
            elif isinstance(val, torch.Tensor):
                return val.to(device, non_blocking=non_blocking)
            else:
                return torch.tensor(val).to(device) # HACK: for bool
                # return val

        transition["observations"] = {
            key: to_tensor_if_needed(val) for key, val in transition["observations"].items()
            if key in self.trans_observation_keys
        }
        transition["actions"] = to_tensor_if_needed(transition["actions"])
        transition["rewards"] = to_tensor_if_needed(transition["rewards"])
        transition["dones"] = to_tensor_if_needed(transition["dones"])
        if "truncateds" in transition:
            transition["truncateds"] = to_tensor_if_needed(transition["truncateds"])
        else:
            transition["truncateds"] = transition["dones"]

        transition["next_observations"] = {
            key: to_tensor_if_needed(val) for key, val in transition["next_observations"].items()
            if key in self.trans_observation_keys
        }

        if transition.get("complementary_info") is not None:
            for key, val in transition["complementary_info"].items():
                transition["complementary_info"][key] = to_tensor_if_needed(val)

        return transition

    def _initialize_storage(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        observation_shapes = {key: val.shape for key, val in observations.items()}
        action_shape = actions.shape

        # Pre-allocate tensors for storage
        self.observations = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in observation_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        self.action_horizon = action_shape[0]
        self.action_dim = action_shape[1]

        if not self.optimize_memory:
            # Standard approach: store observations and next_observations separately
            self.next_observations = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in observation_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_observations = self.observations  # Just a reference for API consistency

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        
        self.success_episodes = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.success_episodes.fill_(False)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}
        if self.has_complementary_info:
            # Only include keys that actually have data
            for key, value in complementary_info.items():
                if value is not None:
                    self.complementary_info_keys.append(key)
                    if isinstance(value, torch.Tensor):
                        value_shape = value.shape
                        self.complementary_info[key] = torch.empty(
                            (self.capacity, *value_shape), device=self.storage_device
                        )
                    elif isinstance(value, (int, float, bool)):
                        self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                    else:
                        raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def add(
        self,
        transition: dict,
    ):
        """
        Saves a transition, ensuring tensors are stored on the designated storage device.
        transition is without batch dimension
        """
        transition = self.move_transition_to_device(transition)
        observations = transition["observations"]
        actions = transition["actions"]
        rewards = transition["rewards"]
        next_observations = transition["next_observations"]
        dones = transition["dones"]
        truncateds = transition["truncateds"]
        complementary_info = transition.get("complementary_info", None)

        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(observations=observations, actions=actions, complementary_info=complementary_info)

        # Store the transition in pre-allocated tensors
        for key in self.observations:
            self.observations[key][self.position].copy_(observations[key])

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_observations[key][self.position].copy_(next_observations[key])

        self.actions[self.position].copy_(actions)
        self.rewards[self.position] = rewards
        self.dones[self.position] = dones
        self.truncateds[self.position] = truncateds

        self.success_episodes[self.position] = False

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info, handling missing keys gracefully
            for key in self.complementary_info_keys:
                if key in complementary_info and key in self.complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][self.position].copy_(value)
                    elif isinstance(value, (int, float, bool)):
                        self.complementary_info[key][self.position] = value

        # Track episode reward sum
        self.current_episode_reward_sum += float(rewards)

        # Check if episode has ended
        if dones or truncateds:
            # Determine episode success based on cumulative reward and threshold
            episode_success = self.current_episode_reward_sum >= self.success_threshold
            
            # Mark all transitions in the current episode as successful or unsuccessful
            self._mark_current_episode_success(episode_success)
            
            # Reset episode tracking
            self.current_episode_reward_sum = 0.0

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _mark_current_episode_success(self, episode_success: bool):
        """
        Mark all transitions in the current episode as successful or unsuccessful.
        This is called internally when an episode ends.
        
        Args:
            episode_success: Whether the episode was successful
        """
        if not self.initialized:
            return
            
        # Mark all transitions from current episode start to current position as successful
        episode_end = self.position
        
        if self.current_episode_start <= episode_end:
            # Episode doesn't wrap around
            self.success_episodes[self.current_episode_start:episode_end + 1] = episode_success
        else:
            # Episode wraps around the buffer
            self.success_episodes[self.current_episode_start:] = episode_success
            self.success_episodes[:episode_end + 1] = episode_success
        
        self.episode_starts.append(self.current_episode_start)
        self.episode_success.append(episode_success)

        self.current_episode_start = (self.position + 1) % self.capacity

    def get_success_count(self) -> int:
        """Get the number of successful transitions in the buffer."""
        if not self.initialized:
            return 0
        return int(torch.sum(self.success_episodes[:self.size]).item())

    def _get_action_chunk_for_indices(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get action chunks for given indices using state chunks from complementary_info.
        
        The transitions structure is:
        - Each transition contains variable-length state chunks in complementary_info["state_chunk"]
        - State chunks represent intermediate poses between observations
        - When sampling, we want action chunks of fixed length (action_horizon) starting from each transition
        - If not enough states available, repeat the last available state
        
        Args:
            indices: Tensor of indices to get action chunks for [batch_size]
            
        Returns:
            action_chunks: Action chunks tensor [batch_size, action_horizon, action_dim]
            action_is_pad: Boolean mask indicating padded actions [batch_size, action_horizon]
        """
        batch_size = len(indices)
        action_dim = self.action_dim
        device = self.device
        capacity = self.capacity
        storage_device = self.storage_device
        
        # Pre-allocate result tensors
        action_chunks = torch.zeros(
            (batch_size, self.action_horizon, action_dim), 
            device=device, 
            dtype=torch.float32
        )
        action_is_pad = torch.zeros(
            (batch_size, self.action_horizon), 
            device=device, 
            dtype=torch.bool
        )
        
        if not self.has_complementary_info or "state_chunk" not in self.complementary_info:
            # cprint("[ReplayBuffer] No state_chunk in complementary_info, returning stored action chunks.", "red")
            action_chunks = self.actions[indices].to(device)
            # For stored action chunks, assume all actions are valid (not padded)
            action_is_pad.fill_(False)
            return action_chunks, action_is_pad
            
        state_chunks_tensor = self.complementary_info["state_chunk"]
        
        for batch_idx, buffer_idx in enumerate(indices):
            actual_buffer_idx = int(buffer_idx.item())
            
            # Convert logical index to actual buffer index
            actual_idx = (self.position - self.size + actual_buffer_idx) % capacity
            
            # Collect states starting from this transition
            collected_states = []
            current_transition_idx = actual_idx
            episode_ended = False
            
            # Continue collecting until we have enough states or reach episode end
            while len(collected_states) < self.action_horizon:
                if current_transition_idx >= len(state_chunks_tensor):
                    episode_ended = True
                    break
                    
                state_chunk = state_chunks_tensor[current_transition_idx]
                # Handle variable length state chunks
                if state_chunk.dim() == 2:
                    # Filter out zero/padding rows (assuming zero padding)
                    non_zero_mask = torch.any(state_chunk != 0, dim=1)
                    valid_states = state_chunk[non_zero_mask]
                    if len(valid_states) > 0:
                        collected_states.append(valid_states)
                else:
                    # Single state
                    if torch.any(state_chunk != 0):
                        collected_states.append(state_chunk.unsqueeze(0))

                current_transition_idx = (current_transition_idx + 1) % capacity
                
                # Stop if we've reached the current buffer position (end of valid data)
                if current_transition_idx == self.position:
                    episode_ended = True
                    break
                # Stop if we've hit an episode boundary (done=True)
                if self.dones[current_transition_idx]:
                    episode_ended = True
                    break
            
            # Concatenate all collected states
            if collected_states:
                all_states = torch.cat(collected_states, dim=0)
                state_dim = min(all_states.shape[1], action_dim)
                # Fill action chunk
                num_states_available = len(all_states)
                if num_states_available >= self.action_horizon:
                    # We have enough states
                    action_chunks[batch_idx, :, :state_dim] = all_states[:self.action_horizon, :state_dim].to(device)
                    # All actions are valid (within episode)
                    action_is_pad[batch_idx, :] = False
                else:
                    # Not enough states, fill what we have and repeat the last state
                    action_chunks[batch_idx, :num_states_available, :state_dim] = all_states[:, :state_dim].to(device)
                    # Mark valid actions as not padded
                    action_is_pad[batch_idx, :num_states_available] = False
                    
                    # Repeat the last state for remaining positions and mark as padded
                    if num_states_available > 0:
                        last_state = all_states[-1:, :state_dim].to(device)
                        for i in range(num_states_available, self.action_horizon):
                            action_chunks[batch_idx, i, :state_dim] = last_state[0]
                            # Mark repeated/padded actions as padded
                            action_is_pad[batch_idx, i] = True
                    else:
                        # No valid states collected, mark all as padded
                        action_is_pad[batch_idx, :] = True
            else:
                # No states collected, mark all as padded
                action_is_pad[batch_idx, :] = True
                
        return action_chunks, action_is_pad

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions from the online buffer with action chunking."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")
        
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")
        
        # Ensure we don't sample more than available
        # actual_batch_size = min(batch_size, self.size)

        actual_batch_size = batch_size
        
        high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size
        idx = torch.randint(low=0, high=high, size=(actual_batch_size,), device=self.storage_device)
        
        actual_idx = (self.position - self.size + idx) % self.capacity

        image_keys = [k for k in self.observations if k.startswith("observation.image")] if self.use_drq else []

        batch_observations = {}
        batch_next_observations = {}

        # First pass: load all observation tensors to target device
        for key in self.observations:
            batch_observations[key] = self.observations[key][actual_idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach, load next_observations directly
                batch_next_observations[key] = self.next_observations[key][actual_idx].to(self.device)
            else:
                # Memory-optimized approach, get next_observations from the next index
                next_idx = (actual_idx + 1) % self.capacity
                batch_next_observations[key] = self.observations[key][next_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            all_images = []
            for key in image_keys:
                all_images.append(batch_observations[key])
                all_images.append(batch_next_observations[key])

            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for observations, batch_size for next_observations)
                # Observations start at index i*2*actual_batch_size and take up batch_size slots
                batch_observations[key] = augmented_images[i * 2 * actual_batch_size : (i * 2 + 1) * actual_batch_size]
                # Next observations start after the observations at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_observations[key] = augmented_images[(i * 2 + 1) * actual_batch_size : (i + 1) * 2 * actual_batch_size]

        batch_actions, batch_action_is_pad = self._get_action_chunk_for_indices(actual_idx)

        batch_rewards = self.rewards[actual_idx].to(self.device)
        batch_dones = self.dones[actual_idx].to(self.device).float()

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][actual_idx].to(self.device)
        else:
            batch_complementary_info = {}
        
        # Always include action_is_pad in complementary_info
        # batch_complementary_info["action_is_pad"] = batch_action_is_pad

        # print(f"[ReplayBuffer] {batch_next_observations['state']=}, {batch_next_observations['state'].dtype=}")
        return BatchTransition(
            observations=batch_observations,
            actions=batch_actions,
            rewards=batch_rewards,
            next_observations=batch_next_observations,
            masks=1.0 - batch_dones,
            dones=batch_dones,
            complementary_info=batch_complementary_info,
        )

    # Ref: https://discuss.pytorch.org/t/how-to-use-dataloader-for-replaybuffer/50879/5
    # def __iter__(self): # TODO: could be optimized like _get_async_iterator in lerobot
    #     while True:
    #         idx = np.random.randint(0, len(self) - 1)
    #         yield self[idx]

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object,
        with the format "o_0, action_0=[state_chunk_0, ..., state_chunk_{action_horizon-1}]" for action chunking.
        Each transition is stored as (o_t, action_t=[state_chunk_t, state_chunk_{t+1}, ..., state_chunk_{t+action_horizon-1}]).
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action", use state chunks as actions for action chunking (flattened)
        # Determine action dimension from actual data - use observation state dimension
        sample_obs = self.observations["state"][0]
        # action_dim = sample_obs.shape[-1]

        features["action"] = {"dtype": "float32", "shape": [self.action_horizon * self.action_dim]}

        features["intervention"] = {"dtype": "bool", "shape": [self.action_horizon]}

        features["next.reward"] = {"dtype": "float32", "shape": (1,)}
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add observation keys
        for transformed_key in self.observations:
            # NOTE: we need to squeeze to remove the time dimension
            key = self.trans_obs_to_obs_map[transformed_key]
            sample_val = self.observations[transformed_key][0].squeeze(0)
            if key.startswith("observation.images"):
                sample_val = sample_val.permute(2, 0, 1)
            # (H, W, C) -> (C, H, W)
            f_info = guess_feature_info(t=sample_val)
            # print(f"[to_lerobot_dataset] {key=}, {f_info=}")
            features[key] = f_info

        # Add complementary_info keys if available
        # if self.has_complementary_info:
        #     for key in self.complementary_info_keys:
        #         sample_val = self.complementary_info[key][0]
        #         if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
        #             sample_val = sample_val.unsqueeze(0)
        #         f_info = guess_feature_info(t=sample_val)
        #         features[key] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )
        # Convert transitions into episodes and frames
        episode_index = 0
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(episode_index=episode_index)

        frame_idx_in_episode = 0
        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            # Get the observation at time t
            obs_t = {}
            for transformed_key in self.observations:
                # NOTE: we need to squeeze to remove the time dimension
                key = self.trans_obs_to_obs_map[transformed_key]
                if key.startswith("observation.images"):
                    # (H, W, C) -> (C, H, W)
                    obs_t[key] = self.observations[transformed_key][actual_idx].squeeze(0).permute(2, 0, 1).cpu().numpy()
                else:
                    obs_t[key] = self.observations[transformed_key][actual_idx].squeeze(0).cpu().numpy()

            # Create single frame for this transition with action chunk
            frame_dict = {}
            
            # Add observations
            for transformed_key in self.observations:
                key = self.trans_obs_to_obs_map[transformed_key]
                frame_dict[key] = obs_t[key]

            # Use the new consistent function to get action chunks
            # Convert single index to tensor for compatibility
            index_tensor = torch.tensor([actual_idx], device=self.storage_device)
            action_chunk_tensor, action_is_pad_tensor = self._get_action_chunk_for_indices(index_tensor)
            action_chunk = action_chunk_tensor[0].cpu().numpy()  # Remove batch dimension, [action_horizon, action_dim]
            # action_is_pad_chunk = action_is_pad_tensor[0].cpu().numpy()  # Remove batch dimension, [action_horizon]
            
            # Flatten for storage [action_horizon * action_dim]
            frame_dict["action"] = action_chunk.reshape(-1)

            intervention_chunk = np.zeros(self.action_horizon, dtype=np.bool_)
            frame_dict["intervention"] = intervention_chunk

            frame_dict["next.reward"] = self.rewards[actual_idx].item()
            frame_dict["next.done"] = self.dones[actual_idx].item()
            
            lerobot_dataset.add_frame(frame_dict)
            frame_idx_in_episode += 1

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode(task=task_name)
                episode_index += 1
                frame_idx_in_episode = 0
                lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(
                    episode_index=episode_index
                )

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode(task=task_name)

        # lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    def _lerobotdataset_to_transitions(self, lerobot_dataset: LeRobotDataset, idx: int) -> dict:
        """
        Convert a dataset sample at the given index to a transition format.
        Handles chunked actions and observation validity flags from saved datasets.
        """
        # NOTE: Be careful! Lerobot data may be already augmented
        # lerobot_start_time = time.time()
        current_sample = lerobot_dataset[idx]

        # print(f"{current_sample.keys()=}")
        # ['observation.state', 'action', 'observation.velocity', 'observation.images.cam_high', 'observation.images.cam_low', 
        # 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'label', 'timestamp', 'frame_index', 
        # 'episode_index', 'index', 'task_index', 'action_is_pad', 'task', 'dataset_index']

        # print(f"[lerobotdataset_to_transitions] lerobot time: {time.time() - lerobot_start_time:.4f} seconds")

        # ----- 1) Current observations -----
        # NOTE: add time dimension and resize image at here, because add() in the main loop will have time dimension
        observations = {}
        for key in self.observation_keys:
            transformed_key = self.obs_to_trans_obs_map[key]
            val = current_sample[key]
            # print(f"[lerobotdataset_to_transitions] {key=} {val.shape=}")
            # (C, H, W) or (state_dim)

            # resize images if needed
            # if key.startswith("observation.images"):
            #     # print(f"{val=}")  # tensor, (C, H, W)
            #     val = val.permute(1, 2, 0).cpu().numpy()    # (C, H, W) -> (H, W, C)
            #     val = cv2.resize(val, (self.image_size[1], self.image_size[0]))
            #     val = torch.from_numpy(val) # (H, W, C)

            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            observations[transformed_key] = val
        
        # ----- 2) Action -----
        actions = current_sample["action"]
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        
        # print(f"[lerobotdataset_to_transitions] {actions.shape=}")

        action_dim = self.action_dim
        if actions.shape[-1] == self.action_horizon * action_dim:
            # If action is flattened, reshape it
            actions = actions.reshape(self.action_horizon, action_dim)

        # ----- 2.5) Intervention information -----
        
        # ----- 3) Reward -----
        if self.has_reward_key and "next.reward" in current_sample:
            reward = float(current_sample["next.reward"].item())
        else:
            # reward = naive_reward_func(current_sample["observation.state"], actions.cpu().numpy())
            if hasattr(self, "lerobot_dataset"):
                index = current_sample["index"]
                episode_index = current_sample["episode_index"]
                dataset_index = current_sample["dataset_index"]
                # ep_end_start_time = time.time()
                ep_end = self.lerobot_dataset._datasets[dataset_index].episode_data_index["to"][episode_index]
                # print(f"ep_end time: {time.time() - ep_end_start_time}")
                # reward = 1.0 if index > ep_end - 25 else 0.0
                reward = 1.0 if index == ep_end - 1 else 0.0
            else:
                raise ValueError(
                    "Reward key not found in current sample and no lerobot_dataset available to compute reward."
                )
                # reward = 0
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward)
        
        # ----- 4) Done flag -----
        if self.has_done_key and "next.done" in current_sample:
            done = bool(current_sample["next.done"].item())
        else:
            # Infer from episode boundaries
            done = False
            if idx == len(lerobot_dataset) - 1:
                done = True
            elif idx < len(lerobot_dataset) - 1:
                next_sample = lerobot_dataset[idx + 1]
                if next_sample["episode_index"] != current_sample["episode_index"]:
                    done = True
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done)

        # ----- 5) Next observations -----
        next_observations = observations.copy()  # default
        if not done and (idx < len(lerobot_dataset) - 1):
            next_sample = lerobot_dataset[idx + 1]
            if next_sample["episode_index"] == current_sample["episode_index"]:
                # Build next_state from the same keys
                next_state_data: dict[str, torch.Tensor] = {}
                for key in self.observation_keys:
                    transformed_key = self.obs_to_trans_obs_map[key]
                    val = next_sample[key]
                    next_state_data[transformed_key] = val
                next_observations = next_state_data
        
        # ----- 6) Complementary info -----
        complementary_info = None
        if self.has_complementary_info:
            complementary_info = {}
            for key in self.complementary_info_keys:
                clean_key = key
                dataset_key = f"complementary_info.{key}" if f"complementary_info.{key}" in current_sample else key
                if dataset_key in current_sample:
                    val = current_sample[dataset_key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(val)
                    complementary_info[clean_key] = val
        
        # Apply image processing if needed
        for key in self.observation_keys:
            transformed_key = self.obs_to_trans_obs_map[key]
            if key.startswith("observation.images"):
                # NOTE: bilinear is used to "match" the INTER_LINEAR mode of cv2.resize
                observations[transformed_key] = resize_no_pad(
                    observations[transformed_key].unsqueeze(0), *self.image_size, mode="bilinear",
                ).permute(0, 2, 3, 1)   # (B, C, H, W) -> (B, H, W, C)
                next_observations[transformed_key] = resize_no_pad(
                    next_observations[transformed_key].unsqueeze(0), *self.image_size, mode="bilinear",
                ).permute(0, 2, 3, 1)   # (B, C, H, W) -> (B, H, W, C)
            elif key == "observation.state":
                # Add time dimension
                observations[transformed_key] = observations[transformed_key].unsqueeze(0)
                next_observations[transformed_key] = next_observations[transformed_key].unsqueeze(0)
            else:
                # Add time dimension
                observations[transformed_key] = observations[transformed_key].unsqueeze(0)
                next_observations[transformed_key] = next_observations[transformed_key].unsqueeze(0)
    
        return {
            "observations": observations,
            "actions": actions,
            "rewards": reward,
            "next_observations": next_observations,
            "dones": done,
            "truncateds": done,  # Use same as done for now
            "complementary_info": complementary_info,
        }

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        capacity: int,
        device: str = "cpu",
        observation_keys: Sequence[str] | None = None,
        trans_observation_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        success_threshold: float = 0.5,
        action_horizon: int = 50,
        action_dim: int = 16,
        image_size: Optional[tuple[int, int]] = (224, 224),
    ) -> "ReplayBuffer":
        """
        Load transitions from a LeRobotDataset into a ReplayBuffer.,
        with the structure [(o_t, [s_t, s_{t+dt_1}, ..., s_{t+dt_action_horizon}]), ...].
        By loading from a dataset, we do not need to store state chunks (intermediate steps).

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.
            success_threshold (float): Threshold for determining episode success based on rewards.
            action_horizon (int): Length of action chunks for sampling.
            action_dim (int): Dimension of the action space.
            image_size (Optional[tuple[int, int]]): Size for resizing images (H, W).

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )
        
        # Check if dataset has done/reward keys and observation validity
        sample = lerobot_dataset[0]
        has_done_key = "next.done" in sample
        has_reward_key = "next.reward" in sample

        # complementary_info_keys = [key for key in sample.keys() if key.startswith("complementary_info.")]
        # has_complementary_info = len(complementary_info_keys) > 0

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            observation_keys=observation_keys,
            trans_observation_keys=trans_observation_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
            success_threshold=success_threshold,
            has_done_key=has_done_key,
            has_reward_key=has_reward_key,
            action_horizon=action_horizon,
            action_dim=action_dim,
            image_size=image_size,
        )

        # Convert dataset to transitions, filtering out None values from intermediate frames
        list_transition = []
        for idx in trange(len(lerobot_dataset), desc="Loading ReplayBuffer from dataset in disk"):
            transition = replay_buffer._lerobotdataset_to_transitions(lerobot_dataset=lerobot_dataset, idx=idx)
            if transition is not None:  # Filter out None values from intermediate frames
                list_transition.append(transition)

        if not list_transition:
            raise ValueError("No valid transitions found in the dataset. All samples were intermediate frames.")

        # Initialize the buffer with the first transition to set up storage tensors
        first_transition = list_transition[0]
        first_observations = {k: v.to(device) for k, v in first_transition["observations"].items()}
        first_actions = first_transition["actions"].to(device)

        if first_actions.shape[-1] == action_horizon * action_dim:   # [1, action_horizon * action_dim]
            first_actions = first_actions.reshape(action_horizon, action_dim)

        # Ensure complementary info includes required keys for action chunking
        first_complementary_info = None
        if (
            "complementary_info" in first_transition
            and first_transition["complementary_info"] is not None
        ):
            first_complementary_info = {
                k: v.to(device) for k, v in first_transition["complementary_info"].items()
            }
        else:
            first_complementary_info = {}
    
        replay_buffer._initialize_storage(
            observations=first_observations, actions=first_actions, complementary_info=first_complementary_info
        )

        # Fill the buffer with all transitions
        for data in list_transition:
            transition = {
                "observations": data["observations"],
                "actions": data["actions"],
                "rewards": data["rewards"],
                "next_observations": data["next_observations"],
                "dones": data["dones"],
                "truncateds": data["dones"],  # NOTE: Truncation are not supported yet in lerobot dataset
                "complementary_info": data.get("complementary_info", None),
            }
            replay_buffer.add(transition)

        return replay_buffer


class OfflineReplayBuffer(ReplayBuffer):
    """
    The offline replay buffer is used to sample from a LeRobotDataset.
    We want to use torch.utils.data.DataLoader to speed up the sampling process.
    """
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        capacity: int,
        device: str = "cpu",
        observation_keys: Sequence[str] | None = None,
        trans_observation_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        success_threshold: float = 0.5,
        action_horizon: int = 50,
        action_dim: int = 16,
        image_size: Optional[tuple[int, int]] = (224, 224),
    ):
        """
        Initialize the offline replay buffer.
        
        Args:
            lerobot_dataset: The LeRobotDataset to sample from
            capacity: Maximum number of transitions to store
            device: Device for tensor operations during sampling
            observation_keys: Keys for observations in the dataset
            trans_observation_keys: Keys to return when sampling (subset of observation_keys)
            image_augmentation_function: Function for image augmentation
            use_drq: Whether to use DrQ image augmentation
            success_threshold: Threshold for determining episode success based on rewards
            action_horizon: Length of action chunks for sampling
            action_dim: Dimension of the action space
            image_size: Size for resizing images (H, W)
        """
        # Set up observation keys
        if observation_keys is None:
            # Infer from dataset
            sample = lerobot_dataset[0]
            observation_keys = [key for key in sample.keys() if key.startswith("observation.")]
        
        # Initialize parent class with dummy capacity (not used for offline buffer)
        super().__init__(
            capacity=capacity,  # do not use capacity for offline buffer
            device=device,
            observation_keys=observation_keys,
            trans_observation_keys=trans_observation_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
            success_threshold=success_threshold,
            action_horizon=action_horizon,
            action_dim=action_dim,
            image_size=image_size,
        )
        
        self.lerobot_dataset = lerobot_dataset
        self.size = len(lerobot_dataset)
        
        # Check if dataset has done/reward keys and observation validity
        sample = lerobot_dataset[0]
        # self.has_done_key = "next.done" in sample
        # self.has_reward_key = "next.reward" in sample
        self.has_done_key = False
        self.has_reward_key = False
        # self.complementary_info_keys = [key for key in sample.keys() if key.startswith("complementary_info.")]
        self.complementary_info_keys = ["action_is_pad", "index", "episode_index", "dataset_index", "label"]
        self.has_complementary_info = len(self.complementary_info_keys) > 0

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> dict:
        """
        Get item for offline mode (reading from dataset).
        """
        if index < 0 or index >= len(self.lerobot_dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.lerobot_dataset)}")
        
        # Convert dataset sample to transition format
        transition = self._lerobotdataset_to_transitions(lerobot_dataset=self.lerobot_dataset, idx=index)

        # Convert to tensor format and move to device
        observations = {}
        next_observations = {}
        
        # Process observations
        for key in self.observation_keys:
            transformed_key = self.obs_to_trans_obs_map[key]
            obs_tensor = transition["observations"][transformed_key]
            next_obs_tensor = transition["next_observations"][transformed_key]
            
            observations[transformed_key] = obs_tensor.to(self.device)
            next_observations[transformed_key] = next_obs_tensor.to(self.device)

        # Process other transition data
        actions = transition["actions"][:, :self.action_dim]
        actions = actions.to(dtype=torch.float32, device=self.device)

        rewards = transition["rewards"].to(dtype=torch.float32, device=self.device)
        dones = transition["dones"].to(dtype=torch.float32, device=self.device)
        
        # Process complementary info
        complementary_info = None
        if self.has_complementary_info and transition["complementary_info"] is not None:
            complementary_info = {}
            for key, val in transition["complementary_info"].items():
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val)
                complementary_info[key] = val.to(dtype=torch.float32, device=self.device)
        
        ret = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "masks": 1.0 - dones,
            "dones": dones,
            # "truncateds": dones,
            "complementary_info": complementary_info,
        }
        return ret
