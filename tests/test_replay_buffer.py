#!/usr/bin/env python3
"""
Pytest for ReplayBuffer class
Usage:
    pytest tests/test_replay_buffer.py -s
"""
import os
import tempfile
import shutil
import torch
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from serl_launcher.data.replay_buffer import ReplayBuffer, OfflineReplayBuffer
from serl_launcher.data.dataset import make_dataset
from serl_launcher.networks.pi0.modeling_pi0 import PI0Config


@pytest.fixture
def observation_keys():
    """Default observation keys for testing."""
    return ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
            "observation.images.cam_high", "observation.state"]

@pytest.fixture
def trans_observation_keys():
    """Default transformed observation keys for testing."""
    return ["cam_left_wrist", "cam_right_wrist", "cam_high", "state"]

@pytest.fixture
def test_dimensions():
    """Standard test dimensions."""
    return {
        "obs_dim": 8,
        "action_dim": 16,
        "image_shape": (224, 224, 3),
        "state_chunk_len": 5
    }


def create_test_transition(step_idx: int, episode_idx: int, 
                         test_dims: Dict[str, Any], action_horizon: int) -> Dict[str, Any]:
    """
    Create a test transition with deterministic arange data for verification.
    
    Args:
        step_idx: Current step within episode (0-9)
        episode_idx: Episode identifier (0 or 1)
        test_dims: Dictionary with test dimensions
        include_observation: Whether this transition has full observation
    
    Returns:
        Transition dictionary with observations, actions, rewards, etc.
    """
    obs_dim = test_dims["obs_dim"]
    action_dim = test_dims["action_dim"]
    image_shape = test_dims["image_shape"]
    state_chunk_len = test_dims["state_chunk_len"]
    
    # Create observation data using arange for determinism
    base_offset = step_idx * 100
    observations = {}
    observations["state"] = torch.arange(base_offset, base_offset + obs_dim, dtype=torch.float32).unsqueeze(0)
    observations["cam_high"] = torch.arange(base_offset, base_offset + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)
    observations["cam_right_wrist"] = torch.arange(base_offset + 1000, base_offset + 1000 + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)
    observations["cam_left_wrist"] = torch.arange(base_offset + 2000, base_offset + 2000 + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)

    # Create next observation (incremented by 50)
    next_base_offset = base_offset + 50
    next_observations = {}
    next_observations["state"] = torch.arange(next_base_offset, next_base_offset + obs_dim, dtype=torch.float32).unsqueeze(0)
    next_observations["cam_high"] = torch.arange(next_base_offset, next_base_offset + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)
    next_observations["cam_right_wrist"] = torch.arange(next_base_offset + 1000, next_base_offset + 1000 + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)
    next_observations["cam_left_wrist"] = torch.arange(next_base_offset + 2000, next_base_offset + 2000 + np.prod(image_shape), dtype=torch.float32).reshape(image_shape).unsqueeze(0)

    # Create action using arange (use obs_dim for action dimension to match state dimension)
    action = torch.randn(action_horizon, action_dim, dtype=torch.float32)

    # Create reward and done flag - use unique values for verification
    reward = 0
    done = (step_idx == 9)  # End of episode at step 9
    truncated = False

    # Create state chunks for action chunking - consecutive states
    state_chunk = torch.arange(
        base_offset, 
        base_offset + state_chunk_len * obs_dim, 
        dtype=torch.float32
    ).reshape(state_chunk_len, obs_dim)
    
    # Create intervention chunk with some True values for testing
    intervention_chunk = torch.zeros(state_chunk_len, dtype=torch.bool)
    if step_idx >= 5:  # Simulate interventions in later steps
        intervention_chunk[-2:] = True
    
    state_timestamp_chunk = torch.arange(state_chunk_len, dtype=torch.float64)

    # Create complementary info
    complementary_info = {
        "state_chunk": state_chunk,
        "intervention_chunk": intervention_chunk,
        "state_timestamp_chunk": state_timestamp_chunk,
    }

    return {
        "observations": observations,
        "actions": action,
        "rewards": reward,
        "next_observations": next_observations,
        "dones": done,
        "truncateds": truncated,
        "complementary_info": complementary_info
    }


def create_episode_data(episode_idx: int, num_transitions: int, test_dims: Dict[str, Any], action_horizon: int) -> list:
    """
    Create data for a complete episode.
    
    Args:
        episode_idx: Episode identifier
        num_transitions: Number of transitions in episode
        test_dims: Dictionary with test dimensions
        
    Returns:
        List of transitions for the episode
    """
    transitions = []
    
    for step_idx in range(num_transitions):
        transition = create_test_transition(step_idx, episode_idx, test_dims, action_horizon)
        transitions.append(transition)
        
    return transitions

def test_replay_buffer(observation_keys, trans_observation_keys, test_dimensions):
    """Test saving to and loading from LeRobot format with action chunking."""
    print("=== Starting test_replay_buffer ===")
    
    test_root = Path(os.environ["HF_LEROBOT_HOME"]) / "dobot"
    repo_id = "test"
    dataset_path = test_root / repo_id
    # action_horizon = 10
    action_horizon = 40
    num_transitions = 5  # Number of transitions per episode
    batch_size = 1
    print(f"Test root: {test_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Action horizon: {action_horizon}")
    
    if dataset_path.exists():
        print("Cleaning existing dataset path...")
        shutil.rmtree(dataset_path)

    # Create an empty replay buffer
    print("Creating empty replay buffer...")
    print(f"Observation keys: {observation_keys}")
    print(f"Transformed observation keys: {trans_observation_keys}")
    buffer = ReplayBuffer(
        capacity=100,
        device="cpu",
        observation_keys=observation_keys,
        trans_observation_keys=trans_observation_keys,
        action_horizon=action_horizon,
        use_drq=False  # Disable DrQ to avoid image augmentation issues
    )
    print(f"Buffer created with capacity: {buffer.capacity}")
    
    # Create and add two episodes of data
    print("Creating episode data...")
    episode1_data = create_episode_data(episode_idx=0, num_transitions=num_transitions, test_dims=test_dimensions, action_horizon=action_horizon)
    episode2_data = create_episode_data(episode_idx=1, num_transitions=num_transitions, test_dims=test_dimensions, action_horizon=action_horizon)
    print(f"Episode 1 created with {len(episode1_data)} transitions")
    print(f"Episode 2 created with {len(episode2_data)} transitions")
    
    all_transitions = episode1_data + episode2_data
    print(f"Total transitions to add: {len(all_transitions)}")
    
    for i, transition in enumerate(all_transitions):
        print(f"transition {i}: {transition['complementary_info']['state_chunk']}")
        buffer.add(transition)
    
    print(f"Final buffer size: {len(buffer)}")
    
    # Verify buffer size
    print("Sampling from buffer...")
    batch = buffer.sample(batch_size=batch_size)
    print(f"Sampled batch keys: {list(batch.keys())}")
    assert "actions" in batch

    print(f"Original buffer batch['actions'].shape: {batch['actions'].shape}")  # [5, 10, 8]
    print(f"Original buffer batch['actions'][0]: {batch['actions'][0]}")

    assert len(batch["actions"].shape) == 3  # [batch_size, action_horizon, obs_dim]
    assert batch["actions"].shape[1] == action_horizon
    # assert batch["actions"].shape[2] == test_dimensions["obs_dim"]    # NOTE: 16 vs. 8

    # Save to LeRobot format
    print("Saving to LeRobot format...")
    lerobot_dataset = buffer.to_lerobot_dataset(
        repo_id=repo_id,
        fps=10,
        root=dataset_path,
    )
    print("LeRobot dataset saved successfully")

    # Load back from LeRobot format
    print("Loading from LeRobot format...")
    replay_cfg = OmegaConf.create({
        "dataset": {
            "repo_id": repo_id,
            "episodes": None,
            "image_transforms": False,
            "local_files_only": True,
            "use_imagenet_stats": False,
            "video_backend": "pyav",
            "lerobot_dir": str(test_root)
        }
    })
    cfg_pi0 = PI0Config(
        n_obs_steps=1,
        chunk_size=1,   # we implement action chunking in ReplayBuffer, so n_action_steps is 1
        n_action_steps=1,
    )
    replay_dataset = make_dataset(cfg_pi0, replay_cfg)
    loaded_buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=replay_dataset,
        capacity=100,
        device="cpu",
        observation_keys=observation_keys,
        trans_observation_keys=trans_observation_keys,
        use_drq=False,
        storage_device="cpu",
        optimize_memory=True,
        success_threshold=0.5,
        action_horizon=action_horizon,
    )
    print(f"Loaded buffer size: {len(loaded_buffer)}")
    
    # Test sampling from loaded buffer and verify action chunking
    print("Sampling from loaded buffer...")
    batch = loaded_buffer.sample(batch_size=batch_size)
    print(f"Loaded buffer sample keys: {list(batch.keys())}")

    assert "actions" in batch

    print(f"Loaded buffer batch['actions'].shape: {batch['actions'].shape}")
    print(f"Loaded buffer batch['actions'][0]: {batch['actions'][0]}")

    assert len(batch["actions"].shape) == 3  # [batch_size, action_horizon, obs_dim]
    assert batch["actions"].shape[1] == action_horizon
    # assert batch["actions"].shape[2] == test_dimensions["obs_dim"]

    # Test: save again
    print("Testing save again...")
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    loaded_buffer.to_lerobot_dataset(
        repo_id=repo_id,
        fps=10,
        root=dataset_path,
    )
    print("Second save completed successfully")


def test_offline_replay_buffer(observation_keys, trans_observation_keys, test_dimensions):
    """Test saving to and loading from LeRobot format with action chunking."""
    print("=== Starting test_replay_buffer ===")
    
    test_root = Path(os.environ["HF_LEROBOT_HOME"]) / "dobot"
    repo_id = "20250620_3_moving_800ml_tora2"
    dataset_path = test_root / repo_id
    action_horizon = 10
    print(f"Test root: {test_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Action horizon: {action_horizon}")

    # Basic test for OfflineReplayBuffer: init and sample
    print("Creating PI0Config and dummy replay_cfg...")
    cfg_pi0 = PI0Config(
        n_obs_steps=1,
        chunk_size=10,
        n_action_steps=10,
    )
    replay_cfg = OmegaConf.create({
        "dataset": {
            "repo_id": repo_id,
            "episodes": None,
            "image_transforms": False,
            "local_files_only": True,
            "use_imagenet_stats": False,
            "video_backend": "pyav",
            "lerobot_dir": str(test_root)
        }
    })

    print("Making offline dataset...")
    offline_dataset = make_dataset(cfg_pi0, replay_cfg)
    print("Initializing OfflineReplayBuffer...")
    buffer = OfflineReplayBuffer(
        lerobot_dataset=offline_dataset,
        capacity=100,
        device="cpu",
        observation_keys=observation_keys,
        trans_observation_keys=trans_observation_keys,
        use_drq=False,
        storage_device="cpu",
        optimize_memory=True,
        success_threshold=0.5,
        action_horizon=action_horizon,
    )
    print(f"OfflineReplayBuffer initialized with size: {len(buffer)}")

    print("Sampling from OfflineReplayBuffer...")
    sample = buffer[0]
    print(f"Sampled batch keys: {list(sample.keys())}")
    assert "actions" in sample
    print(f"Sampled batch['actions'].shape: {sample['actions'].shape}")
    assert len(sample["actions"].shape) == 2
    assert sample["actions"].shape[0] == action_horizon

if __name__ == "__main__":
    pytest.main([__file__, "-v"])