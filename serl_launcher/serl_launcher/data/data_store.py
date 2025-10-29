from threading import Lock
from typing import Union, Iterable, Optional, Callable, Sequence

import gymnasium as gym
from serl_launcher.data.replay_buffer import ReplayBuffer

from agentlace.data.data_store import DataStoreBase


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        capacity: int,
        **kwargs
    ):
        ReplayBuffer.__init__(self, capacity=capacity, **kwargs)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).add(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO

    # HACK: we need to load from dataset for DataStoreBase
    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset,
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
        image_size: tuple[int, int] | None = (224, 224),
    ) -> "ReplayBufferDataStore":
        """
        Create a ReplayBufferDataStore from a LeRobotDataset.
        """
        replay_buffer = ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=lerobot_dataset,
            capacity=capacity,
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
        data_store = cls(
            capacity=capacity,
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
        data_store.observations = replay_buffer.observations
        data_store.actions = replay_buffer.actions
        data_store.rewards = replay_buffer.rewards
        data_store.dones = replay_buffer.dones
        data_store.truncateds = replay_buffer.truncateds
        data_store.success_episodes = replay_buffer.success_episodes
        data_store.position = replay_buffer.position
        data_store.size = replay_buffer.size
        data_store.initialized = replay_buffer.initialized
        data_store.episode_starts = replay_buffer.episode_starts
        data_store.episode_success = replay_buffer.episode_success
        data_store.current_episode_start = replay_buffer.current_episode_start
        data_store.current_episode_reward_sum = replay_buffer.current_episode_reward_sum
        data_store.action_dim = replay_buffer.action_dim
        
        if hasattr(replay_buffer, 'has_complementary_info'):
            data_store.has_complementary_info = replay_buffer.has_complementary_info
        if hasattr(replay_buffer, 'complementary_info_keys'):
            data_store.complementary_info_keys = replay_buffer.complementary_info_keys
        if hasattr(replay_buffer, 'complementary_info'):
            data_store.complementary_info = replay_buffer.complementary_info
        if hasattr(replay_buffer, 'next_observations'):
            data_store.next_observations = replay_buffer.next_observations
            
        return data_store
