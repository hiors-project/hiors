from functools import partial
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
# from gymnasium.utils import seeding
import torch

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
import logging
from pathlib import Path
from pprint import pformat

from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.configs.default import DatasetConfig
from lerobot.common.datasets.transforms import ImageTransformsConfig
from omegaconf import DictConfig


DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            raise TypeError("Unsupported type.")
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
    dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch

def to_dtype(data, dtype=None):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    elif isinstance(data, dict):
        return {k: to_dtype(v, dtype) for k, v in data.items()}
    else:
        return data

class ReplayBufferDataset(torch.utils.data.Dataset):
    """Dataset wrapper for ReplayBuffer."""

    def __init__(self, replay_buffer, sample_args):
        self.replay_buffer = replay_buffer
        self.sample_args = sample_args

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        # Only main process samples from buffer, others return None
        if self.replay_buffer is None:
            return None
        return self.replay_buffer.sample(**self.sample_args)


def collate_fn(batch):
    """Collate function for ReplayBufferDataset."""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return batch[0]  # Return the first (and only) item

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}
def make_dataset(cfg_pi0: PreTrainedConfig, cfg: DictConfig, train: bool=True) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        cfg_pi0 (PreTrainedConfig): A PreTrainedConfig config which contains a DatasetConfig.
        train (bool): Whether to use image transforms or not.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """

    if train:
        transform_config = ImageTransformsConfig(enable=cfg.dataset.image_transforms, max_num_transforms=3)
    else:
        transform_config = ImageTransformsConfig(enable=False)
    dataset_config = DatasetConfig(repo_id=cfg.dataset.repo_id,
                                   episodes=cfg.dataset.episodes,
                                   image_transforms=transform_config,
                                   local_files_only=cfg.dataset.local_files_only,
                                   use_imagenet_stats=cfg.dataset.use_imagenet_stats,
                                   video_backend=cfg.dataset.video_backend)

    image_transforms = (
        ImageTransforms(dataset_config.image_transforms) if dataset_config.image_transforms.enable else None
    )

    lerobot_dir = Path(cfg.dataset.lerobot_dir)
    print(f"Training Model on {cfg.dataset.repo_id} from {lerobot_dir}")
    if isinstance(dataset_config.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(dataset_config.repo_id,
                                         root=lerobot_dir / dataset_config.repo_id,
                                         local_files_only=dataset_config.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        dataset = LeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir / dataset_config.repo_id,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            local_files_only=dataset_config.local_files_only,
        )
    else:
        ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id[0],
                                         root=lerobot_dir / cfg.dataset.repo_id[0],
                                         local_files_only=cfg.dataset.local_files_only)
        delta_timestamps = resolve_delta_timestamps(cfg_pi0, ds_meta)
        dataset = MultiLeRobotDataset(
            dataset_config.repo_id,
            root=lerobot_dir,
            episodes=dataset_config.episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_config.video_backend,
            local_files_only=dataset_config.local_files_only,
        )
        # logging.info(
        #     "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
        #     f"{pformat(dataset.repo_id_to_index , indent=2)}"
        # )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset