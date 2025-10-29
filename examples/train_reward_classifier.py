#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import wandb
from termcolor import cprint

from serl_launcher.data.replay_buffer import ReplayBuffer, OfflineReplayBuffer, concatenate_batch_transitions
from serl_launcher.data.dataset import make_dataset
from serl_launcher.networks.pi0.modeling_pi0 import PI0Config
from serl_launcher.utils.train_utils import compute_param_norm, compute_grad_norm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.logging_utils import print_rich_single_line_metrics
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.utils.launcher import (
    make_wandb_logger,
    make_tensorboard_logger,
)
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


class BinaryOfflineReplayBuffer(OfflineReplayBuffer):
    """Pre-filtered version that only includes positive or negative samples based on episode logic."""
    
    def __init__(self, lerobot_dataset, filter_positive=True, train_split=True, val_split_ratio=0.2, seed=42, **kwargs):
        super().__init__(lerobot_dataset, **kwargs)

        self.filter_positive = filter_positive
        self.train_split = train_split
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.filtered_indices = []

        # Multi-dataset case
        self._filter_multi_dataset()

        # Split into train/validation
        if self.val_split_ratio > 0:
            self._split_train_val()

        self.original_size = self.size
        self.size = len(self.filtered_indices)
        
    def _filter_multi_dataset(self):
        """Filter indices for MultiLeRobotDataset."""
        
        for dataset_idx, dataset in enumerate(self.lerobot_dataset._datasets):
            # Calculate the starting global index for this dataset
            dataset_start_idx = sum(d.num_frames for d in self.lerobot_dataset._datasets[:dataset_idx])
            
            episode_data_index = dataset.episode_data_index
            
            for ep_idx in tqdm(range(len(episode_data_index["from"])), desc="Filtering episodes"):
                ep_start = episode_data_index["from"][ep_idx]
                ep_end = episode_data_index["to"][ep_idx]
                
                if isinstance(ep_start, torch.Tensor):
                    ep_start = ep_start.item()
                if isinstance(ep_end, torch.Tensor):
                    ep_end = ep_end.item()
                ep_start = int(ep_start)
                ep_end = int(ep_end)
                # ep_length = ep_end - ep_start + 1
                ep_length = ep_end - ep_start
                
                for frame_idx in range(ep_length):
                    local_frame_index = ep_start + frame_idx
                    global_frame_index = dataset_start_idx + local_frame_index
                    # Sanity check: ensure global index is within bounds
                    if global_frame_index >= len(self.lerobot_dataset):
                        print(f"Warning: Global index {global_frame_index} exceeds dataset size {len(self.lerobot_dataset)}")
                        continue

                    # Determine if this is a positive sample 
                    # Method #1: (last n frames of episode)
                    # Assume 10hz, last 3s = 30 frames
                    # NOTE: we have viewed one episode and know the last 25 frames are positive
                    # But we set 30 frames to be positive to consider falling down.
                    # is_positive = (ep_start + frame_idx) > ep_end - 30

                    # Method #2: use 'next.reward' key
                    reward = self.lerobot_dataset[global_frame_index]["next.reward"]
                    is_positive = reward > 0.5
                    
                    if (self.filter_positive and is_positive) or (not self.filter_positive and not is_positive):
                        self.filtered_indices.append(global_frame_index)
    
    def _split_train_val(self):
        """Split filtered indices into train and validation sets."""
        if len(self.filtered_indices) == 0:
            return
            
        rng = np.random.RandomState(self.seed)
        indices = np.array(self.filtered_indices)
        
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - self.val_split_ratio))
        
        if self.train_split:
            self.filtered_indices = indices[:split_idx].tolist()
        else:
            self.filtered_indices = indices[split_idx:].tolist()
                    
    def __len__(self):
        return self.size
        
    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for filtered dataset of size {self.size}")
        
        original_index = self.filtered_indices[index]
        if isinstance(original_index, torch.Tensor):
            original_index = original_index.item()
        
        transition = super().__getitem__(original_index)
        
        # Override the reward based on our filtering logic
        reward_value = 1.0 if self.filter_positive else 0.0
        transition["rewards"] = torch.tensor(reward_value, dtype=torch.float32, device=self.device)
        
        return transition


class NWayOfflineReplayBuffer(OfflineReplayBuffer):
    """N-way classification version that assigns subtask labels based on episode progress."""
    
    def __init__(self, lerobot_dataset, n_way=3, train_split=True, val_split_ratio=0.2, seed=42, **kwargs):
        super().__init__(lerobot_dataset, **kwargs)

        self.n_way = n_way
        self.train_split = train_split
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.filtered_indices = []
        self.subtask_labels = {}  # Map from global index to subtask label

        # Multi-dataset case
        self._assign_subtask_labels()

        # Split into train/validation
        if self.val_split_ratio > 0:
            self._split_train_val()

        self.original_size = self.size
        self.size = len(self.filtered_indices)
        
    def _assign_subtask_labels(self):
        """Assign subtask labels based on episode progress."""
        
        for dataset_idx, dataset in enumerate(self.lerobot_dataset._datasets):
            # Calculate the starting global index for this dataset
            dataset_start_idx = sum(d.num_frames for d in self.lerobot_dataset._datasets[:dataset_idx])
            
            episode_data_index = dataset.episode_data_index
            
            for ep_idx in tqdm(range(len(episode_data_index["from"])), desc="Filtering episodes"):
                ep_start = episode_data_index["from"][ep_idx]
                ep_end = episode_data_index["to"][ep_idx]
                
                if isinstance(ep_start, torch.Tensor):
                    ep_start = ep_start.item()
                if isinstance(ep_end, torch.Tensor):
                    ep_end = ep_end.item()
                ep_start = int(ep_start)
                ep_end = int(ep_end)
                # ep_length = ep_end - ep_start + 1
                ep_length = ep_end - ep_start
                
                for frame_idx in range(ep_length):
                    local_frame_index = ep_start + frame_idx
                    global_frame_index = dataset_start_idx + local_frame_index
                    
                    # Sanity check: ensure global index is within bounds
                    if global_frame_index >= len(self.lerobot_dataset):
                        print(f"Warning: Global index {global_frame_index} exceeds dataset size {len(self.lerobot_dataset)}")
                        continue
                    
                    # Assign subtask label based on episode progress
                    progress = frame_idx / max(ep_length - 1, 1)  # 0 to 1
                    subtask_label = min(int(progress * self.n_way), self.n_way - 1)
                    
                    self.subtask_labels[global_frame_index] = subtask_label
                    self.filtered_indices.append(global_frame_index)
    
    def _split_train_val(self):
        """Split filtered indices into train and validation sets."""
        if len(self.filtered_indices) == 0:
            return
            
        rng = np.random.RandomState(self.seed)
        indices = np.array(self.filtered_indices)
        
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - self.val_split_ratio))
        
        if self.train_split:
            self.filtered_indices = indices[:split_idx].tolist()
        else:
            self.filtered_indices = indices[split_idx:].tolist()
                    
    def __len__(self):
        return self.size
        
    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for filtered dataset of size {self.size}")
        
        original_index = self.filtered_indices[index]
        if isinstance(original_index, torch.Tensor):
            original_index = original_index.item()
        
        transition = super().__getitem__(original_index)
        
        # Override the reward with the subtask label
        # subtask_label = self.subtask_labels[original_index]

        subtask_label = transition["complementary_info"]["label"] - 1   # (1 to 12) -> (0 to 11)

        # print(f"{subtask_label=}")
        transition["rewards"] = subtask_label.long()
        
        return transition

def apply_classifier_image_crop(observations, crop_config):
    """Apply cropping to classifier images based on configuration.
    
    Args:
        observations: Dict containing image tensors with shape (B, T, C, H, W)
        crop_config: Dict mapping camera keys to crop parameters [x, y, width, height]
    
    Returns:
        Dict with cropped image tensors
    """
    cropped_observations = {}
    for key, tensor in observations.items():
        if key in crop_config:
            x, y, width, height = crop_config[key]
            # tensor shape: (B, T, C, H, W)
            cropped_tensor = tensor[:, :, :, y:y+height, x:x+width]
            cropped_observations[key] = cropped_tensor
        else:
            cropped_observations[key] = tensor
    return cropped_observations


class MetricsTracker:
    def __init__(self, n_way=2):
        self.n_way = n_way
        self.reset()
        
    def reset(self):
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.count = 0
        
    def update(self, loss, logits, labels):
        self.total_loss += loss.item()
        # Calculate accuracy
        if self.n_way == 2:
            # Binary classification with single output
            predictions = torch.sigmoid(logits) > 0.5
            # print(f"{predictions=}, {labels=}")
            accuracy = (predictions.squeeze(-1) == labels).float().mean()
        else:
            # Multi-class classification
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
        self.total_accuracy += accuracy.item()
        self.count += 1
        
    def compute(self):
        if self.count == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": self.total_loss / self.count,
            "accuracy": self.total_accuracy / self.count
        }


def train_step(model, batch, cfg):
    """Train the model for one step."""
    model.train()

    # (B, T, H, W, C) -> (B, T, C, H, W)
    batch["observations"]["cam_high"] = batch["observations"]["cam_high"].permute(0, 1, 4, 2, 3)
    batch["observations"]["cam_left_wrist"] = batch["observations"]["cam_left_wrist"].permute(0, 1, 4, 2, 3)
    batch["observations"]["cam_right_wrist"] = batch["observations"]["cam_right_wrist"].permute(0, 1, 4, 2, 3)
    batch["observations"]["state"] = batch["observations"]["state"][..., :cfg.environment.state_dim]

    # Apply classifier image cropping if configured
    observations = apply_classifier_image_crop(
        batch["observations"], 
        cfg.environment.classifier_image_crop
    )
    labels = batch["labels"]

    # Forward pass
    logits = model(observations, train=True)
    
    # Calculate loss based on classifier type
    n_way = cfg.environment.classifier_n_way
    if n_way == 2:
        # Binary classification
        logits = logits.squeeze(-1)
        labels = labels.float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    else:
        # Multi-class classification
        labels = labels.long()
        loss = nn.functional.cross_entropy(logits, labels)
    
    return loss, logits, labels

@torch.no_grad()
def validate_step(model, val_dataloader, accelerator, step, logger, cfg):
    """Run validation and log metrics and sample images to wandb."""
    model.eval()
    
    n_way = cfg.environment.classifier_n_way
    val_metrics = MetricsTracker(n_way=n_way)
    
    # Sample a few batches for validation
    num_val_batches = min(10, len(val_dataloader))
    
    val_data_iterator = iter(val_dataloader)
    
    sample_images = []
    
    with torch.no_grad():
        for i in tqdm(range(num_val_batches), desc="Validation", disable=not accelerator.is_main_process):
            try:
                batch = next(val_data_iterator)
            except StopIteration:
                val_data_iterator = iter(val_dataloader)
                batch = next(val_data_iterator)
                
            batch["labels"] = batch["rewards"]
            
            # Prepare observations (same as train_step)
            batch["observations"]["cam_high"] = batch["observations"]["cam_high"].permute(0, 1, 4, 2, 3)
            batch["observations"]["cam_left_wrist"] = batch["observations"]["cam_left_wrist"].permute(0, 1, 4, 2, 3)
            batch["observations"]["cam_right_wrist"] = batch["observations"]["cam_right_wrist"].permute(0, 1, 4, 2, 3)
            batch["observations"]["state"] = batch["observations"]["state"][..., :cfg.environment.state_dim]
            
            # Apply classifier image cropping if configured
            observations = apply_classifier_image_crop(
                batch["observations"], 
                cfg.environment.classifier_image_crop
            )
            labels = batch["labels"]
            
            logits = model(observations, train=False)
            
            # Calculate loss based on classifier type
            if n_way == 2:
                # Binary classification
                logits = logits.squeeze(-1)
                labels = labels.float()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                predictions = torch.sigmoid(logits)
            else:
                # Multi-class classification
                labels = labels.long()
                loss = nn.functional.cross_entropy(logits, labels)
                predictions = torch.softmax(logits, dim=-1)
            
            # Update validation metrics
            val_metrics.update(loss, logits, labels)
            
            # Collect sample images for logging (only from first batch)
            if i == 0 and accelerator.is_main_process and logger is not None:
                batch_size = min(6, logits.shape[0])
                sampled_indices = torch.randperm(logits.shape[0])[:batch_size]
                for j in sampled_indices:
                    for cam_key in list(cfg.environment.classifier_keys):
                        # Get first timestep image: (C, H, W) -> (H, W, C)
                        img = observations[cam_key][j, 0].permute(1, 2, 0).cpu().numpy()
                        
                        if n_way == 2:
                            pred_value = predictions[j].item()
                            pred_class = int(pred_value > 0.5)
                        else:
                            pred_class = torch.argmax(predictions[j]).item()
                            pred_value = predictions[j, pred_class].item()
                            
                        sample_images.append({
                            "image": wandb.Image(img, caption=f"{cam_key}_sample_{j}_class_{pred_class}"),
                            "prediction": pred_value,
                            "predicted_class": pred_class,
                            "true_label": labels[j].item(),
                            "camera": cam_key,
                            "sample_idx": j
                        })
    
    val_results = val_metrics.compute()
    if accelerator.is_main_process and logger is not None:
        log_data = {
            "classifier/val_loss": val_results["loss"],
            "classifier/val_accuracy": val_results["accuracy"]
        }
        if len(sample_images) > 0:
            # Group images by predicted class
            for class_id in range(n_way):
                class_images = [img for img in sample_images if img["predicted_class"] == class_id]
                if class_images:
                    log_data[f"classifier/class_{class_id}_preds"] = [img["image"] for img in class_images[:2]]
        
        logger.log(log_data, step=step)
    
    model.train()
    return val_results

@torch.no_grad()
def validate_step_binary(model, positive_dataloader, negative_dataloader, accelerator, step, logger, cfg):
    """Run binary validation and log metrics and sample images to wandb."""
    model.eval()
    
    n_way = cfg.environment.classifier_n_way
    val_metrics = MetricsTracker(n_way=n_way)
    
    # Sample a few batches for validation
    num_val_batches = 10
    positive_data_iterator = iter(positive_dataloader)
    negative_data_iterator = iter(negative_dataloader)
    
    sample_images = []
    
    with torch.no_grad():
        for i in tqdm(range(num_val_batches), desc="Validation", disable=not accelerator.is_main_process):
            try:
                positive_batch = next(positive_data_iterator)
            except StopIteration:
                positive_data_iterator = iter(positive_dataloader)
                positive_batch = next(positive_data_iterator)
            try:
                negative_batch = next(negative_data_iterator)
            except StopIteration:
                negative_data_iterator = iter(negative_dataloader)
                negative_batch = next(negative_data_iterator)
                
            batch = concatenate_batch_transitions(positive_batch, negative_batch)
            batch["labels"] = (batch["rewards"] > 0).float()
            
            # Prepare observations (same as train_step)
            batch["observations"]["cam_high"] = batch["observations"]["cam_high"].permute(0, 1, 4, 2, 3)
            batch["observations"]["cam_left_wrist"] = batch["observations"]["cam_left_wrist"].permute(0, 1, 4, 2, 3)
            batch["observations"]["cam_right_wrist"] = batch["observations"]["cam_right_wrist"].permute(0, 1, 4, 2, 3)
            batch["observations"]["state"] = batch["observations"]["state"][..., :cfg.environment.state_dim]
            
            # Apply classifier image cropping if configured
            observations = apply_classifier_image_crop(
                batch["observations"], 
                cfg.environment.classifier_image_crop
            )
            labels = batch["labels"]
            
            logits = model(observations, train=False)
            logits = logits.squeeze(-1)
            
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            predictions = torch.sigmoid(logits)
            
            # Update validation metrics
            val_metrics.update(loss, logits, labels)
            
            # Collect sample images for logging (only from first batch)
            if i == 0 and accelerator.is_main_process and logger is not None:
                batch_size = min(6, logits.shape[0])
                sampled_indices = torch.randperm(logits.shape[0])[:batch_size]
                for j in sampled_indices:
                    for cam_key in list(cfg.environment.classifier_keys):
                        # Get first timestep image: (C, H, W) -> (H, W, C)
                        img = observations[cam_key][j, 0].permute(1, 2, 0).cpu().numpy()
                        sample_images.append({
                            "image": wandb.Image(img, caption=f"{cam_key}_sample_{j}"),
                            "prediction": predictions[j].item(),
                            "label": labels[j].item(),
                            "camera": cam_key,
                            "sample_idx": j
                        })
    
    val_results = val_metrics.compute()
    if accelerator.is_main_process and logger is not None:
        log_data = {
            "classifier/val_loss": val_results["loss"],
            "classifier/val_accuracy": val_results["accuracy"]
        }
        if len(sample_images) > 0:
            # Group images by positive/negative predictions
            pos_images = [img for img in sample_images if img["prediction"] > 0.5]
            neg_images = [img for img in sample_images if img["prediction"] <= 0.5]

            if pos_images:
                log_data["classifier/positive_preds"] = [img["image"] for img in pos_images[:3]]
            if neg_images:
                log_data["classifier/negative_preds"] = [img["image"] for img in neg_images[:3]]
        
        logger.log(log_data, step=step)
    
    model.train()
    return val_results

@hydra.main(version_base=None, config_path="../config/", config_name="reward_classifier")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    # set_seed(cfg.seed)    # FIXME: should be comment, as it will reduce the performance

    # Clear previous checkpoints if exists
    if os.path.exists(cfg.checkpoint_path):
        cprint(f"Removing existing checkpoint directory: {cfg.checkpoint_path}", "red")
        shutil.rmtree(cfg.checkpoint_path)
    os.makedirs(cfg.checkpoint_path, exist_ok=True)

    # Initialize Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        project_dir=cfg.checkpoint_path if cfg.checkpoint_path else "./reward_model",
        total_limit=3  # Keep last n checkpoints
    )
    
    # Enable DeepSpeed if config provided
    # deepspeed_plugin = None
    # if FLAGS.deepspeed_config:
    #     deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=FLAGS.deepspeed_config)
    
    accelerator = Accelerator(
        # mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        # deepspeed_plugin=deepspeed_plugin,
        project_config=project_config,
    )

    cfg.batch_size = cfg.train_micro_batch_size_per_gpu * cfg.training.num_gpus

    print(cfg.environment)

    # Setup logging (only on main process)
    logger = None
    if accelerator.is_main_process:
        if cfg.logging.logger == "tensorboard":
            logger = make_tensorboard_logger(
                project=cfg.logging.project,
                description=cfg.exp_name,
                unique_identifier=cfg.unique_identifier,
                debug=False,
            )
        elif cfg.logging.logger == "debug":
            logger = make_wandb_logger(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                description=cfg.exp_name,
                unique_identifier=cfg.unique_identifier,
                debug=True,
                offline=False,
                variant=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        elif cfg.logging.logger == "offline":
            logger = make_wandb_logger(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                description=cfg.exp_name,
                unique_identifier=cfg.unique_identifier,
                debug=False,
                offline=True,
                variant=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        else:  # cfg.logging.logger == "wandb"
            logger = make_wandb_logger(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                description=cfg.exp_name,
                unique_identifier=cfg.unique_identifier,
                debug=False,
                offline=False,
                variant=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )

    # Initialize offline replay buffer from dataset (on all processes)
    pi0_config = PI0Config(
        n_obs_steps=1,
        chunk_size=1,   # we implement action chunking in ReplayBuffer, so n_action_steps is 1
        n_action_steps=1,
    )
    offline_dataset = make_dataset(pi0_config, cfg)
    offline_replay_buffer = OfflineReplayBuffer(
        lerobot_dataset=offline_dataset,
        capacity=cfg.replay_buffer_capacity,
        device="cpu",
        observation_keys=cfg.dataset.observation_keys,
        trans_observation_keys=list(cfg.image_keys) + ["state"],
        use_drq=cfg.dataset.use_drq,
        storage_device="cpu",
        optimize_memory=True,
        success_threshold=cfg.dataset.success_threshold,
    )

    # Loop over offline replay buffer and partition transitions (on all processes)
    cprint(f"Loaded {len(offline_replay_buffer)} transitions from offline dataset", "green")

    # Get n_way from config, defaulting to 3 for subtask classification
    n_way = cfg.environment.classifier_n_way
    
    # Use appropriate buffer type based on n_way
    if n_way == 2:
        # Binary classification
        # Create positive samples buffer
        train_pos_buffer = BinaryOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            filter_positive=True,
            train_split=True,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
        
        # Create negative samples buffer
        train_neg_buffer = BinaryOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            filter_positive=False,
            train_split=True,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
        
        # Combine positive and negative samples for training
        train_buffer = ConcatDataset([train_pos_buffer, train_neg_buffer])
        
        # Create validation buffers
        val_pos_buffer = BinaryOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            filter_positive=True,
            train_split=False,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
        
        val_neg_buffer = BinaryOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            filter_positive=False,
            train_split=False,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
        
        val_buffer = ConcatDataset([val_pos_buffer, val_neg_buffer])
        
    else:
        # Multi-class classification
        train_buffer = NWayOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            n_way=n_way,
            train_split=True,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
        
        val_buffer = NWayOfflineReplayBuffer(
            lerobot_dataset=offline_dataset,
            n_way=n_way,
            train_split=False,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
            capacity=cfg.replay_buffer_capacity,
            device="cpu",
            observation_keys=cfg.dataset.observation_keys,
            trans_observation_keys=list(cfg.image_keys) + ["state"],
            use_drq=cfg.dataset.use_drq,
            storage_device="cpu",
            optimize_memory=True,
            success_threshold=cfg.dataset.success_threshold,
            action_horizon=cfg.environment.action_horizon,
            action_dim=cfg.environment.action_dim,
            image_size=cfg.environment.image_size,
        )
    
    if n_way == 2:
        # Binary classification logging
        cprint(f"Created binary train buffer with {len(train_buffer)} samples", "green")
        cprint(f"  - Positive samples: {len(train_pos_buffer)}", "green")
        cprint(f"  - Negative samples: {len(train_neg_buffer)}", "green")
        cprint(f"Created binary validation buffer with {len(val_buffer)} samples", "green")
        cprint(f"  - Positive samples: {len(val_pos_buffer)}", "green")
        cprint(f"  - Negative samples: {len(val_neg_buffer)}", "green")
    else:
        # Multi-class classification logging
        cprint(f"Created train buffer with {len(train_buffer)} samples", "green")
        cprint(f"Created validation buffer with {len(val_buffer)} samples", "green")
    
    cprint(f"Total dataset size: {len(offline_dataset)}", "green")
    cprint(f"Train samples: {len(train_buffer)}", "green")
    cprint(f"Validation samples: {len(val_buffer)}", "green")

    # Create train and validation dataloaders
    if n_way == 2:
        # For binary classification, use separate dataloaders for balanced sampling
        positive_train_dataloader = torch.utils.data.DataLoader(
            train_pos_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu // 2,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        negative_train_dataloader = torch.utils.data.DataLoader(
            train_neg_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu // 2,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        positive_val_dataloader = torch.utils.data.DataLoader(
            val_pos_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu // 2,
            shuffle=True,  # Shuffle for validation to get diverse valid samples
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        negative_val_dataloader = torch.utils.data.DataLoader(
            val_neg_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu // 2,
            shuffle=True,  # Shuffle for validation to get diverse valid samples
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        train_dataloader = None
        val_dataloader = None
    else:
        # For multi-class, use unified dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_buffer,
            batch_size=cfg.train_micro_batch_size_per_gpu,
            shuffle=True,  # Shuffle for validation to get diverse valid samples
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        positive_train_dataloader = None
        negative_train_dataloader = None
        positive_val_dataloader = None
        negative_val_dataloader = None

    # Create classifier and optimizer
    classifier = create_classifier(
        cfg.environment.classifier_keys, 
        use_proprio=cfg.environment.classifier_use_proprio, 
        state_dim=cfg.environment.state_dim,
        n_way=n_way
    )
    optimizer = optim.AdamW(classifier.parameters(), **cfg.optimizer)

    # Create learning rate scheduler
    lr_scheduler = CosineDecayWithWarmupSchedulerConfig(
        **cfg.lr_scheduler,
    ).build(optimizer, num_training_steps=cfg.num_steps)

    # Prepare with accelerator
    if n_way == 2:
        # Binary classification: prepare separate dataloaders
        classifier, optimizer, positive_train_dataloader, negative_train_dataloader, lr_scheduler = accelerator.prepare(
            classifier, optimizer, positive_train_dataloader, negative_train_dataloader, lr_scheduler
        )
        positive_val_dataloader, negative_val_dataloader = accelerator.prepare(
            positive_val_dataloader, negative_val_dataloader
        )
        train_data_iterator = None
        val_data_iterator = None
    else:
        # Multi-class classification: prepare unified dataloaders
        classifier, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            classifier, optimizer, train_dataloader, lr_scheduler
        )
        val_dataloader = accelerator.prepare(val_dataloader)
        positive_train_dataloader = None
        negative_train_dataloader = None
        positive_val_dataloader = None
        negative_val_dataloader = None

    # TODO: Load checkpoint if exists
    start_step = 0

    # Create metrics tracker
    metrics = MetricsTracker(n_way=n_way)
    timer = Timer()
    
    # Training loop
    if n_way == 2:
        # Binary classification: initialize separate iterators
        positive_data_iterator = iter(positive_train_dataloader)
        negative_data_iterator = iter(negative_train_dataloader)
        train_data_iterator = None
    else:
        # Multi-class classification: initialize unified iterator
        train_data_iterator = iter(train_dataloader)
        positive_data_iterator = None
        negative_data_iterator = None

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    for step in tqdm(
        range(start_step, cfg.num_steps), 
        dynamic_ncols=True, 
        desc=f"Classifier training",
        disable=not accelerator.is_main_process
    ):
        with accelerator.accumulate(classifier):
            timer.tick("total")
            
            with timer.context("sample_replay_buffer"):
                if n_way == 2:
                    # Binary classification: sample from both dataloaders and concatenate
                    try:
                        positive_batch = next(positive_data_iterator)
                    except StopIteration:
                        positive_data_iterator = iter(positive_train_dataloader)
                        positive_batch = next(positive_data_iterator)
                    try:
                        negative_batch = next(negative_data_iterator)
                    except StopIteration:
                        negative_data_iterator = iter(negative_train_dataloader)
                        negative_batch = next(negative_data_iterator)

                    batch = concatenate_batch_transitions(positive_batch, negative_batch)
                    # print(f"{batch['rewards']=}, {batch['rewards'].shape=}")
                    batch["labels"] = (batch["rewards"] > 0).float()
                else:
                    # Multi-class classification: sample from unified dataloader
                    try:
                        batch = next(train_data_iterator)
                    except StopIteration:
                        train_data_iterator = iter(train_dataloader)
                        batch = next(train_data_iterator)

                    batch["labels"] = batch["rewards"]
                
            with timer.context("train"):
                loss, logits, labels = train_step(classifier, batch, cfg)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(classifier.parameters(), cfg.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                metrics.update(loss, logits, labels)

            timer.tock("total")

        # Run validation every eval_period steps
        if step % cfg.eval_period == 0 and step > 0:
        # if step % cfg.eval_period == 0:
            with timer.context("validation"):
                if n_way == 2:
                    # Binary classification: use separate validation dataloaders
                    val_results = validate_step_binary(
                        classifier, positive_val_dataloader, negative_val_dataloader,
                        accelerator, step, logger, cfg
                    )
                else:
                    # Multi-class classification: use unified validation dataloader
                    val_results = validate_step(
                        classifier, val_dataloader, 
                        accelerator, step, logger, cfg
                    )
            if accelerator.is_main_process:
                cprint(f"[Validation] Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}", "green")
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()

        # Logging
        if step % cfg.log_period == 0 and accelerator.is_main_process:
            log_data = {"classifier": metrics.compute()}
            log_data["classifier"]["param_norm"] = compute_param_norm(accelerator.unwrap_model(classifier))
            log_data["classifier"]["grad_norm"] = grad_norm if 'grad_norm' in locals() else 0.0
            log_data["classifier"]["lr"] = lr_scheduler.get_last_lr()[0]
            
            if logger is not None:
                logger.log(log_data, step=step)
                logger.log({"timer": timer.get_average_times()}, step=step)
            
            # print_rich_single_line_metrics(log_data)
            metrics.reset()

        # Save checkpoints
        if step % cfg.checkpoint_period == 0 and step > 0:
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
            
            # Save state for resuming
            # checkpoint_path = os.path.join(FLAGS.checkpoint_path or "./reward_model", f"checkpoint_step_{step}")
            # accelerator.save_state(checkpoint_path)
            # accelerator.print(f"Accelerator checkpoint saved at step {step}")
            
            # Save the model
            if accelerator.is_main_process:
                model_checkpoint_path = os.path.join(cfg.checkpoint_path, str(step))
                unwrapped_classifier = accelerator.unwrap_model(classifier)
                accelerator.save_model(unwrapped_classifier, model_checkpoint_path)
                accelerator.print(f"Model checkpoint saved at step {step} to {model_checkpoint_path}")

    # Final save
    if accelerator.is_main_process:
        cprint("Training completed. Saving final model...", "green")
        final_path = os.path.join(cfg.checkpoint_path, "final")
        unwrapped_classifier = accelerator.unwrap_model(classifier)
        accelerator.save_model(unwrapped_classifier, final_path)
        cprint(f"Final model saved to {final_path}", "green")

    accelerator.print("Training completed")
    accelerator.end_training()


if __name__ == "__main__":
    main()