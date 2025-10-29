# !/usr/bin/env python3

import os
import torch
from termcolor import cprint

from serl_launcher.agents.continuous.sac_pi0 import SACPiAgent
from serl_launcher.vision.data_augmentations import batched_random_crop, batched_color_transform
from serl_launcher.utils.train_utils import count_params, count_params_trainable
from serl_launcher.common.typing import Batch


def make_sac_pi0_agent(
    sample_obs,
    sample_action,
    # OpenPI model configuration
    openpi_config_name: str = "pi0_xtrainer",
    openpi_checkpoint_dir: str = None,
    # Model architecture for critic networks
    image_keys=("image",),
    critic_keys=("image",),
    critic_network_kwargs=None,
    encoder_type="resnet-pretrained",
    use_proprio=True,
    # SAC parameters
    discount=0.97,
    soft_target_update_rate=0.005,
    reward_bias=0.0,
    critic_ensemble_size=2,
    cql_weight=1.0,
    distill_weight=0.0,
    # Optimizer parameters
    optimizer_kwargs=None,
    lr_scheduler_kwargs=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):  
    # Create SAC Pi0 agent and get unified optimizer
    agent, unified_optimizer, lr_scheduler = SACPiAgent.create_pixels(
        sample_obs,
        sample_action,
        # OpenPI configuration
        openpi_config_name=openpi_config_name,
        openpi_checkpoint_dir=openpi_checkpoint_dir,
        # Model architecture
        encoder_type=encoder_type,
        use_proprio=use_proprio,
        image_keys=image_keys,
        critic_keys=critic_keys,
        critic_network_kwargs=critic_network_kwargs,
        # SAC parameters
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=None,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        reward_bias=reward_bias,
        cql_weight=cql_weight,
        distill_weight=distill_weight,
        # Unified optimizer
        optimizer_kwargs=optimizer_kwargs,
        # Data augmentation
        # augmentation_function=make_batch_augmentation_func(image_keys),
        augmentation_function=None,
        device=device,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    
    trainable_params, total_params = get_param_count(agent)
    cprint(f"SAC Pi0 Agent Created:", "green")
    cprint(f"  OpenPI Config: {openpi_config_name}", "cyan")
    cprint(f"  Trainable parameters: {trainable_params/1e6:.2f}M (OpenPI + critic networks)", "yellow")
    cprint(f"  Total parameters: {total_params/1e6:.2f}M", "yellow")
    
    return agent, unified_optimizer, lr_scheduler


#########################
# Helper functions
#########################

def get_param_count(agent):
    """
    Returns the number of trainable parameters and total parameters of an agent.
    """
    # Count total parameters - assumes agent has state dict or direct access to parameters
    param_count_dict = {}
    param_count_dict["actor"] = count_params(agent.actor)
    param_count_dict["critic"] = count_params(agent.critic)
    param_count_dict["temperature"] = count_params(agent.temperature)
    
    for key, value in param_count_dict.items():
        print(f"{key}: {value/1e6:.2f}M")

    trainable_param_count_dict = {}
    trainable_param_count_dict["actor"] = count_params_trainable(agent.actor)
    trainable_param_count_dict["critic"] = count_params_trainable(agent.critic)
    trainable_param_count_dict["temperature"] = count_params_trainable(agent.temperature)

    for key, value in trainable_param_count_dict.items():
        print(f"{key} (trainable): {value/1e6:.2f}M")
    
    total_params = sum(param_count_dict.values())
    trainable_params = sum(trainable_param_count_dict.values())
    
    return trainable_params, total_params
    
def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(observations):
        """Data augmentation function that works with PyTorch tensors"""
        
        # Image augmentation
        for pixel_key in image_keys:
            if pixel_key in observations:
                # Apply random crop with padding
                observations[pixel_key] = batched_random_crop(
                    observations[pixel_key], padding=8, num_batch_dims=2
                )
                # Uncomment for color augmentation
                # observations[pixel_key] = batched_color_transform(
                #     observations[pixel_key],
                #     brightness=0.1,
                #     contrast=0.1,
                #     saturation=0.1,
                #     hue=0.03,
                #     to_grayscale_prob=0.0,
                #     color_jitter_prob=1.0,
                #     apply_prob=0.5,
                #     shuffle=True,
                #     num_batch_dims=2,
                # )
        
        # State (proprioception) augmentation
        # if "state" in observations:
        #     state_noise_scale = 0.1
        #     state = observations["state"]
        #     # Add Gaussian noise to state
        #     noise = torch.randn_like(state) * state_noise_scale
        #     augmented_state = state + noise
        #     observations["state"] = augmented_state
            
        return observations
    
    def augment_batch(batch: Batch) -> Batch:
        """Augment a batch of data"""
        obs = data_augmentation_fn(batch["observations"])
        next_obs = data_augmentation_fn(batch["next_observations"])
        
        # Update batch with augmented observations
        batch = batch.copy() if hasattr(batch, 'copy') else dict(batch)
        batch["observations"] = obs
        batch["next_observations"] = next_obs
        
        return batch
    
    return augment_batch


def make_wandb_logger(
    project: str = "hiors-pytorch",
    entity: str = "hiors-pytorch",
    description: str = "serl_launcher",
    unique_identifier: str = "",
    debug: bool = False,
    offline: bool = False,
    variant: dict = {},
):
    from serl_launcher.common.wandb import WandBLogger
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "entity": entity,
            "exp_descriptor": description,
            "unique_identifier": unique_identifier,
            "tag": description,
        }
    )

    wandb_output_dir = "wandb"
    os.makedirs(wandb_output_dir, exist_ok=True)
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        wandb_output_dir=wandb_output_dir,
        variant=variant,
        debug=debug,
        offline=offline,
    )
    return wandb_logger


def make_tensorboard_logger(
    project: str = "hiors-pytorch",
    description: str = "serl_launcher",
    unique_identifier: str = "",
    debug: bool = False,
    log_dir: str = None,
):
    from serl_launcher.common.tensorboard import TensorBoardLogger
    tensorboard_config = TensorBoardLogger.get_default_config()
    tensorboard_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "unique_identifier": unique_identifier,
            "log_dir": log_dir,
        }
    )

    tensorboard_output_dir = log_dir if log_dir is not None else "tensorboard"
    os.makedirs(tensorboard_output_dir, exist_ok=True)
    tensorboard_logger = TensorBoardLogger(
        tensorboard_config=tensorboard_config,
        tensorboard_output_dir=tensorboard_output_dir,
        variant={},
        debug=debug,
    )
    return tensorboard_logger
