import logging
import copy
from functools import partial
from typing import Iterable, Optional, Tuple, Dict, Any
from termcolor import cprint
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, Data
from serl_launcher.networks.actor_critic_nets import Critic, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.transformer import Transformer
from serl_launcher.utils.logging_utils import print_dict_mean
from serl_launcher.networks.pi0.modeling_pi0 import PI0Policy, PI0Config

from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)

logger = logging.getLogger(__name__)


class SACPiAgent(nn.Module):
    """
    SAC agent with PI0 policy and small critic networks (PyTorch implementation).
    
    This agent uses a pre-trained PI0 model as the policy which gets fine-tuned
    during SAC training, while training small critic networks for value estimation.
    Both the PI0 policy parameters and critic networks are updated during training.
    """

    def __init__(
        self,
        config: dict,
        actor: PI0Policy,
        critic: nn.Module,
        temperature: nn.Module,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.config = config
        self.device = device
        self.action_dim = config["action_dim"]
        
        # Networks
        self.actor = actor
        self.ref_actor = copy.deepcopy(actor) if config["distill_weight"] > 0 else None
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        # turn off gradients
        if self.ref_actor is not None:
            for param in self.ref_actor.parameters():
                param.requires_grad_(False)
        for param in self.target_critic.parameters():
            param.requires_grad_(False)
        self.temperature = temperature
        self.axis_to_id = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5, "rw": 6, "gripper": 7}
        
    def _batch_transform_observations(self, observations: Data) -> Data:
        """
        Transform batched SERL observations to PI0 format.
        Args:
            observations: Batched observations with format [B, T, H, W, C] for images
        Returns:
            observations in PI0 format
        """
        observations = copy.deepcopy(observations)
        batch_size = observations["state"].shape[0]
    
        pi0_obs = observations.copy()

        # Handle images (B, T, H, W, C) -> (B, T, C, H, W)
        pi0_obs["cam_high"] = observations["cam_high"].permute(0, 1, 4, 2, 3)
        pi0_obs["cam_left_wrist"] = observations["cam_left_wrist"].permute(0, 1, 4, 2, 3)
        pi0_obs["cam_right_wrist"] = observations["cam_right_wrist"].permute(0, 1, 4, 2, 3)

        # Add task prompts (default to a fixed prompt for now)
        pi0_obs["task"] = ["put the objects in the box"] * batch_size

        return pi0_obs

    def forward_critic(
        self,
        observations: Data,
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for critic network.

        Args:
            observations: Input observations in PI0 format
            actions: (B, T, action_dim) tensor of actions
        """
        # Ensure actions are in the right format (B, T, A) and truncate to 8 dimensions
        if actions.dim() == 2:  # (B, A) -> (B, 1, A)
            actions = actions.unsqueeze(1)
        actions = actions[..., :self.action_dim]
        
        if train:
            self.critic.train()
        else:
            self.critic.eval()
        return self.critic(observations, actions)
    
    @torch.no_grad()
    def forward_target_critic(
        self,
        observations: Data,
        actions: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        # Ensure actions are in the right format (B, T, A) and truncate to 8 dimensions
        if actions.dim() == 2:  # (B, A) -> (B, 1, A)
            actions = actions.unsqueeze(1)
        actions = actions[..., :self.action_dim]
        
        self.target_critic.eval()
        return self.target_critic(observations, actions)

    def forward_policy(
        self,
        observations: Data,
        deterministic: bool = False,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for PI0 policy network.
        
        Args:
            observations: Input observations in PI0 format
            deterministic: Whether to use deterministic sampling
        Returns:
            actions: [batch_size, action_horizon, action_dim]
            log_probs: [batch_size]
        """
        if train:
            self.actor.train()
        else:
            self.actor.eval()
        transform_obs = observations.copy()
        transform_obs["cam_high"] = observations["cam_high"][:, 0, ...]
        transform_obs["cam_left_wrist"] = observations["cam_left_wrist"][:, 0, ...]
        transform_obs["cam_right_wrist"] = observations["cam_right_wrist"][:, 0, ...]
        transform_obs["state"] = observations["state"][:, 0, ...]
        # observations["task"] = ["put the objects in the box"] * observations["state"].shape[0]
        
        # Prepare inputs using PI0's preprocessing methods
        images, img_masks = self.actor.prepare_images(transform_obs)
        state = self.actor.prepare_state(transform_obs)
        lang_tokens, lang_masks = self.actor.prepare_language(transform_obs)

        batch_size = state.shape[0]
        log_probs = torch.zeros(batch_size, device=state.device)
        actions = self.actor.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None, deterministic=deterministic
        )
        actions = actions[..., :self.action_dim]

        return actions, log_probs

    @torch.no_grad()
    def forward_ref_policy(
        self,
        observations: Data,
        deterministic: bool = False,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for PI0 policy network.
        
        Args:
            observations: Input observations in PI0 format
            deterministic: Whether to use deterministic sampling
        Returns:
            actions: [batch_size, action_horizon, action_dim]
            log_probs: [batch_size]
        """
        self.ref_actor.eval()
        transform_obs = observations.copy()
        transform_obs["cam_high"] = observations["cam_high"][:, 0, ...]
        transform_obs["cam_left_wrist"] = observations["cam_left_wrist"][:, 0, ...]
        transform_obs["cam_right_wrist"] = observations["cam_right_wrist"][:, 0, ...]
        transform_obs["state"] = observations["state"][:, 0, ...]
        # observations["task"] = ["put the objects in the box"] * observations["state"].shape[0]
        
        # Prepare inputs using PI0's preprocessing methods
        images, img_masks = self.ref_actor.prepare_images(transform_obs)
        state = self.ref_actor.prepare_state(transform_obs)
        lang_tokens, lang_masks = self.ref_actor.prepare_language(transform_obs)

        batch_size = state.shape[0]
        log_probs = torch.zeros(batch_size, device=state.device)
        actions = self.ref_actor.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None, deterministic=True
        )
        actions = actions[..., :self.action_dim]

        return actions, log_probs

    def forward_temperature(self) -> torch.Tensor:
        """Forward pass for temperature Lagrange multiplier."""
        return self.temperature()

    def _compute_next_actions(self, batch):
        """Shared computation between loss functions"""
        # Sample actions from PI0 policy
        next_actions, next_actions_log_probs = self.forward_policy(
            batch["next_observations"], deterministic=False, train=False
        )
        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """
        Critic loss function combining SAC Bellman loss with Cal-QL regularization.
        """
        with torch.no_grad():
            next_actions, next_actions_log_probs = self._compute_next_actions(batch)
            # Evaluate next Qs for all ensemble members using target critic
            target_next_qs = self.forward_target_critic(
                batch["next_observations"],
                next_actions,
            )  # [ensemble_size, batch_size]

            # Subsample if requested
            if self.config["critic_subsample_size"] is not None:
                indices = torch.randint(0, self.config["critic_ensemble_size"], 
                                    (self.config["critic_subsample_size"],), device=target_next_qs.device)
                target_next_qs = target_next_qs[indices]

            target_next_min_q = target_next_qs.min(dim=0)[0]  # [batch_size]

            target_q = (
                batch["rewards"]
                + self.config["discount"] * batch["masks"] * target_next_min_q
            )

            if self.config["backup_entropy"]:
                temperature = self.forward_temperature()
                target_q = target_q - temperature * next_actions_log_probs

        predicted_qs_1 = self.forward_critic(batch["observations"], batch["actions"], train=True)
        target_qs = target_q.unsqueeze(0).repeat(self.config["critic_ensemble_size"], 1)    # [ensemble_size, batch_size]
        critic_loss_1 = F.mse_loss(predicted_qs_1, target_qs.detach())  # [ensemble_size, batch_size] -> [1]

        # Initialize CQL-related variables for logging
        critic_loss_2 = torch.tensor(0.0, device=predicted_qs_1.device)
        cql_qf_ood = torch.zeros_like(predicted_qs_1)
        cql_q_random = torch.zeros_like(predicted_qs_1)
        cql_q_current = torch.zeros_like(predicted_qs_1)
        cql_q_next = torch.zeros_like(predicted_qs_1)

        critic_info = {}

        critic_loss = critic_loss_1

        # Calibrated Conservative Q-Learning (Cal-QL) regularization
        # Only compute CQL loss when cql_weight > 0
        if self.config["cql_weight"] > 0:
            # Ref: Cal-QL implementation from https://github.com/nakamotoo/Cal-QL
            # https://github.com/young-geng/JaxCQL/blob/master/JaxCQL/conservative_sac.py
            # https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/cql.py
            batch_size = batch["actions"].shape[0]
            cql_n_actions = self.config["cql_n_actions"]
            
            def repeat_observations(obs_dict, repeat_factor):
                """Repeat observations for CQL sampling."""
                repeated_obs = {}
                for key, value in obs_dict.items():
                    if isinstance(value, list):  # Handle task strings
                        repeated_obs[key] = value * repeat_factor
                    else:  # Handle tensors
                        # Repeat each batch element repeat_factor times: [B, ...] -> [B*N, ...]
                        repeated_obs[key] = value.repeat_interleave(repeat_factor, dim=0)
                return repeated_obs
            
            with torch.no_grad():
                # all actions is (B*N, T, A) with action chunking
                cql_random_actions = torch.rand(
                    batch_size * cql_n_actions, *batch["actions"].shape[1:], 
                    device=batch["actions"].device,
                    dtype=batch["actions"].dtype
                ) * 40.0 - 20.0
                cql_random_actions = cql_random_actions[..., :self.action_dim]
                repeated_current_obs = repeat_observations(batch["observations"], cql_n_actions)
                cql_current_actions, cql_current_log_probs = self.forward_policy(
                    repeated_current_obs, deterministic=False, train=False
                )
                repeated_next_obs = repeat_observations(batch["next_observations"], cql_n_actions)
                cql_next_actions, cql_next_log_probs = self.forward_policy(
                    repeated_next_obs, deterministic=False, train=False
                )   # [B*N, T, A], [B*N]

            cql_current_actions, cql_current_log_probs = (
                cql_current_actions.detach(),
                cql_current_log_probs.detach(),
            )
            cql_next_actions, cql_next_log_probs = (
                cql_next_actions.detach(),
                cql_next_log_probs.detach(),
            )

            all_cql_actions = torch.cat([
                cql_random_actions,
                cql_current_actions,
                cql_next_actions
            ], dim=0)  # [3*B*N, T, A]
            
            all_repeated_current_obs = repeat_observations(repeated_current_obs, 3)
            all_cql_q_values = self.forward_critic(all_repeated_current_obs, all_cql_actions, train=True)  # [ensemble_size, 3*B*N]

            total_samples = batch_size * cql_n_actions
            cql_q_random = all_cql_q_values[:, :total_samples]  # [ensemble_size, B*N]
            cql_q_current = all_cql_q_values[:, total_samples:2*total_samples]  # [ensemble_size, B*N]
            cql_q_next = all_cql_q_values[:, 2*total_samples:]  # [ensemble_size, B*N]

            cql_q_random = cql_q_random.reshape(self.config["critic_ensemble_size"], batch_size, cql_n_actions)
            cql_q_current = cql_q_current.reshape(self.config["critic_ensemble_size"], batch_size, cql_n_actions)
            cql_q_next = cql_q_next.reshape(self.config["critic_ensemble_size"], batch_size, cql_n_actions)
            
            if self.config["cql_importance_sample"]:
                random_density = torch.log(torch.tensor(
                    0.5 ** batch["actions"].shape[-1],
                    device=batch["actions"].device,
                    dtype=batch["actions"].dtype
                ))
                # [B*N]
                cql_next_log_probs_expand = cql_next_log_probs.reshape(batch_size, cql_n_actions).unsqueeze(0)  # [1, B, N]
                cql_current_log_probs_expand = cql_current_log_probs.reshape(batch_size, cql_n_actions).unsqueeze(0)  # [1, B, N]
                # Concatenate Q-values with importance sampling correction
                cql_cat_q = torch.cat([
                    cql_q_random - random_density,
                    cql_q_next - cql_next_log_probs_expand,
                    cql_q_current - cql_current_log_probs_expand,
                ], dim=2)
            else:   # True
                data_q_expanded = predicted_qs_1.unsqueeze(2)  # [ensemble_size, batch_size, 1]
                cql_cat_q = torch.cat([
                    cql_q_random, 
                    data_q_expanded, 
                    cql_q_next, 
                    cql_q_current
                ], dim=2)   # [ensemble_size, batch_size, 4]
            
            cql_temp = 1.0
            cql_qf_ood = torch.logsumexp(cql_cat_q / cql_temp, dim=2) * cql_temp
            
            # CQL loss: E[log sum exp(Q(s,a)) - Q(s,a_data)]
            critic_loss_2 = (cql_qf_ood - predicted_qs_1).mean()    # [ensemble_size, batch_size] -> [1]

            critic_info.update(
                {
                    "critic_loss_2": critic_loss_2.item(),  # CQL regularization loss
                    "predicted_qs_ood": cql_qf_ood.mean().item(),
                    "predicted_qs_random": cql_q_random.mean().item(),
                    "predicted_qs_current": cql_q_current.mean().item(),
                    "predicted_qs_next": cql_q_next.mean().item(),
                }
            )
            critic_loss += self.config["cql_weight"] * critic_loss_2

        critic_info.update({
            "critic_loss": critic_loss.item(),
            "critic_loss_1": critic_loss_1.item(),  # Bellman loss
            "predicted_qs_1": predicted_qs_1.mean().item(),
            # **{f"predicted_next_{key}": next_actions[0, :, self.axis_to_id[key]].mean().item() \
            #    for key in self.axis_to_id.keys()},
            "target_qs": target_qs.mean().item(),
            "rewards": batch["rewards"].mean().item(),
            "dones": 1.0 - batch["masks"].float().mean().item(),
        })

        return critic_loss, critic_info

    def policy_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """Policy loss function for PI0 fine-tuning with SAC"""
        batch_size = batch["rewards"].shape[0]
        with torch.no_grad():
            temperature = self.forward_temperature()

        # Get actions and log probs from PI0 policy
        actions, log_probs = self.forward_policy(
            batch["observations"], deterministic=False, train=True
        )

        # prevent critic gradients but keep actor gradients
        for param in self.critic.parameters():
            param.requires_grad_(False)
            
        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            train=False,
        )
        predicted_q = predicted_qs.mean(dim=0)
        
        # Re-enable gradients for critic parameters
        for param in self.critic.parameters():
            param.requires_grad_(True)

        # SAC actor objective: maximize Q - temperature * entropy
        actor_objective = predicted_q - temperature * log_probs
        q_loss = - actor_objective.mean()

        info = {}

        # Distillation loss: compare predicted actions with reference actor actions
        actor_loss = q_loss
        if self.ref_actor is not None:
            # Use reference actor actions as target for distillation
            with torch.no_grad():
                ref_actions, _ = self.forward_ref_policy(
                    batch["observations"], deterministic=True, train=False
                )
            target_actions = ref_actions[..., :self.action_dim]  # [B, T, action_dim]
            predicted_actions = actions[..., :self.action_dim]  # [B, T, action_dim]
            distill_loss = F.mse_loss(predicted_actions, target_actions)
            info.update({
                "distill_loss": distill_loss.item(),
                # "diff_mean": (predicted_actions - target_actions).pow(2).mean().item(),
                # "diff_std": (predicted_actions - target_actions).pow(2).std().item(),
            })
            actor_loss += q_loss + self.config["distill_weight"] * distill_loss

        info.update({
            "actor_loss": actor_loss.item(),
            "q_loss": q_loss.item(),
            **{f"predicted_{key}": actions[:, :, self.axis_to_id[key]].mean().item() \
                for key in self.axis_to_id.keys()}, 
            # if only use demo buffer
            # **{f"gt_{key}": batch["actions"][:, :, self.axis_to_id[key]].mean().item() \
            #     for key in self.axis_to_id.keys()},
            # **{f"diff_{key}": (actions[:, :, self.axis_to_id[key]] - batch["actions"][:, :, self.axis_to_id[key]]).pow(2).mean().item() \
            #     for key in self.axis_to_id.keys()},
            # "diff_mean": (actions[:, :, :self.action_dim] - batch["actions"][:, :, :self.action_dim]).pow(2).mean().item(),
            # "diff_std": (actions[:, :, :self.action_dim] - batch["actions"][:, :, :self.action_dim]).pow(2).std().item(),
            "temperature": temperature.item(),
            "entropy": -log_probs.mean().item(),
        })

        return actor_loss, info

    def temperature_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """Temperature loss function for entropy regularization"""
        next_actions, next_actions_log_probs = self._compute_next_actions(batch)

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature(
            lhs=entropy, 
            rhs=torch.tensor(self.config["target_entropy"], device=entropy.device)
        )
        
        return temperature_loss, {"temperature_loss": temperature_loss.item()}

    def sft_loss_fn(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """Supervised Fine-Tuning loss function for policy imitation"""
        self.actor.train()

        # (B, T, C, H, W) -> (B, C, H, W)
        transform_obs = batch["observations"].copy()
        transform_obs["cam_high"] = batch["observations"]["cam_high"][:, 0, ...]
        transform_obs["cam_left_wrist"] = batch["observations"]["cam_left_wrist"][:, 0, ...]
        transform_obs["cam_right_wrist"] = batch["observations"]["cam_right_wrist"][:, 0, ...]
        transform_obs["state"] = batch["observations"]["state"][:, 0, ...]

        images, img_masks = self.actor.prepare_images(transform_obs)
        state = self.actor.prepare_state(transform_obs)
        lang_tokens, lang_masks = self.actor.prepare_language(transform_obs)
        actions = self.actor.prepare_action(batch)
        # actions_is_pad = None   # (B, T)
        # if "complementary_info" in batch:
        #     actions_is_pad = batch["complementary_info"].get("action_is_pad") # [B, T], min: 0, max: 1, bfloat16

        losses = self.actor.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

        # if actions_is_pad is not None:
        #     in_episode_bound = ~(actions_is_pad.bool())
        #     losses = losses * in_episode_bound.unsqueeze(-1)

        # Remove unrelated action dimensions (e.g., right arm, tactile?)
        # losses = losses[:, :, :self.action_dim]
        losses = losses[:, :, :7]   # only left arm

        sft_loss = losses.mean()
        
        # Report the diff between gt and predicted action 
        with torch.no_grad():
            actions = self.actor.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=None, deterministic=False
            )
            actions = actions[:, :, :self.action_dim]

        info = {
            "sft_loss": sft_loss.item(),
            **{f"predicted_{key}": actions[:, :, self.axis_to_id[key]].mean().item() \
                for key in self.axis_to_id.keys()},
            **{f"gt_{key}": batch["actions"][:, :, self.axis_to_id[key]].mean().item() \
                for key in self.axis_to_id.keys()},
            **{f"diff_{key}": (actions[:, :, self.axis_to_id[key]] - batch["actions"][:, :, self.axis_to_id[key]]).pow(2).mean().item() \
                for key in self.axis_to_id.keys()},
            "diff_mean": (actions[:, :, :self.action_dim] - batch["actions"][:, :, :self.action_dim]).pow(2).mean().item(),
            "diff_std": (actions[:, :, :self.action_dim] - batch["actions"][:, :, :self.action_dim]).pow(2).std().item(),
        }
        
        return sft_loss, info

    def prepare_batches(self, batch: Batch) -> Batch:
        """
        Prepare batches for critic and actor by applying necessary transformations.
        Args:
            batch: Original batch of data
        Returns:
            Tuple of (critic_batch, actor_batch) where actor_batch has transformed observations
        """
        # Apply augmentation if configured
        if self.config.get("augmentation_function") is not None:
            batch = self.config["augmentation_function"](batch)

        # Add reward bias
        batch["rewards"] = batch["rewards"] + self.config["reward_bias"]

        # Transform observations to (B, T, C, H, W) format
        batch["observations"] = self._batch_transform_observations(batch["observations"])
        batch["next_observations"] = self._batch_transform_observations(batch["next_observations"])

        return batch

    def update_target_critic(self):
        """Update target critic using soft update."""
        tau = self.config["soft_target_update_rate"]
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def forward(self, batch: Batch, networks_to_update: list, training_mode: str = "rl") -> tuple:
        self.train()
        batch = self.prepare_batches(batch)
        
        info = {}
        total_loss = 0.0
        
        if training_mode == "rl":
            if "critic" in networks_to_update:
                critic_loss, critic_info = self.critic_loss_fn(batch)
                total_loss += critic_loss
                info["critic"] = critic_info

            if "actor" in networks_to_update:
                actor_loss, actor_info = self.policy_loss_fn(batch)
                total_loss += actor_loss
                info["actor"] = actor_info

            if "temperature" in networks_to_update:
                temp_loss, temp_info = self.temperature_loss_fn(batch)
                total_loss += temp_loss
                info["temperature"] = temp_info
                
        elif training_mode == "sft":
            if "actor" in networks_to_update:
                sft_loss, sft_info = self.sft_loss_fn(batch)
                total_loss += sft_loss
                info["actor"] = sft_info

        return total_loss, info
    
    def train(self):
        """Set training mode."""
        self.critic.train()
        self.target_critic.train()
        self.temperature.train()
        self.actor.train()
        
    def eval(self):
        """Set evaluation mode."""
        self.critic.eval()
        self.target_critic.eval()
        self.temperature.eval()
        self.actor.eval()

    def sample_actions(
        self,
        obs: Data,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample actions from the policy."""
        self.actor.eval()
        
        # Get device from model parameters
        device = next(self.actor.parameters()).device
        
        # (B, H, W, C) -> (B, C, H, W)
        batch = {
            "cam_high": torch.Tensor(obs['cam_high']).permute(0, 3, 1, 2).to(device),
            "cam_left_wrist": torch.Tensor(obs['cam_left_wrist']).permute(0, 3, 1, 2).to(device),
            "cam_right_wrist": torch.Tensor(obs['cam_right_wrist']).permute(0, 3, 1, 2).to(device),
            "state": torch.Tensor(obs['state']).to(device),
            'task': [obs['prompt']],
        }

        # convert to bfloat16
        batch["cam_high"] = batch["cam_high"].to(torch.bfloat16)
        batch["cam_left_wrist"] = batch["cam_left_wrist"].to(torch.bfloat16)
        batch["cam_right_wrist"] = batch["cam_right_wrist"].to(torch.bfloat16)
        batch["state"] = batch["state"].to(torch.bfloat16)

        # Prepare inputs using PI0's preprocessing methods
        images, img_masks = self.actor.prepare_images(batch)
        state = self.actor.prepare_state(batch)
        lang_tokens, lang_masks = self.actor.prepare_language(batch)

        with torch.no_grad():
            actions = self.actor.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=None, deterministic=deterministic
            )
        actions = actions[0, :, :self.action_dim]    # For dobot, we use :14
        return actions

    @classmethod
    def create_pixels(
        cls,
        observations: Data,
        actions: torch.Tensor,
        # PI0 configuration
        openpi_checkpoint_dir: str,
        # Model architecture for critics only
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        critic_keys: Iterable[str] = ("cam_high",),
        augmentation_function: Optional[callable] = None,
        # Optimizer configs
        optimizer_kwargs: dict = None,
        lr_scheduler_kwargs: dict = None,
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        reward_bias: float = 0.0,
        cql_weight: float = 1.0,
        distill_weight: float = 0.0,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Create a new pixel-based SAC agent with PI0 policy and small critic networks.
        """
        # Load PI0 policy
        pi0_config = PI0Config(
            # freeze_vision_encoder=False,
        )
        pi0_config.scheduler_warmup_steps = lr_scheduler_kwargs["num_warmup_steps"]
        pi0_config.scheduler_decay_steps = lr_scheduler_kwargs["num_decay_steps"]
        actor = PI0Policy.from_pretrained(
            openpi_checkpoint_dir, 
            config=pi0_config, 
            local_files_only=True,
        )
        # actor.config.num_steps = 1  # for debugging
        del actor.model.paligemma_with_expert.gemma_expert.model.embed_tokens
        del actor.model.paligemma_with_expert.gemma_expert.lm_head
        # actor.model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.norm.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].post_attention_layernorm.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.down_proj.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.up_proj.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].mlp.gate_proj.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].self_attn.o_proj.requires_grad_(False)
        actor.model.paligemma_with_expert.paligemma.language_model.model.layers[17].self_attn.q_proj.requires_grad_(False)
        gc.collect()

        for module in actor.model.modules():
            if isinstance(module, torch.nn.Dropout):
                cprint(f"Disabling dropout in module: {module}", "red")
                module.p = 0

        cprint(f"Successfully loaded PI0 policy from {openpi_checkpoint_dir}", "green")

        # Create encoders for critic
        if encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                resnetv1_configs,
            )
            # Create separate encoder instances for each image key
            encoders = {}
            for image_key in critic_keys:
                pretrained_encoder = resnetv1_configs["pretrained-resnet18"]()
                encoders[image_key] = pretrained_encoder
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        # Create encoder wrapper for critic
        state_dim = observations["state"].shape[-1] if use_proprio else 0
        encoder = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            state_dim=state_dim,
            enable_stacking=True,
            image_keys=critic_keys,
        )
        
        # Create critic network
        critic_backbone = ensemblize(
            partial(MLP, **critic_network_kwargs), 
            # partial(Transformer, **critic_network_kwargs), 
            critic_ensemble_size
        )()
        
        critic = Critic(
            encoder=encoder,
            network=critic_backbone,
            network_output_dim=critic_network_kwargs["hidden_dims"][-1],
        )

        # Create temperature Lagrange multiplier
        temperature = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
        )

        all_params = [
            {
                "params": filter(lambda p: p.requires_grad, actor.parameters()),
                **optimizer_kwargs["actor"],
            }, 
            {
                "params": filter(lambda p: p.requires_grad, critic.parameters()),
                **optimizer_kwargs["critic"],
            }, 
            {
                "params": filter(lambda p: p.requires_grad, temperature.parameters()),
                **optimizer_kwargs["temperature"],
            }
        ]
        unified_optimizer = optim.AdamW(all_params)
        # unified_optimizer = optim.Adam(all_params)

        lr_scheduler = CosineDecayWithWarmupSchedulerConfig( 
            **lr_scheduler_kwargs,
        ).build(unified_optimizer, num_training_steps=0)    # HACK: num_training_steps is not used
        # lr_scheduler = optim.lr_scheduler.CyclicLR(
        #     unified_optimizer,
        #     base_lr=lr_scheduler_kwargs["decay_lr"],
        #     max_lr=lr_scheduler_kwargs["peak_lr"],
        #     step_size_up=lr_scheduler_kwargs["num_warmup_steps"],
        #     step_size_down=lr_scheduler_kwargs["num_decay_steps"],
        # )
        # lr_scheduler = optim.lr_scheduler.ConstantLR(unified_optimizer, factor=1.0)

        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2  # e.g., -8 / 2 = -4

        config = {
            "critic_ensemble_size": critic_ensemble_size,
            "critic_subsample_size": critic_subsample_size,
            "discount": discount,
            "soft_target_update_rate": soft_target_update_rate,
            "target_entropy": target_entropy,
            "backup_entropy": backup_entropy,
            "image_keys": image_keys,
            "reward_bias": reward_bias,
            "augmentation_function": augmentation_function,
            "action_dim": actions.shape[-1],
            "cql_weight": cql_weight,
            "distill_weight": distill_weight,
            "cql_n_actions": 4,
            "cql_importance_sample": True,
            **kwargs,
        }

        agent = cls(
            config=config,
            actor=actor,
            critic=critic,
            temperature=temperature,
            device=device,
        )
        
        return agent, unified_optimizer, lr_scheduler

