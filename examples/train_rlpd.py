#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig, OmegaConf
import glob
import time
from functools import partial
import torch
import torch.distributed as dist
import numpy as np
import random
import tqdm
import os
import copy
import shutil
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
# from natsort import natsorted
from termcolor import cprint
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, DeepSpeedPlugin, HfDeepSpeedConfig
# from accelerate import DistributedDataParallelKwargs
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import compute_param_norm, compute_grad_norm
from serl_launcher.utils.logging_utils import print_rich_single_line_metrics, print_dict_mean, register_features_types
from serl_launcher.utils.launcher import (
    make_sac_pi0_agent,
    make_wandb_logger,
    make_tensorboard_logger,
)
from serl_launcher.data.dataset import ReplayBufferDataset, collate_fn, make_dataset, to_dtype
from serl_launcher.data.replay_buffer import ReplayBuffer, OfflineReplayBuffer, concatenate_batch_transitions
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.networks.pi0.modeling_pi0 import PI0Config
from agentlace.trainer import TrainerServer, TrainerClient, TrainerConfig
from agentlace.data.data_store import QueuedDataStore

# Device setup (will be handled by Accelerate now)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )

def create_dummy_batch(
        batch_size: int, 
        image_shape: tuple, 
        state_shape: tuple, 
        action_shape: tuple,
        device, dtype,
        obs_horizon: int = 1,
    ):
    """
    Create a dummy global batch structure for non-main processes.
    This will be overwritten by the scatter operation.
    """
    B = batch_size  # Full global batch size
    T = obs_horizon
    return {
        "observations": {
            "cam_high": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "cam_left_wrist": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "cam_right_wrist": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "state": torch.zeros((B, T, *state_shape), device=device, dtype=dtype),
        },
        "actions": torch.zeros((B, *action_shape), device=device, dtype=dtype),
        "next_observations": {
            "cam_high": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "cam_left_wrist": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "cam_right_wrist": torch.zeros((B, T, *image_shape), device=device, dtype=dtype),
            "state": torch.zeros((B, T, *state_shape), device=device, dtype=dtype),
        },
        "masks": torch.zeros((B), device=device, dtype=dtype),
        "dones": torch.zeros((B), device=device, dtype=dtype),
        "rewards": torch.zeros((B), device=device, dtype=dtype),
        # "complementary_info": {
        #     "action_is_pad": torch.zeros((B, 50), device=device, dtype=dtype),
        # }
    }

def scatter_batch(global_batch, accelerator, global_batch_size,
                  image_shape, state_shape, action_shape):
    """
    Scatter global batch from main process to all processes.
    
    Args:
        global_batch: Full batch from main process (None on non-main processes)
        accelerator: Accelerator instance
        global_batch_size: Size of the full global batch
        image_shape: Shape of image observations (C, H, W)
        state_shape: Shape of state observations
        action_shape: Shape of actions
    
    Returns:
        local_batch: Local batch for current process
    """
    device = accelerator.device
    world_size = accelerator.num_processes
    
    # Determine weight dtype
    weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    
    if world_size == 1:
        # Single process case
        return global_batch
    
    # Calculate local batch size for each process
    local_batch_size = global_batch_size // world_size
    remainder = global_batch_size % world_size
    current_local_size = local_batch_size + (1 if accelerator.process_index < remainder else 0)

    local_batch = create_dummy_batch(
        batch_size=current_local_size, 
        image_shape=image_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device, 
        dtype=weight_dtype
    )
    
    def scatter_tensor(global_tensor, local_tensor):
        """Scatter a single tensor from global to local batches"""
        if accelerator.is_main_process:
            # Split global tensor into chunks for each process
            scatter_list = []
            start_idx = 0
            for rank in range(world_size):
                rank_local_size = local_batch_size + (1 if rank < remainder else 0)
                end_idx = start_idx + rank_local_size
                # Extract chunk for this rank
                chunk = global_tensor[start_idx:end_idx].contiguous()
                scatter_list.append(chunk)
                start_idx = end_idx
        else:
            scatter_list = None
        dist.scatter(local_tensor, scatter_list, src=0)
        return local_tensor
    
    # Scatter observation tensors
    if accelerator.is_main_process and global_batch is not None:
        global_obs = global_batch["observations"]
        global_next_obs = global_batch["next_observations"]
    else:
        global_obs = None
        global_next_obs = None
    obs_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist", "state"]
    for key in obs_keys:
        if global_obs is not None:
            scatter_tensor(global_obs[key], local_batch["observations"][key])
            scatter_tensor(global_next_obs[key], local_batch["next_observations"][key])
        else:   # other processes still need to scatter to participate in the collective operation
            scatter_tensor(None, local_batch["observations"][key])
            scatter_tensor(None, local_batch["next_observations"][key])
    
    # Scatter other tensors
    simple_keys = ["actions", "masks", "dones", "rewards"]
    for key in simple_keys:
        if accelerator.is_main_process and global_batch is not None:
            global_tensor = global_batch[key]
        else:
            global_tensor = None
        scatter_tensor(global_tensor, local_batch[key])
    
    return local_batch

def initialize_replay_buffer(
    cfg,
) -> ReplayBufferDataStore:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.
    """
    # if not cfg.resume:
    return ReplayBufferDataStore(
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

def initialize_demo_buffer(
    cfg,
) -> ReplayBuffer:
    """
    Initialize an offline demo replay buffer from a dataset.
    """
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
        action_horizon=cfg.environment.action_horizon,
        action_dim=cfg.environment.action_dim,
        image_size=cfg.environment.image_size,
    )
    return offline_replay_buffer


##############################################################################

def actor(agent, data_store, successful_data_store, env, accelerator, cfg):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    torch.set_float32_matmul_precision('high')    # Ref: serve_policy.py
    
    start_step = 0
    device = accelerator.device
    agent = agent.to(device).to(torch.bfloat16)

    if cfg.eval_checkpoint_step:
        cprint("evaluating checkpoint", "yellow")
        success_counter = 0
        valid_episodes = 0  # count only episodes longer than 3 steps
        time_list = []

        while valid_episodes < cfg.eval_n_trajs:
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            episode_steps = 0
            
            while not done:
                # inference_start_time = time.time()
                
                actions = agent.sample_actions(obs, deterministic=True)
                actions = actions.float().cpu().numpy()

                # accelerator.print(f"[actor] inference time: {time.time() - inference_start_time}")

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                episode_steps += 1

                if done:
                    success = reward > cfg.dataset.success_threshold
                    # Only count episodes longer than 3 steps, like in actor logic
                    if episode_steps > cfg.environment.ignore_step_threshold:
                        valid_episodes += 1
                        if success:
                            success_counter += 1
                        dt = time.time() - start_time
                        time_list.append(dt)
                        accelerator.print(f"[eval] episode time: {dt:.2f} (s)")
                        accelerator.print(f"[eval] reward: {reward:.2f}")
                        accelerator.print(f"[eval] success rate: {success_counter}/{valid_episodes}")
                    else:
                        accelerator.print(f"[eval] Episode too short ({episode_steps} steps), not counting in success rate")

        accelerator.print(f"success rate: {success_counter / valid_episodes} ({success_counter}/{valid_episodes})")
        accelerator.print(f"average time: {np.mean(time_list)}")
        return

    datastore_dict = {
        "actor_env": data_store,
        # "actor_env_intvn": intvn_data_store,
        "actor_env_successful": successful_data_store,
    }

    cprint(f"Starting actor loop with {cfg.ip} at port 5488", "yellow")
    client = TrainerClient(
        "actor_env",
        cfg.ip,
        make_trainer_config(
            port_number=5488,
            broadcast_port=5489,
        ),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        with torch.no_grad():
            for name, param in agent.named_parameters():
                if name in params:
                    assert isinstance(params[name], torch.Tensor), f"params[name] is not a tensor: {type(params[name])}"
                    param.copy_(params[name].to(device))
        cprint(f"[Actor] Updated network", "green")

    client.recv_network_callback(update_params)

    episode_transitions = []

    cprint("env reset...", "green")
    obs, _ = env.reset()
    done = False
    cprint("env reset done", "green")

    # Build computation graph (even for pytorch)
    cprint("Building graph...", "yellow")
    graph_build_start_time = time.time()
    _ = agent.sample_actions(obs, deterministic=False)
    cprint(f"Graph built in {time.time() - graph_build_start_time} seconds", "green")

    # Training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_count = 0
    axis_to_id = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5, "rw": 6, "gripper": 7}
    predicted_actions = {k: [] for k in axis_to_id.keys()}

    pbar = tqdm.tqdm(range(start_step, cfg.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < cfg.random_steps:
                actions = env.action_space.sample()
            else:
                actions = agent.sample_actions(obs, deterministic=False)
                actions = actions.float().cpu().numpy()

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                truncateds=truncated,
                complementary_info={
                    "state_chunk": info["state_chunk"], # [max_state_chunk_len, state_dim]
                    "intervention_chunk": info["intervention_chunk"],
                    "state_timestamp_chunk": info["state_timestamp_chunk"],
                },
            )
            
            for key in predicted_actions.keys():
                predicted_actions[key].append(actions[:, axis_to_id[key]].mean())

            # data_store.insert(transition)
            episode_transitions.append(transition)
        
            # if reward > cfg.dataset.success_threshold:  # per-step success, for dense reward
            #     successful_data_store.insert(transition)

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                # if return > reward_threshold, then episode is successful
                # episode_success = info["episode"]["r"][0] > cfg.dataset.success_threshold # episodic, no discount
                episode_success = reward > cfg.dataset.success_threshold # single-step
                info["episode"]["succeed"] = float(episode_success)
                info["episode"]["actor_param_norm"] = compute_param_norm(agent.actor)
                info["episode"]["critic_param_norm"] = compute_param_norm(agent.critic)
                for key in predicted_actions.keys():
                    info["episode"][f"predicted_{key}"] = np.mean(predicted_actions[key])
                
                # per-episode success, for sparse reward
                # NOTE: we filter out too short episodes to avoid repetitive success by reward model
                if len(episode_transitions) > cfg.environment.ignore_step_threshold:
                    for ep_t in range(len(episode_transitions)):
                        if ep_t == len(episode_transitions) - 1:
                            # always insert the last transition
                            data_store.insert(episode_transitions[ep_t].copy())
                            if episode_success:
                                successful_data_store.insert(episode_transitions[ep_t].copy())
                        else:
                            # NOTE: we also filter out no-ops transitions at the beginning of episode
                            # check the 'state_chunk''s std is larger than a small threshold
                            curr_transition = episode_transitions[ep_t]
                            # get non-zero part of state_chunk
                            state_chunk = curr_transition['complementary_info']['state_chunk'][:, :7]  # left arm only
                            non_zero_mask = (state_chunk != 0).any(axis=-1)
                            criterion = state_chunk[non_zero_mask].std(axis=0).max()
                            if criterion > 1e-3:
                                data_store.insert(curr_transition.copy())
                                if episode_success:
                                    successful_data_store.insert(curr_transition.copy())
                            else:
                                cprint(f"Filter out no-op transition at step {ep_t} of ep {episode_count}", "red")
                else:
                    cprint(f"Episode too short, not inserting into buffer", "red") 
                
                stats = {"environment": info}
                client.request("send-stats", stats)
                # print_rich_single_line_metrics(stats)
                pbar.set_description(f"ep: {episode_count}, last return: {running_return}") # no discount
                
                running_return = 0.0
                already_intervened = False
                intervention_count = 0
                intervention_steps = 0
                episode_count += 1
                predicted_actions = {k: [] for k in axis_to_id.keys()}
                episode_transitions = []
                client.update()
                obs, _ = env.reset()

        timer.tock("total")

        if step % cfg.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

##############################################################################

def learner(agent, unified_optimizer, replay_buffer, demo_buffer, successful_buffer, lr_scheduler, cfg, accelerator):
    """
    The learner loop using Accelerate for distributed training with buffer broadcasting.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup logging
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
        elif cfg.logging.logger == "wandb":
            logger = make_wandb_logger(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                description=cfg.exp_name,
                unique_identifier=cfg.unique_identifier,
                debug=False,
                offline=False,
                variant=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            )
        else:
            raise ValueError(f"Unknown logger: {cfg.logging.logger}, choose from [tensorboard, debug, offline, wandb]")

    # Load state checkpoint if exists
    start_step = 0
    if cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        checkpoint_dirs = [d for d in os.listdir(cfg.checkpoint_path) 
                          if os.path.isdir(os.path.join(cfg.checkpoint_path, d)) and d.startswith("state_")]
        if checkpoint_dirs:
            # Find latest checkpoint
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('_')[-1]))
            checkpoint_path = os.path.join(cfg.checkpoint_path, latest_checkpoint)
            
            accelerator.print(f"Loading state checkpoint from {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            start_step = int(latest_checkpoint.split('_')[-1])
            accelerator.print(f"Resumed from step {start_step}")
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if logger is not None and accelerator.is_main_process:
            logger.log(payload, step=step)
        return {}
    
    # Create server (only on main process for actor communication)
    server = None
    if accelerator.is_main_process:
        server = TrainerServer(
            make_trainer_config(port_number=5488, broadcast_port=5489), 
            request_callback=stats_callback
        )
        if replay_buffer is not None:
            server.register_data_store("actor_env", replay_buffer)
        if successful_buffer is not None:
            server.register_data_store("actor_env_successful", successful_buffer)
        server.start(threaded=True)
        accelerator.print("TrainerServer started for actor communication")

    # Wait for replay buffer to fill (only main process monitors)
    if accelerator.is_main_process and replay_buffer is not None:
        pbar = tqdm.tqdm(
            total=cfg.training_starts,
            initial=len(replay_buffer),
            desc="Filling up replay buffer",
            position=0,
            leave=True,
        )
        while len(replay_buffer) < cfg.training_starts:
            pbar.update(len(replay_buffer) - pbar.n)
            time.sleep(1)
        pbar.update(len(replay_buffer) - pbar.n)
        pbar.close()
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    # Wait for successful buffer to fill (only main process monitors)
    if accelerator.is_main_process and successful_buffer is not None:
        pbar = tqdm.tqdm(
            total=cfg.successful_training_starts,
            initial=len(successful_buffer),
            desc="Filling up successful buffer",
            position=0,
            leave=True,
        )
        while len(successful_buffer) < cfg.successful_training_starts:
            pbar.update(len(successful_buffer) - pbar.n)
            time.sleep(1)
        pbar.update(len(successful_buffer) - pbar.n)
        pbar.close()
    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    # Create PyTorch Datasets and DataLoaders for replay buffers
    replay_dataset = ReplayBufferDataset(
        replay_buffer, 
        sample_args={
            "batch_size": cfg.batch_size // 2,
        }, 
    )
    replay_dataloader = None
    if accelerator.is_main_process:
        replay_dataloader = torch.utils.data.DataLoader(
            replay_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=0,
        )
    
    demo_dataloader = None
    if demo_buffer is not None:
        demo_dataset = demo_buffer
        demo_dataloader = torch.utils.data.DataLoader(
            demo_dataset,
            batch_size=cfg.batch_size // 2 // accelerator.num_processes,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    successful_dataset = ReplayBufferDataset(
        successful_buffer, 
        sample_args={
            "batch_size": cfg.batch_size // 2,
        }, 
    )
    successful_dataloader = None
    if accelerator.is_main_process:
        successful_dataloader = torch.utils.data.DataLoader(
            successful_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Prepare with accelerator
    prepare_list = [agent, unified_optimizer, lr_scheduler]
    if cfg.rl_use_online_data:
        prepare_list.append(replay_dataloader)
    if cfg.sft_use_online_data:
        prepare_list.append(successful_dataloader)
    if cfg.rl_use_offline_data or cfg.sft_use_offline_data:
        prepare_list.append(demo_dataloader)
    
    prepared = accelerator.prepare(*prepare_list)
    agent = prepared[0]
    unified_optimizer = prepared[1]
    lr_scheduler = prepared[2]
    dataloader_index = 3
    if cfg.rl_use_online_data:
        replay_dataloader = prepared[dataloader_index]
        dataloader_index += 1
    if cfg.sft_use_online_data:
        successful_dataloader = prepared[dataloader_index]
        dataloader_index += 1
    if cfg.rl_use_offline_data or cfg.sft_use_offline_data:
        demo_dataloader = prepared[dataloader_index]

    # Parameter publishing function
    def publish_network_params():
        """Publish network parameters to actors"""
        if accelerator.is_main_process and server:
            # Get unwrapped model for parameter extraction
            unwrapped_agent = accelerator.unwrap_model(agent)
            params_dict = {}
            for name, param in unwrapped_agent.named_parameters():
                # if param.requires_grad:
                params_dict[name] = param.detach().cpu()
            
            server.publish_network(params_dict)
            accelerator.print("Published network parameters")
    
    # Send initial network to actor
    publish_network_params()

    train_critic_networks_to_update = ["critic"]
    train_networks_to_update = ["critic", "actor", "temperature"]

    # Training loop
    timer = Timer()

    demo_iterator = None
    if demo_dataloader is not None:
        demo_iterator = iter(demo_dataloader)
    successful_iterator = None
    if successful_dataloader is not None:
        successful_iterator = iter(successful_dataloader)
    replay_iterator = None
    if replay_dataloader is not None:
        replay_iterator = iter(replay_dataloader)
    
    moving_loss = []
    last_rl_info = {}
    last_sft_info = {}
    cycle_length = cfg.rl_steps + cfg.sft_steps
    assert cycle_length > 0, "rl_steps + sft_steps must be greater than 0"
    
    for step in tqdm.tqdm(
        range(start_step, cfg.max_steps), 
        dynamic_ncols=True, 
        desc=f"learner_rank_{accelerator.process_index}",
        disable=not accelerator.is_main_process
    ):
        if step < cfg.sft_training_starts:
            training_mode = "rl"
        else:
            # Determine training mode: RL or SFT based on rl_steps and sft_steps
            # Example: rl_steps=4, sft_steps=1 -> RL for steps 0,1,2,3, SFT for step 4, RL for steps 5,6,7,8, SFT for step 9, etc.
            step_in_cycle = step % cycle_length
            training_mode = "rl" if step_in_cycle < cfg.rl_steps else "sft"
        
        if training_mode == "rl":
            with accelerator.accumulate(agent):
                # Standard RL training logic
                if step < cfg.actor_training_starts:
                    networks_to_update = train_critic_networks_to_update
                else:
                    # run n-1 critic updates and 1 critic + actor update.
                    if step % cfg.cta_ratio == 0:
                        networks_to_update = train_networks_to_update
                    else:
                        networks_to_update = train_critic_networks_to_update
                        
                with timer.context("sample_replay_buffer"):
                    local_batches = []
                    
                    # Sample offline demo batch if using offline data for RL
                    if cfg.rl_use_offline_data:
                        try:
                            local_demo_batch = next(demo_iterator)
                        except StopIteration:
                            demo_iterator = iter(demo_dataloader)
                            local_demo_batch = next(demo_iterator)
                        local_batches.append(local_demo_batch)

                    # Sample online replay batch if using online data for RL
                    if cfg.rl_use_online_data:
                        if accelerator.is_main_process:
                            try:
                                replay_batch = next(replay_iterator)
                            except StopIteration:
                                replay_iterator = iter(replay_dataloader)
                                replay_batch = next(replay_iterator)
                        else:
                            replay_batch = None
                        local_replay_batch = scatter_batch(
                            replay_batch, accelerator, cfg.batch_size // 2,
                            image_shape=(cfg.environment.image_size[1], cfg.environment.image_size[0], 3),
                            state_shape=(cfg.environment.state_dim,),
                            action_shape=(cfg.environment.action_horizon, cfg.environment.action_dim,)
                        )
                        local_batches.append(local_replay_batch)
                    
                    if len(local_batches) == 1:
                        local_batch = local_batches[0]
                    else:
                        local_batch = concatenate_batch_transitions(*local_batches)
                    
                    local_batch = to_dtype(local_batch, dtype=weight_dtype)

                with timer.context("train"):
                    total_loss, update_info = agent.forward(local_batch, networks_to_update, training_mode)

                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)

                    unified_optimizer.step()
                    lr_scheduler.step()
                    unified_optimizer.zero_grad()
                    
                    # Update target critic network
                    unwrapped_agent = accelerator.unwrap_model(agent)
                    if "critic" in networks_to_update:
                        unwrapped_agent.update_target_critic()

                    # to avoid critic update overwriting actor update info
                    if step < cfg.actor_training_starts or "actor" in networks_to_update:   
                        last_rl_info = update_info

        else:  # SFT training mode
            with accelerator.accumulate(agent):
                networks_to_update = ["actor"]
                
                with timer.context("sample_replay_buffer"):
                    local_batches = []
                    
                    # Sample offline demo batch if using offline data for SFT
                    if cfg.sft_use_offline_data:
                        try:
                            local_demo_batch = next(demo_iterator)
                        except StopIteration:
                            demo_iterator = iter(demo_dataloader)
                            local_demo_batch = next(demo_iterator)
                        local_batches.append(local_demo_batch)

                    # Sample online successful batch if using online data for SFT
                    if cfg.sft_use_online_data:
                        if accelerator.is_main_process:
                            try:
                                successful_batch = next(successful_iterator)
                            except StopIteration:
                                successful_iterator = iter(successful_dataloader)
                                successful_batch = next(successful_iterator)
                        else:
                            successful_batch = None
                        local_successful_batch = scatter_batch(
                            successful_batch, accelerator, cfg.batch_size // 2,
                            image_shape=(cfg.environment.image_size[1], cfg.environment.image_size[0], 3),
                            state_shape=(cfg.environment.state_dim,),
                            action_shape=(cfg.environment.action_horizon, cfg.environment.action_dim,)
                        )
                        local_batches.append(local_successful_batch)
                    
                    if len(local_batches) == 1:
                        local_batch = local_batches[0]
                    else:
                        local_batch = concatenate_batch_transitions(*local_batches)

                    local_batch = to_dtype(local_batch, dtype=weight_dtype)

                with timer.context("train"):
                    total_loss, update_info = agent.forward(local_batch, networks_to_update, training_mode)

                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)

                    unified_optimizer.step()
                    lr_scheduler.step()
                    unified_optimizer.zero_grad()
                    
                    last_sft_info = update_info

        moving_loss.append(total_loss.item())

        # Logging (only on main process)
        if step % cfg.log_period == 0:
            log_data = {"critic": {}, "actor": {}, "temperature": {}, "training": {}}
        
            # Compute parameter norms using unwrapped agent
            unwrapped_agent = accelerator.unwrap_model(agent)
            critic_param_norm = compute_param_norm(unwrapped_agent.critic)
            critic_target_param_norm = compute_param_norm(unwrapped_agent.target_critic)
            actor_param_norm = compute_param_norm(unwrapped_agent.actor)

            if accelerator.is_main_process:
                log_data.update(last_rl_info)
                log_data.update(last_sft_info)
                log_data["training"]["is_sft_step"] = float(training_mode == "sft")
                log_data["training"]["success_trans"] = float(len(successful_buffer) if successful_buffer is not None else 0)
                log_data["training"]["total_trans"] = float(len(replay_buffer) if replay_buffer is not None else 0)
                log_data["training"]["grad_norm"] = grad_norm    # this is actually the total grad norm (unified optimizer)
                log_data["training"]["moving_loss"] = np.mean(moving_loss) # this is actually the total loss
                log_data["training"]["lr_actor"] = unified_optimizer.param_groups[0]["lr"]
                log_data["training"]["lr_critic"] = unified_optimizer.param_groups[1]["lr"]
                log_data["training"]["lr_temperature"] = unified_optimizer.param_groups[2]["lr"]
                log_data["critic"]["param_norm"] = critic_param_norm
                log_data["critic"]["target_param_norm"] = critic_target_param_norm
                log_data["actor"]["param_norm"] = actor_param_norm
                moving_loss = []
                logger.log(log_data, step=step)
                logger.log({"timer": timer.get_average_times()}, step=step)
                # print_rich_single_line_metrics(log_data)

        # Publish updated network
        if step > 0 and step % cfg.steps_per_update == 0:
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
            publish_network_params()

        # TODO: Save buffer periodically
        if step == 20 or (step > 0 and step % cfg.buffer_period == 0):
            if accelerator.is_main_process:
                # clear old buffer before saving new buffer
                if replay_buffer is not None and len(replay_buffer) > 0:
                    replay_buffer_dir = os.path.join(cfg.dataset.lerobot_dir, cfg.dataset.replay_buffer_name)
                    if os.path.exists(replay_buffer_dir):
                        shutil.rmtree(replay_buffer_dir)
                    # save the buffer
                    replay_buffer.to_lerobot_dataset(
                        cfg.dataset.replay_buffer_name, fps=10, root=replay_buffer_dir
                    )
                    accelerator.print(f"Saved replay buffer to {replay_buffer_dir}")
                
                if successful_buffer is not None and len(successful_buffer) > 0:
                    successful_buffer_dir = os.path.join(cfg.dataset.lerobot_dir, cfg.dataset.successful_buffer_name)
                    if os.path.exists(successful_buffer_dir):
                        shutil.rmtree(successful_buffer_dir)
                    successful_buffer.to_lerobot_dataset(
                        cfg.dataset.successful_buffer_name, fps=10, root=successful_buffer_dir
                    )
                    accelerator.print(f"Saved successful buffer to {successful_buffer_dir}")

            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()

        # Save checkpoint
        if step > 0 and step % cfg.checkpoint_period == 0:
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()

            # save the model for inference
            if accelerator.is_main_process:
                unwrapped_agent = accelerator.unwrap_model(agent)
                # require copy or clone to avoid shared memory of tensors. https://github.com/huggingface/safetensors/issues/202
                model_to_save = copy.deepcopy(unwrapped_agent.actor)
                register_features_types()
                model_to_save.save_pretrained(os.path.join(cfg.checkpoint_path, f"{step}", "model"))
                del model_to_save

            # also save the state for resuming training (TODO: rl is hard to resume? So we save the model only)
            # checkpoint_path = os.path.join(cfg.checkpoint_path, f"state_{step}")
            # accelerator.save_state(checkpoint_path)
            accelerator.print(f"Saved checkpoint to {cfg.checkpoint_path} at step {step}")

    accelerator.print("Training completed")
    accelerator.end_training()

##############################################################################

@hydra.main(version_base=None, config_path="../config/", config_name="base")
def main(cfg: DictConfig):
    # Set seed for reproducibility (disabled for better performance)
    # set_seed(cfg.seed)

    # Initialize Accelerate
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        # project_dir=cfg.checkpoint_path if cfg.checkpoint_path else "./checkpoints",
        total_limit=3  # Keep last n checkpoints (but seems not working)
    )

    cfg.batch_size = cfg.train_micro_batch_size_per_gpu * cfg.training.num_gpus
    
    # Enable DeepSpeed if config provided
    deepspeed_plugin = None
    if cfg.training.deepspeed_config:
        deepspeed_config = HfDeepSpeedConfig(cfg.training.deepspeed_config)
        # HACK: as we use multiple dataloaders, it is hard for deepspeed to know the batch size, so we set it manually
        deepspeed_config.config["train_micro_batch_size_per_gpu"] = cfg.train_micro_batch_size_per_gpu // 2
        deepspeed_config.config["train_batch_size"] = cfg.batch_size // 2
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=deepspeed_config,
        )
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],   # for zero1 (ddp)
        deepspeed_plugin=deepspeed_plugin,    # for zero2/zero3
        project_config=project_config,
        # cpu=False
    )

    # Setup environment
    cprint(f"Setting up environment: {cfg.exp_name}", "yellow")
    if cfg.exp_name == "dobot_pnp":
        from experiments.dobot_pnp.config import get_environment
        env = get_environment(
            fake_env=cfg.learner,
            env_config=cfg.environment,
            proprio_keys=cfg.proprio_keys,
        )
    else:
        raise NotImplementedError(f"Unknown experiment name: {cfg.exp_name}")

    env = RecordEpisodeStatistics(env)

    # Create agent
    if cfg.agent == "sac_pi0":
        cfg.lr_scheduler.num_warmup_steps *= accelerator.num_processes
        cfg.lr_scheduler.num_decay_steps *= accelerator.num_processes
        agent, unified_optimizer, lr_scheduler = make_sac_pi0_agent(
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=cfg.image_keys,
            critic_keys=cfg.critic_keys,
            critic_network_kwargs=cfg.critic_network,
            encoder_type=cfg.encoder_type,
            discount=cfg.discount,
            openpi_config_name=cfg.openpi.config_name,
            openpi_checkpoint_dir=cfg.openpi.checkpoint_dir,
            critic_ensemble_size=cfg.sac.critic_ensemble_size,
            cql_weight=cfg.sac.cql_weight,
            distill_weight=cfg.sac.distill_weight,
            optimizer_kwargs=cfg.optimizer,
            lr_scheduler_kwargs=cfg.lr_scheduler,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {cfg.setup_mode}")
    
    # Handle checkpoint loading/clearing
    if accelerator.num_processes > 1: 
        accelerator.wait_for_everyone()
    if accelerator.is_main_process and cfg.checkpoint_path is not None and os.path.exists(cfg.checkpoint_path):
        checkpoint_files = glob.glob(os.path.join(cfg.checkpoint_path, "*"))
        if checkpoint_files:
            # time.sleep(5)   # wait for all outputs to be printed
            cprint(f"Found existing checkpoint: {checkpoint_files}", "yellow")
            
            choice = input("Do you want to (r)esume from checkpoint or (c)lear checkpoint folder? [r/c]:").lower().strip()
            if choice in ['r', 'resume']:
                cprint(f"Resuming from checkpoint...", "green")
                # Checkpoint loading will be handled in learner function
            elif choice in ['c', 'clear']:
                if os.path.exists(cfg.checkpoint_path):
                    cprint(f"Clearing checkpoint folder: {cfg.checkpoint_path}", "red")
                    shutil.rmtree(cfg.checkpoint_path)
                    cprint("Checkpoint folder cleared. Starting fresh training.", "green")
                else:
                    cprint(f"Checkpoint folder does not exist: {cfg.checkpoint_path}", "red")
        else:
            cprint("No checkpoint found. Starting fresh training.", "green")

    if cfg.learner:
        replay_buffer = None
        if accelerator.is_main_process:
            replay_buffer = initialize_replay_buffer(cfg)
        
        demo_buffer = None
        if cfg.rl_use_offline_data or cfg.sft_use_offline_data:
            demo_buffer = initialize_demo_buffer(cfg)
            cprint(f"demo buffer size: {len(demo_buffer)}", "green")
        else:
            cprint("demo buffer not initialized", "yellow")
        
        successful_buffer = None
        if accelerator.is_main_process:
            successful_buffer = initialize_replay_buffer(cfg)

        # clear/load replay buffer cache if exists
        if accelerator.is_main_process:
            # Determine which buffer patterns to check based on configuration
            replay_buffer_dirs = []
            successful_buffer_dirs = []
            
            if cfg.rl_use_online_data:
                replay_buffer_pattern = os.path.join(cfg.dataset.lerobot_dir, cfg.dataset.replay_buffer_name)
                replay_buffer_dirs = glob.glob(replay_buffer_pattern)
            
            if cfg.sft_use_online_data:
                successful_buffer_pattern = os.path.join(cfg.dataset.lerobot_dir, cfg.dataset.successful_buffer_name)
                successful_buffer_dirs = glob.glob(successful_buffer_pattern)

            # Only prompt if there are any buffers to clear
            buffers_to_clear = replay_buffer_dirs + successful_buffer_dirs
            if buffers_to_clear:
                # time.sleep(5)   # wait for all outputs to be printed
                cprint(f"Found existing replay/successful buffer caches: {buffers_to_clear}", "yellow")
                choice = input("Do you want to (r)esume or (c)lear the replay buffer caches? [r/c]: ").lower().strip()
                if choice in ['c', 'clear']:
                    # Clear replay_buffer_* directories if using online data for RL
                    if cfg.rl_use_online_data:
                        for replay_buffer_dir in replay_buffer_dirs:
                            if os.path.exists(replay_buffer_dir):
                                cprint(f"Clearing replay buffer cache: {replay_buffer_dir}", "red")
                                shutil.rmtree(replay_buffer_dir)
                                cprint("Replay buffer cache cleared.", "green")
                    # Clear successful_buffer_* directories if using online data for SFT
                    if cfg.sft_use_online_data:
                        for successful_buffer_dir in successful_buffer_dirs:
                            if os.path.exists(successful_buffer_dir):
                                cprint(f"Clearing successful buffer cache: {successful_buffer_dir}", "red")
                                shutil.rmtree(successful_buffer_dir)
                                cprint("Successful buffer cache cleared.", "green")
                else:
                    recover_cfg_pi0 = PI0Config(
                        n_obs_steps=1,
                        chunk_size=1,   # we implement action chunking in ReplayBuffer, so n_action_steps is 1
                        n_action_steps=1,
                    )
                    if replay_buffer_dirs and cfg.rl_use_online_data:
                        cprint("Resuming from existing replay buffer caches.", "yellow")
                        replay_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
                        # replay_cfg.dataset.repo_id = [replay_buffer_dirs[0].split("/")[-1]]
                        replay_cfg.dataset.repo_id = replay_buffer_dirs[0].split("/")[-1]
                        replay_cfg.dataset.image_transforms = False
                        replay_dataset = make_dataset(recover_cfg_pi0, replay_cfg)
                        replay_buffer = ReplayBufferDataStore.from_lerobot_dataset(
                            lerobot_dataset=replay_dataset,
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
                        cprint(f"replay buffer size: {len(replay_buffer)}", "green")

                    if successful_buffer_dirs and cfg.sft_use_online_data:
                        cprint("Resuming from existing successful buffer caches.", "yellow")
                        successful_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
                        # successful_cfg.dataset.repo_id = [successful_buffer_dirs[0].split("/")[-1]]
                        successful_cfg.dataset.repo_id = successful_buffer_dirs[0].split("/")[-1]
                        successful_cfg.dataset.image_transforms = False
                        successful_dataset = make_dataset(recover_cfg_pi0, successful_cfg)
                        successful_buffer = ReplayBufferDataStore.from_lerobot_dataset(
                            lerobot_dataset=successful_dataset,
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
                        cprint(f"successful buffer size: {len(successful_buffer)}", "green")
        
        # Wait for all processes to complete buffer setup
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

        # Start learner loop
        cprint("starting learner loop", "green")
        learner(agent, unified_optimizer, replay_buffer, demo_buffer, successful_buffer, lr_scheduler, cfg, accelerator)

    elif cfg.actor:
        cprint("creating data stores", "yellow")
        data_store = QueuedDataStore(50000)
        successful_data_store = QueuedDataStore(50000)

        # Start actor loop
        cprint("starting actor loop", "green")
        actor(agent, data_store, successful_data_store, env, accelerator, cfg)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    main() 