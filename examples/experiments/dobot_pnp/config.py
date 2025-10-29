import os
import torch
import gymnasium as gym
import numpy as np
from functools import partial
from termcolor import cprint

from serl_launcher.wrappers.reward_wrapper import (
    HumanClassifierWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    MultiCameraSubtaskRewardClassifierWrapper,
    HumanResetWrapper,
)
from serl_launcher.utils.train_utils import count_params
from serl_launcher.networks.reward_classifier import (
    load_classifier_func, 
    outcome_reward_func,
    process_classifier_func,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.norm import NormalizeObsEnv
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.video_wrapper import VideoWrapper
from experiments.dobot_pnp.wrapper import PnpEnv


def naive_reward_func(obs, action):
    # set a goal pose, and give reward based on how close the current pose is to the goal pose
    goal_pose = np.array([
        # -1.26093048,  0.05079622, -0.86221378, -0.74359046, 1.52011571, 0.40968805,  0.98039216
        -1.36831033, -0.11180587, -1.05396258, -0.42605944, 1.51858346, 0.19339831, 0.97647059,
    ])  # up slightly

    # calculate the distance between the current pose and the goal pose
    current_pose = obs["state"][0] if isinstance(obs, dict) else obs
    distance = np.linalg.norm(current_pose[:7] - goal_pose)

    # Mean absolute error (MAE) between current pose and goal pose
    # distance = np.mean(np.abs(action[:, :7] - np.tile(goal_pose, (action.shape[0], 1))))

    dense_reward = 0.0
    sparse_reward = 0.0

    # dense reward
    max_distance = 0.5    # use state
    # max_distance = 0.3  # use action
    dense_reward = (max_distance - distance) / max_distance

    # sparse reward
    max_distance = 0.05
    sparse_reward = int(distance < max_distance)

    reward = dense_reward + sparse_reward

    print(f"[naive_reward_func] distance={distance:.3f}, reward={reward:.3f}")

    return reward

def get_environment(fake_env=False, env_config=None, proprio_keys=None):
    env = PnpEnv(
        fake_env=fake_env,
        config=env_config,
        hz=env_config.hz,
    )

    env = SERLObsWrapper(env, proprio_keys=proprio_keys)
    env = NormalizeObsEnv(env, image_keys=env_config.image_keys)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    if not fake_env:
        if env_config.reward_type in ["outcome", "process"]:
            classifier = load_classifier_func(
                image_keys=env_config.classifier_keys,
                checkpoint_path=env_config.classifier_path,
                use_proprio=env_config.classifier_use_proprio,
                state_dim=env_config.state_dim,
                n_way=env_config.classifier_n_way,
            )
            classifier = classifier.cuda()
            cprint(f"classifier parameters: {count_params(classifier)/1e6:.2f}M", "yellow")
        
        if env_config.reward_type == "naive":
            env = MultiCameraBinaryRewardClassifierWrapper(
                env, naive_reward_func, success_threshold=env_config.success_threshold
            )
        elif env_config.reward_type == "outcome":
            env = MultiCameraBinaryRewardClassifierWrapper(
                env, 
                partial(outcome_reward_func, classifier=classifier, env_config=env_config),
                success_threshold=env_config.success_threshold
            )
        elif env_config.reward_type == "process":
            env = MultiCameraSubtaskRewardClassifierWrapper(
                env, 
                partial(process_classifier_func, classifier=classifier, env_config=env_config), 
                n_way=env_config.classifier_n_way
            )
        elif env_config.reward_type == "human":
            # env = HumanClassifierWrapper(env)
            pass
        else:
            raise ValueError(f"Unknown reward type: {env_config.reward_type}")

        env = HumanResetWrapper(
            env,
            wait_for_after_done_time=env_config.wait_for_after_done_time,
            wait_for_after_reset_time=env_config.wait_for_after_reset_time
        )

        if env_config.save_video:
            env = VideoWrapper(
                env,
                image_keys=env_config.image_keys,
                video_dir='./videos',
                fps=5,
            )

    return env
