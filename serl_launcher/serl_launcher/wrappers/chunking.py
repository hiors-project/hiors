from collections import deque
from typing import Optional

import gymnasium as gym
import gymnasium.spaces
import numpy as np


def stack_obs(obs):
    def _stack_tree(tree):
        """Recursively traverse tree structure and stack lists."""
        if isinstance(tree, list):
            return np.stack(tree)
        elif isinstance(tree, dict):
            return {k: _stack_tree(v) for k, v in tree.items()}
        else:
            return tree
    
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return _stack_tree(dict_list)


def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    # elif isinstance(space, gym.spaces.Text):
    #     return gym.spaces.Text(max_length=space.max_length)
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()


class ChunkingWrapper(gym.Wrapper):
    """
    Enables observation histories and receding horizon control.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        self.current_obs = deque(maxlen=self.obs_horizon)

        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon
            )

    def step(self, action, *args):
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon, f"Action of shape {action.shape} is less than act_exec_horizon {act_exec_horizon}"

        # for i in range(act_exec_horizon):
        #     obs, reward, done, trunc, info = self.env.step(action[i], *args)
        #     self.current_obs.append(obs)
        obs, reward, done, trunc, info = self.env.step(action[0], *args)
        self.current_obs.append(obs)

        next_obs = stack_obs(self.current_obs)

        return (next_obs, reward, done, trunc, info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs, obs_horizon=1):
    if obs_horizon != 1:
        # TODO: Support proper stacking
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    obs = {k: v[None] for k, v in obs.items()}
    return obs