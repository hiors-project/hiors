import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
from collections import OrderedDict


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space, the images, and optionally prompts.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            OrderedDict((key, self.env.observation_space["state"][key]) for key in self.proprio_keys)
        )
        # Build the new observation space
        new_obs_space = {
            "state": flatten_space(self.proprio_space),
            **self.env.observation_space["images"],
        }
        
        # Add prompt space if it exists in the original observation space
        if "prompt" in self.env.observation_space:
            new_obs_space["prompt"] = self.env.observation_space["prompt"]

        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, obs):
        new_obs = {
            "state": flatten(
                self.proprio_space,
                OrderedDict((key, obs["state"][key]) for key in self.proprio_keys),
            ),
            **obs["images"],
        }
        
        # Add prompt if it exists in the observation
        if "prompt" in obs:
            new_obs["prompt"] = obs["prompt"]
            
        return new_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
    new_obs = {
        "state": flatten(
            proprio_space,
            OrderedDict((key, obs["state"][key]) for key in proprio_keys),
        ),
        **obs["images"],
    }
    
    # Add prompt if it exists in the observation
    if "prompt" in obs:
        new_obs["prompt"] = obs["prompt"]
        
    return new_obs