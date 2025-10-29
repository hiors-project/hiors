import gymnasium as gym
from gymnasium import Env, spaces
from typing import List
import numpy as np
import cv2

class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
        normalization_type: str = "normal",
    ):
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        if self.normalization_type == "normal":
            return (data * metadata["std"]) + metadata["mean"]
        elif self.normalization_type == "bounds":
            return (data * (metadata["max"] - metadata["min"])) + metadata["min"]
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def action(self, action):
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs):
        obs["proprio"] = self.unnormalize(
            obs["proprio"], self.action_proprio_metadata["proprio"]
        )
        return obs


class NormalizeObsEnv(gym.ObservationWrapper):
    """
    Normalizes the observation.
    """
    def __init__(self, env: Env, image_keys: List[str] = None):
        super().__init__(env)
        self.image_keys = image_keys if image_keys is not None else []
        # TODO: also revise observation space

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        obs = preprocess_observation(obs, self.image_keys)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = preprocess_observation(obs, self.image_keys)
        return obs, info

def preprocess_observation(observations: dict[str, np.ndarray], image_keys: List[str]) -> dict[str, np.ndarray]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as numpy arrays.
    """
    # map to expected inputs for the policy
    return_observations = observations.copy()

    imgs = {k: v for k, v in observations.items() if k in image_keys}

    for imgkey, img in imgs.items():
        # Convert to numpy array if not already
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
        # This is the case for human-in-the-loop RL where there is only one environment.
        # if img.ndim == 3:
        #     img = np.expand_dims(img, axis=0)
        # sanity check that images are channel last
        # _, h, w, c = img.shape
        h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {imgkey=}, {img.shape=}"

        # sanity check that images are uint8
        assert img.dtype == np.uint8, f"expect np.uint8, but instead got {imgkey=}, {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        # img = np.transpose(img, (0, 3, 1, 2))  # b h w c -> b c h w
        img = img.astype(np.float32)
        img /= 255.0

        return_observations[imgkey] = img

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = np.array(observations["state"], dtype=np.float32)
    # if agent_pos.ndim == 1:
    #     agent_pos = np.expand_dims(agent_pos, axis=0)
    return_observations["state"] = agent_pos

    return return_observations
