import time
from gymnasium import Env, spaces
import gymnasium as gym
import signal
from termcolor import cprint
import threading

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Input timeout")
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            try:
                max_time = 5  # seconds
                # Set up timeout signal
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(max_time)  # 5 second timeout
                
                while True:
                    try:
                        rew = int(input("Success? (1/0): "))
                        assert rew == 0 or rew == 1
                        break
                    except ValueError:
                        print("Please enter 0 or 1")
                        continue
                    except AssertionError:
                        print("Please enter 0 or 1")
                        continue
                        
            except TimeoutError:
                cprint(f"\n Warning: {max_time}s timeout reached, setting reward to 0", "red")
                rew = 0
            finally:
                signal.alarm(0)  # Cancel the alarm
                
        # info["environment"]["succeed"] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()
    def run(self):
        while True:
            self.input_cbk(input()) #waits to get input + Return

class HumanResetWrapper(gym.Wrapper):
    """
    Wrapper that allows human to terminate episode by pressing specified keys.
    Uses threading-based keyboard input instead of keyboard package.
    's' for success (reward=1), 'f' for failure (reward=0).
    """
    def __init__(self, env, success_key='s', failure_key='f', wait_for_after_done_time=3, wait_for_after_reset_time=5):
        super().__init__(env)
        self.success_key = success_key
        self.failure_key = failure_key
        self.human_reset_requested = False
        self.human_reward = None
        self.last_input = None
        self.wait_for_after_done_time = wait_for_after_done_time  # seconds
        self.wait_for_after_reset_time = wait_for_after_reset_time  # seconds
        self.keyboard_thread = KeyboardThread(input_cbk=self._keyboard_callback)
        cprint(
            f"Keyboard monitoring enabled."
            f"Type '{self.success_key}' for success (reward=1) or '{self.failure_key}' for failure "
            f"(reward=0) and press Enter to reset episode.", "cyan"
        )
    
    def _keyboard_callback(self, inp):
        """Callback function for keyboard input"""
        self.last_input = inp.strip().lower()
        if self.last_input == self.success_key:
            if not self.human_reset_requested:  # Prevent multiple triggers
                self.human_reset_requested = True
                self.human_reward = 1.0
                cprint(f"Human reset requested via '{self.success_key}' key! (Success, reward=1)", "green")
        elif self.last_input == self.failure_key:
            if not self.human_reset_requested:  # Prevent multiple triggers
                self.human_reset_requested = True
                self.human_reward = 0.0
                cprint(f"Human reset requested via '{self.failure_key}' key! (Failure, reward=0)", "red")
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if self.human_reset_requested:
            done = True
            truncated = True
            rew = self.human_reward  # Use the reward set by keyboard input
            # info['human_reset'] = True
        if self.wait_for_after_done_time > 0 and (done or truncated):
            cprint(f"Episode ended. Please switch to 'server' mode in {self.wait_for_after_done_time} seconds...", "yellow")
            # Allow time for switch mode from "human" to "server"
            time.sleep(self.wait_for_after_done_time)
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        self.human_reset_requested = False
        self.human_reward = None
        self.last_input = None
        obs, info = self.env.reset(**kwargs)

        if self.wait_for_after_reset_time > 0:
            cprint(f"Reset finished. Please place the object back in {self.wait_for_after_reset_time} seconds...", "yellow")
            time.sleep(self.wait_for_after_reset_time)

        obs, info = self.env.reset(**kwargs)    # reset again to get the latest observation
        return obs, info


class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """Compute reward by directly using binary reward classifier results."""
    def __init__(self, env: Env, reward_classifier_func, target_hz = None, success_threshold=0.5):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz
        self.success_threshold = success_threshold

    def compute_reward(self, obs, action):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs, action)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew += self.compute_reward(obs, action)     # NOTE: add reward from classifier
        done = done or (rew > self.success_threshold)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
        if done:
            cprint(f"[reward] reward={rew}", "green" if rew == 1 else "white")
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class MultiCameraSubtaskRewardClassifierWrapper(gym.Wrapper):
    """Compute reward from the difference of last and current classifier predictions."""
    def __init__(self, env: Env, subtask_classifier_func, n_way: int, target_hz=None):
        super().__init__(env)
        self.subtask_classifier_func = subtask_classifier_func
        self.target_hz = target_hz
        self.n_way = n_way  # Number of subtasks (e.g., 2 for binary, 3 for ternary, etc.)
        self.last_subtask_label = None
        self.current_subtask_label = None

    def compute_reward(self, obs, action):
        if self.subtask_classifier_func is not None:
            # Get the current subtask prediction (returns probabilities for each subtask)
            self.current_subtask_label = self.subtask_classifier_func(obs, action)
            if self.last_subtask_label is None:
                self.last_subtask_label = self.current_subtask_label
            reward = self.current_subtask_label - self.last_subtask_label
            cprint(
                f"[reward] current={self.current_subtask_label}, last={self.last_subtask_label}, reward={reward}", 
                "green" if reward > 0 else "white"
            )
            self.last_subtask_label = self.current_subtask_label
            return reward
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        
        # Add reward from subtask classifier
        subtask_reward = self.compute_reward(obs, action)
        rew += subtask_reward

        done = done or (self.current_subtask_label >= self.n_way - 1)
        # done = done or (rew > 0)
        if done:
            cprint(
                f"[reward] done, reward={rew}, subtask_label={self.current_subtask_label}", 
                "green" if rew > 0 else "white"
            )
        
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
        
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.last_subtask_label = None
        self.current_subtask_label = None
        obs, info = self.env.reset(**kwargs)
        return obs, info

