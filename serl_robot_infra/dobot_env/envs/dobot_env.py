"""Gym Interface for Dobot Robot Environment

This environment implements a websocket server that communicates with inference_xtrainer.py
running on the robot desktop. It receives real robot observations and sends actions back.
"""
import os
from datetime import datetime
import asyncio
import logging
import traceback
import threading
import queue
import time
import numpy as np
import gymnasium as gym
import cv2
import copy
from typing import Dict, Optional, Any
from termcolor import cprint
from scipy.spatial.transform import Rotation
import websockets.asyncio.server
import websockets.frames
from openpi_client import msgpack_numpy


MAX_TIMEOUT = 1e8   # for real robot, to avoid giving dummy observation
# MAX_TIMEOUT = 0.1 # use this if you don't have a real robot connected

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.name = name

    def run(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


class DobotEnv(gym.Env):
    def __init__(
        self,
        fake_env=False,
        hz=10,
        config=None,
    ):
        self.config = config
        self.hz = hz
        self.fake_env = fake_env
        self.display_image = config.display_image
        self.max_episode_length = config.max_episode_length
        self.reset_pose = config.reset_pose
        self.randomreset = config.random_reset
        self.random_xyz_range = config.random_xyz_range
        self.max_state_chunk_len = config.max_state_chunk_len
        # self.gripper_sleep = config.gripper_sleep
        self.gripper_threshold = config.gripper_threshold
        self.last_gripper_act = time.time()
        
        # Communication queues and async setup
        self.loop = None
        self.obs_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)  # Normal queue for actions
        self.server_metadata = {}
        
        # Episode state
        self.curr_path_length = 0
        self.current_obs = None
        self.current_info = None
        
        # Robot state tracking
        self.currpos = None
        self.reset_pose = np.array(list(config.reset_pose))  # Store the reset pose configuration
        
        # boundary box
        bounding_box_low = np.array(list(config.pose_limits.low))
        bounding_box_high = np.array(list(config.pose_limits.high))
        self.bounding_box = gym.spaces.Box(
            bounding_box_low,
            bounding_box_high,
            dtype=np.float32,
        )
        # Define action space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.action_horizon, config.action_dim),
            dtype=np.float32
        )
        self.state_dim = 14

        # Define observation space
        state_dict = {
            "end_effector/left_hand/observation": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper/left_hand/observation": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "end_effector/right_hand/observation": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "gripper/right_hand/observation": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            # "tactile/left_hand": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            # "tactile/right_hand": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        }
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict(state_dict),
            "images": gym.spaces.Dict({
                "cam_high": gym.spaces.Box(0, 255, shape=(config.image_size[1], config.image_size[0], 3), dtype=np.uint8),
                "cam_left_wrist": gym.spaces.Box(0, 255, shape=(config.image_size[1], config.image_size[0], 3), dtype=np.uint8),
                "cam_right_wrist": gym.spaces.Box(0, 255, shape=(config.image_size[1], config.image_size[0], 3), dtype=np.uint8),
            }),
            "prompt": gym.spaces.Text(max_length=256),
        })

        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "Dobot Robot Environment")
            self.displayer.start()

        # Only start websocket server if not using fake environment
        if not self.fake_env:
            # Start websocket server in background thread
            self.server_thread = threading.Thread(target=self._start_server, daemon=True)
            self.server_thread.start()    
            print(f"🚀 Starting websocket server on {config.host}:{config.port}...")
            time.sleep(3)  # Give server time to start
            print(f"Dobot Environment websocket server started on {config.host}:{config.port}")
        else:
            print("Dobot Environment initialized in fake mode (no robot connection)")



    def _start_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    async def _run_server(self):
        async with websockets.asyncio.server.serve(
            self._websocket_handler,
            self.config.host,
            self.config.port,
            compression=None,
            max_size=None,
        ) as server:
            print(f"Websocket server listening on {self.config.host}:{self.config.port}")
            await server.serve_forever()

    async def _websocket_handler(self, websocket):
        """Handle websocket connections from inference_xtrainer.py."""
        print(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send server metadata
        await websocket.send(packer.pack(self.server_metadata))

        try:
            while True:
                # Receive observation from robot
                start_msg_time = time.time()
                # print("Waiting to receive observation from robot...")
                msg = await websocket.recv()
                # print('websocket_policy_server.py recv time', time.time() - start_msg_time)

                start_obs_msg_time = time.time()
                obs_data = msgpack_numpy.unpackb(msg)
                # print('websocket_policy_server.py unpack time', time.time() - start_obs_msg_time)

                # Store observation for gym environment
                try:
                    self.obs_queue.put_nowait(obs_data)  # Use put_nowait to avoid blocking
                    # print("Observation successfully queued")
                except queue.Full:
                    print("Observation queue full, dropping old observation")
                    try:
                        self.obs_queue.get_nowait()  # Remove old observation
                        self.obs_queue.put_nowait(obs_data)  # Add new one
                    except queue.Empty:
                        pass
                
                # Wait for action from gym environment
                start_action_msg_time = time.time()
                try:
                    action = self.action_queue.get(timeout=MAX_TIMEOUT)
                except queue.Empty:
                    raise Exception("No action received")
                # print('websocket_policy_server.py inference time', time.time() - start_action_msg_time)
                
                # Validate action for NaN values before sending
                if np.isnan(action).any():
                    cprint(f"[env] ERROR: Action contains NaN values, replacing with zeros", "red")
                    action = np.tile(self.reset_pose, (self.config.action_horizon, 1))
                
                start_send_action_time = time.time()
                action_dict = {
                    "actions": action,
                }
                # Send action back to robot
                await websocket.send(packer.pack(action_dict))
                # print('websocket_policy_server.py send time', time.time() - start_send_action_time)

        except websockets.ConnectionClosed:
            print(f"Connection from {websocket.remote_address} closed")
        except Exception as e:
            cprint(f"Error in websocket handler: {e}", "red")
            traceback.print_exc()
            try:
                await websocket.send(packer.pack({"error": str(e)}))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
            except:
                pass  # Connection might already be closed

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation.
        
        Note: This method assumes it's called within the appropriate lock context
        when synchronization is needed.
        """
        # steps = max(1, int(timeout * self.hz))
        
        # Safe interpolation move
        # steps = 1
        # currpos_pad = np.concatenate([self.currpos, goal[8:]])
        # path = np.linspace(currpos_pad, goal, steps)

        # Direct move
        path = [goal]
        # Send each waypoint as an action
        for i, waypoint in enumerate(path):
            # Create action array: repeat the waypoint for all action horizon steps
            action = np.tile(waypoint, (self.config.action_horizon, 1))
            # Send action to robot
            if self.action_queue:
                # Clear old actions from queue
                while not self.action_queue.empty():
                    try:
                        self.action_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Put new action in queue
                try:
                    self.action_queue.put_nowait(action)
                except queue.Full:
                    # If queue is full, remove old action and put new one
                    try:
                        self.action_queue.get_nowait()
                        self.action_queue.put_nowait(action)
                    except queue.Empty:
                        pass
            
            time.sleep(1.0 / self.hz)

        # Update final position
        self.currpos = goal.copy()
        # cprint(f"Interpolation move completed to goal: {goal[:3]}", "green")

    def clip_safety_box(
            self, 
            pose: np.ndarray, 
            clip_indices: Optional[range] = range(0, 14), 
            tolerance: Optional[float] = 1e-2
        ) -> np.ndarray:
        """Clip the pose to be within the safety box (only xyz coordinates)."""
        original_pose = pose.copy() # [action_horizon, action_dim]
        pose_clipped = pose.copy()
        # Left arm
        pose_clipped[..., clip_indices] = np.clip(
            pose[..., clip_indices], self.bounding_box.low[clip_indices], self.bounding_box.high[clip_indices]
        )
        # Right arm
        # ...

        # Apply gripper control logic to each action step
        if self.gripper_threshold > 0:
            pose_clipped[..., 6] = np.where(pose_clipped[..., 6] > self.gripper_threshold, 1.0, 0.0)
            pose_clipped[..., 13] = np.where(pose_clipped[..., 13] > self.gripper_threshold, 1.0, 0.0)

        # Check if any values were clipped
        # was_clipped = np.abs(original_pose[..., clip_indices] - pose_clipped[..., clip_indices]).mean() >= tolerance
        was_clipped = np.abs(original_pose[..., 0:7] - pose_clipped[..., 0:7]).mean() >= tolerance
        # was_clipped = np.abs(original_pose[..., 1] - pose_clipped[..., 1]).mean() > tolerance        # only consider y-axis clipping
        if was_clipped:
            # Get mean values across action horizon for display
            original_mean = original_pose.mean(axis=0)[clip_indices]
            clipped_mean = pose_clipped.mean(axis=0)[clip_indices]
            message_parts = ["Action clipping: "]
            for i, idx in enumerate(clip_indices):
                original_val = original_mean[i]
                clipped_val = clipped_mean[i]
                if np.abs(original_val - clipped_val) > tolerance:
                    message_parts.append(f"\033[91m {original_val:.2f}->{clipped_val:.2f}\033[0m ")
                else:
                    message_parts.append(f" {original_val:.2f} ")
            print("".join(message_parts))

        return pose_clipped, was_clipped

    def step(self, action: np.ndarray):
        """Execute action and return next observation."""
        start_time = time.time()
        
        # Ensure action is the right shape and type
        # Fix: Check the correct dimension for multi-dimensional action space
        expected_shape = (self.config.action_horizon, self.config.action_dim)
        if action.shape != expected_shape:
            raise ValueError(f"Action must have shape {expected_shape}, got {action.shape}")
        
        was_clipped = False
        
        # Send action to robot via websocket
        if self.action_queue:
            # Clear old actions from queue
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    break
            
            # action_clipped, was_clipped = self.clip_safety_box(action)
            action_clipped = action
            # action_clipped = np.tile(self.reset_pose, (self.config.action_horizon, 1))

            # replace right arm with reset pose
            action_clipped[:, 7:13] = self.reset_pose[7:13]

            # Put action in queue
            try:
                self.action_queue.put_nowait(action_clipped)
            except queue.Full:
                # If queue is full, remove old action and put new one
                try:
                    self.action_queue.get_nowait()
                    self.action_queue.put_nowait(action_clipped)
                except queue.Empty:
                    pass
        
        # Wait for next observation from robot
        self._update_currpos()
        self._log_images(self.current_obs)

        self.curr_path_length += 1
        
        done = self.curr_path_length >= self.max_episode_length
        
        # Apply negative reward if action was clipped
        reward = 0.0
        # reward = -1.0 if was_clipped else 0.0  # Negative reward for clipping, otherwise 0.0
        
        # Control frequency
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        return self.current_obs, reward, done, False, self.current_info

    def reset(self, **kwargs):
        """Reset the environment."""
        cprint("Resetting Dobot environment...", "cyan")
        
        self.curr_path_length = 0

        # wait for the first observation to initialize currpos
        if self.currpos is None:
            cprint("Waiting for initial observation from robot...", "yellow")
            self._update_currpos()

        # Perform Cartesian reset
        if self.randomreset:
            reset_pose = self.reset_pose.copy()
            reset_pose[0:3] += np.random.uniform(
                -self.random_xyz_range, self.random_xyz_range, (3,)
            )  # Left arm xyz
            # reset_pose[8:11] += np.random.uniform(
            #     -self.random_xyz_range, self.random_xyz_range, (3,)
            # )  # Right arm xyz
            self.interpolate_move(reset_pose, timeout=0.8)
        else:
            reset_pose = self.reset_pose.copy()
            self.interpolate_move(reset_pose, timeout=0.8)
        
        # Clear any existing items in queues
        if self.action_queue:
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Wait for final observation after reset
        self._update_currpos()
        self._log_images(self.current_obs)

        self.current_info = {}  # inference_xtrainer.py has memory, so we need to clear it after reset
        
        return self.current_obs, self.current_info

    def _update_currpos(self):
        def _process_observation(raw_obs_data) -> dict:
            """Process observation data received from robot (inference_xtrainer.py) to match the observation space."""
            # [C, H, W] -> [H, W, C]
            obs_data = {}
            obs_data["images"] = {}

            # uint8, 0, 255
            obs_data["images"]["cam_high"] = np.transpose(raw_obs_data["images"]["cam_high"], (1, 2, 0)) \
                if raw_obs_data["images"]["cam_high"].shape[0] == 3 else raw_obs_data["images"]["cam_high"]
            obs_data["images"]["cam_left_wrist"] = np.transpose(raw_obs_data["images"]["cam_left_wrist"], (1, 2, 0)) \
                if raw_obs_data["images"]["cam_left_wrist"].shape[0] == 3 else raw_obs_data["images"]["cam_left_wrist"]
            obs_data["images"]["cam_right_wrist"] = np.transpose(raw_obs_data["images"]["cam_right_wrist"], (1, 2, 0)) \
                if raw_obs_data["images"]["cam_right_wrist"].shape[0] == 3 else raw_obs_data["images"]["cam_right_wrist"]
            
            for key in obs_data["images"]:
                obs_data["images"][key] = obs_data["images"][key].astype(np.uint8)

            obs_data["state"] = {
                "end_effector/left_hand/observation": raw_obs_data["state"][:6],
                "gripper/left_hand/observation": raw_obs_data["state"][6:7],
                "end_effector/right_hand/observation": raw_obs_data["state"][7:13],
                "gripper/right_hand/observation": raw_obs_data["state"][13:14],
            }
            if "prompt" in raw_obs_data:
                obs_data["prompt"] = raw_obs_data["prompt"]

            # Add intermediate states to info
            state_chunk = np.zeros((self.max_state_chunk_len, self.state_dim), dtype=np.float32)
            intervention_chunk = np.zeros((self.max_state_chunk_len,), dtype=bool)
            state_timestamp_chunk = np.zeros((self.max_state_chunk_len,), dtype=np.float64)
            for i, mid_state in enumerate(raw_obs_data["state_chunk"]):
                if i >= self.max_state_chunk_len:
                    cprint(f"Warning: received state_chunk exceeds max_len, truncating to {self.max_state_chunk_len}", "red")
                    break
                state_chunk[i] = mid_state["state"]
                intervention_chunk[i] = (mid_state["server_teleoperation_flag"] == "from server to teleoperation")
                state_timestamp_chunk[i] = mid_state["state_timestamp"]
            info_data = {
                "state_chunk": state_chunk,
                "intervention_chunk": intervention_chunk,
                "state_timestamp_chunk": state_timestamp_chunk,
            }

            # Update current position tracking
            self.currpos = np.concatenate([
                obs_data["state"]["end_effector/left_hand/observation"],
                obs_data["state"]["gripper/left_hand/observation"],
                obs_data["state"]["end_effector/right_hand/observation"],
                obs_data["state"]["gripper/right_hand/observation"]
            ])
            return obs_data, info_data
        
        try:
            # cprint("Waiting for final observation after reset...", "yellow")
            obs_data = self.obs_queue.get(timeout=MAX_TIMEOUT)
            self.current_obs, self.current_info = _process_observation(obs_data)
            # cprint("Received final observation after reset", "green")
        except queue.Empty:
            cprint("Warning: No observation received, creating dummy observation", "red")
            self.current_obs = self._create_dummy_obs()
            self.current_info = {
                "state_chunk": np.ones((self.max_state_chunk_len, self.state_dim), dtype=np.float32),
                "intervention_chunk": np.ones((self.max_state_chunk_len,), dtype=bool),
                "state_timestamp_chunk": np.ones((self.max_state_chunk_len,), dtype=np.float64),
            }

    def _log_images(self, obs_data: Dict) -> None:
        # display them
        if "images" in obs_data and self.display_image:
            images = obs_data["images"].copy()
            frame_list = []
            for name, img_data in images.items():
                frame_list.append(img_data.astype(np.uint8)) # [H, W, C]
            if frame_list:
                # Concatenate all camera views horizontally
                concatenated_frame = np.concatenate(frame_list, axis=1)
                self.img_queue.put(concatenated_frame)

    def _create_dummy_obs(self) -> Dict:
        """Create a dummy observation for fallback."""
        dummy_obs = {
            "state": {
                "end_effector/left_hand/observation": np.zeros(6, dtype=np.float32),
                "gripper/left_hand/observation": np.zeros(1, dtype=np.float32),
                "end_effector/right_hand/observation": np.zeros(6, dtype=np.float32),
                "gripper/right_hand/observation": np.zeros(1, dtype=np.float32),
                # "tactile/left_hand": np.zeros(1, dtype=np.float32),
                # "tactile/right_hand": np.zeros(1, dtype=np.float32),
            },
            "images": {
                "cam_high": np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8),
                "cam_left_wrist": np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8),
                "cam_right_wrist": np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8),
            },
            "prompt": "put the objects in the box",
        }
        # Update current position tracking
        self.currpos = np.concatenate([
            dummy_obs["state"]["end_effector/left_hand/observation"],
            dummy_obs["state"]["gripper/left_hand/observation"],
            dummy_obs["state"]["end_effector/right_hand/observation"],
            dummy_obs["state"]["gripper/right_hand/observation"],
        ])
        return dummy_obs

    def close(self):
        """Clean up resources."""
        cprint("Closing Dobot environment...", "cyan")

        # Close image displayer
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            try:
                self.displayer.join(timeout=1)
            except:
                pass


def main():
    """Main function to test the environment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dobot Environment Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default='8000', help='Server port')
    parser.add_argument('--fake_env', action='store_true', help='Use fake environment (no robot connection)')
    args = parser.parse_args()
    
    # Create config
    config = None
    config.HOST = args.host
    config.PORT = args.port
    
    # Create environment
    env = DobotEnv(
        fake_env=args.fake_env,
        config=config
    )


if __name__ == '__main__':
    main()