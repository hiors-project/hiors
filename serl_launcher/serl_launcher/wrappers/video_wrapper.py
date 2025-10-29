from collections import OrderedDict
import os
import cv2
import shutil
from datetime import datetime
import gym
import numpy as np
from typing import List
from termcolor import cprint


class VideoWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        image_keys: List[str] = None,
        name: str = "video",
        video_dir: str = "./videos",
        fps: float = 5.0,
    ):
        super().__init__(env)
        self._name = name
        self.image_keys = image_keys
        self.video_dir = video_dir
        self.fps = fps
        self.recording_frames = []
        self.recording_infos = []
        
        # Initialize video recording if enabled
        cprint("Video recording enabled in VideoWrapper!", "yellow")
        # NOTE: this will cause error when using multiple processes
        # clean video directory
        if os.path.exists(self.video_dir):
            shutil.rmtree(self.video_dir)
            cprint(f"Video directory cleaned: {os.path.abspath(self.video_dir)}", "red")

    def _add_frame(self, obs, reward=0.0, current_subtask_label=None, last_subtask_label=None):
        """Add frames for video recording if enabled"""
        frame_list = []
        for key in self.image_keys:
            if key in obs:
                img_data = obs[key]
                if img_data.ndim == 4:
                    # Convert (T, H, W, C) to (H, W, C) for single frame
                    img_data = img_data[0]
                # Convert to uint8 if not already
                if img_data.dtype != np.uint8:
                    img_data = (img_data * 255).astype(np.uint8)
                # Frame should be in BGR format for OpenCV
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                frame_list.append(img_data)  # [H, W, C]

        if frame_list:
            # Concatenate all camera views horizontally
            concatenated_frame = np.concatenate(frame_list, axis=1)
            self.recording_frames.append(concatenated_frame)
            # Store reward info and subtask labels for this frame
            self.recording_infos.append({
                'reward': reward,
                'current_subtask_label': current_subtask_label,
                'last_subtask_label': last_subtask_label,
            })

    def reset(self, **kwargs):
        # Save video from previous episode before clearing
        self.save_video_recording()

        obs, info = super().reset(**kwargs)
        # Try to get subtask labels from the environment if it has them
        current_subtask_label = getattr(self.env, 'current_subtask_label', None)  # will be removed
        last_subtask_label = getattr(self.env, 'last_subtask_label', None)
        # current_subtask_label = self.env.get_wrapper_attr('current_subtask_label')
        # last_subtask_label = self.env.get_wrapper_attr('last_subtask_label')
        self._add_frame(obs, reward=0.0, current_subtask_label=current_subtask_label, last_subtask_label=last_subtask_label)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, done, truncate, info = super().step(action)
        # Try to get subtask labels from the environment if it has them
        current_subtask_label = getattr(self.env, 'current_subtask_label', None)  # will be removed
        last_subtask_label = getattr(self.env, 'last_subtask_label', None)
        # current_subtask_label = self.env.get_wrapper_attr('current_subtask_label')
        # last_subtask_label = self.env.get_wrapper_attr('last_subtask_label')
        self._add_frame(obs, reward=reward, current_subtask_label=current_subtask_label, last_subtask_label=last_subtask_label)
        return obs, reward, done, truncate, info

    def _add_text_overlay(self, frame, reward_info, frame_idx):
        """Add additional info to the frame."""
        frame_with_text = frame.copy()
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1   # should be integer
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        
        # Prepare text lines
        reward = reward_info['reward']
        current_subtask_label = reward_info.get('current_subtask_label')
        last_subtask_label = reward_info.get('last_subtask_label')

        lines = [
            f"Frame: {frame_idx}",
            f"Rew: {reward:.2f}",
        ]
        
        # Add subtask label info if available
        if current_subtask_label is not None:
            lines.append(f"Curr Task: {current_subtask_label}")
        if last_subtask_label is not None:
            lines.append(f"Last Task: {last_subtask_label}")
        y_offset = 30   # initial vertical margin for text
        for i, line in enumerate(lines):
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Position in top-left corner with some padding
            x = 10
            y = y_offset + i * (text_height + 10)
            
            # Draw background rectangle
            cv2.rectangle(frame_with_text, 
                         (x - 5, y - text_height - 5), 
                         (x + text_width + 5, y + baseline + 5), 
                         bg_color, -1)
            cv2.putText(frame_with_text, line, (x, y), font, font_scale, text_color, thickness)
        
        return frame_with_text

    def save_video_recording(self):
        if len(self.recording_frames) < 3:
            cprint("Not enough frames to save video.", "yellow")
            return
        
        try:
            os.makedirs(self.video_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(self.video_dir, f'{timestamp}.mp4')
            
            # Use OpenCV VideoWriter instead of imageio to avoid forking
            frame_height, frame_width = self.recording_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (frame_width, frame_height))
            
            for frame_idx, frame in enumerate(self.recording_frames):
                reward_info = self.recording_infos[frame_idx] \
                    if frame_idx < len(self.recording_infos) else {
                        'reward': 0.0, 
                        'current_subtask_label': None, 
                        'last_subtask_label': None
                    }
                frame_with_text = self._add_text_overlay(frame, reward_info, frame_idx)
                out.write(frame_with_text)
            
            out.release()
            cprint(f"Saved video at {video_path}", "green")
            self.recording_frames.clear()
            self.recording_infos.clear()
        except Exception as e:
            cprint(f"Failed to save video: {e}", "red")

    def close(self):
        self.save_video_recording()
        super().close()
