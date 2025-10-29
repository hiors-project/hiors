"""
merge_video.py

Usage:
python tools/merge_video.py \
    --video_folder examples/experiments/dobot_pnp/videos \
    --output_path examples/experiments/dobot_pnp/output.mp4 \
    --fps 5
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted


def merge_videos(video_folder, output_path, fps=30, extension='mp4', quality=7):
    """
    Merge multiple videos from a folder into a single video.
    
    Args:
        video_folder: Path to the folder containing videos
        output_path: Path to save the merged video file
        fps: Frames per second for the output video
        extension: Video file extension to look for
        quality: Video quality (0-10, 10 being highest)
    """
    # Get list of video files in the folder, sorted by name
    video_files = natsorted(glob(os.path.join(video_folder, f'*.{extension}')))
    
    if not video_files:
        print(f"No videos with extension '{extension}' found in {video_folder}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Read all videos and extract frames
    all_frames = []
    total_frames = 0
    
    for video_file in tqdm(video_files, desc="Reading videos"):
        try:
            # Read the video using cv2
            cap = cv2.VideoCapture(video_file)
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue
            
            # Extract all frames from this video
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            all_frames.extend(frames)
            total_frames += len(frames)
            
            print(f"Added {len(frames)} frames from {os.path.basename(video_file)}")
            
        except Exception as e:
            print(f"Error reading {video_file}: {e}")
    
    if not all_frames:
        print("No valid videos found")
        return
    
    # Get video dimensions from the first frame
    height, width = all_frames[0].shape[:2]
    
    # Set up video writer with cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write all frames to the output video
    print(f"Saving merged video to {output_path} with {total_frames} frames at {fps} FPS")
    for frame in tqdm(all_frames, desc="Writing frames"):
        out.write(frame)
    
    out.release()
    print(f"Merged video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple videos from a folder into a single video")
    parser.add_argument("--video_folder", required=True, help="Path to the folder containing videos")
    parser.add_argument("--output_path", required=True, help="Path to save the merged video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    parser.add_argument("--extension", default="mp4", help="Video file extension to look for")
    parser.add_argument("--quality", type=int, default=7, help="Video quality (0-10, 10 being highest)")
    
    args = parser.parse_args()
    
    merge_videos(
        args.video_folder,
        args.output_path,
        fps=args.fps,
        extension=args.extension,
        quality=args.quality
    )
