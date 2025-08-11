#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Utilities module
--------------------
Helper functions for video processing and management.
"""

import cv2
import os
import numpy as np


def save_video_from_frames(frames, output_path, fps=30, resolution=None):
    """
    Save a list of frames as a video file.

    Args:
        frames: List of image frames (numpy arrays)
        output_path: Path to save the video to
        fps: Frames per second for the output video
        resolution: Optional tuple (width, height) for video resolution

    Returns:
        True if successful, False otherwise
    """
    if not frames or len(frames) == 0:
        print("Error: No frames to save.")
        return False

    try:
        # Get video dimensions from the first frame if not provided
        if resolution is None:
            height, width = frames[0].shape[:2]
        else:
            width, height = resolution

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec for MP4 files
        output_dir = os.path.dirname(output_path)

        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create the video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to the video
        for frame in frames:
            # Resize frame if necessary
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            # Ensure frame is in BGR format for OpenCV
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            out.write(frame)

        # Release the video writer
        out.release()

        return True

    except Exception as e:
        print(f"Error saving video: {str(e)}")
        return False


def extract_frames_from_video(video_path, max_frames=None):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        Tuple of (frames, fps, resolution, frame_count) or None if failed
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist: {video_path}")
        return None

    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return None

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (width, height)

        # Limit the number of frames if specified
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        # Extract frames
        frames = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Release the video capture
        cap.release()

        return frames, fps, resolution, len(frames)

    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return None


def resize_frame(frame, target_size):
    """
    Resize a frame to the target size.

    Args:
        frame: Input frame (numpy array)
        target_size: Tuple of (width, height)

    Returns:
        Resized frame
    """
    if frame is None or target_size is None:
        return frame

    try:
        width, height = target_size
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame
    except Exception as e:
        print(f"Error resizing frame: {str(e)}")
        return frame


def create_thumbnail(frame, max_size=(320, 240)):
    """
    Create a thumbnail of a frame.

    Args:
        frame: Input frame (numpy array)
        max_size: Maximum size for the thumbnail (width, height)

    Returns:
        Thumbnail frame
    """
    if frame is None:
        return None

    try:
        # Get original dimensions
        height, width = frame.shape[:2]

        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = min(width, max_size[0])
            new_height = int(height * (new_width / width))
        else:
            new_height = min(height, max_size[1])
            new_width = int(width * (new_height / height))

        # Resize the frame
        thumbnail = cv2.resize(frame, (new_width, new_height))
        return thumbnail
    except Exception as e:
        print(f"Error creating thumbnail: {str(e)}")
        return frame
