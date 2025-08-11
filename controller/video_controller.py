#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Controller module
----------------------
Handles video loading and pre-processing operations.
"""

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import cv2
import os


class VideoController(QObject):
    """Controller for video operations like loading and basic processing."""

    # Define signals
    video_loaded = pyqtSignal(object)  # Emits video data when loaded

    def __init__(self, encryption_model):
        """Initialize the video controller with a reference to the encryption model."""
        super().__init__()
        self.encryption_model = encryption_model
        self.current_video_path = None
        self.current_video_data = None
        self.frame_count = 0
        self.fps = 0
        self.resolution = (0, 0)

    @pyqtSlot(str)
    def load_video(self, video_path):
        """
        Load a video from the specified path and extract its frames.

        Args:
            video_path: Path to the video file
        """
        if not os.path.exists(video_path):
            return False

        try:
            # Load video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            # Get video properties
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (width, height)

            # Extract frames
            frames = []
            for _ in range(self.frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            cap.release()

            # Store the video data
            self.current_video_path = video_path
            self.current_video_data = {
                "frames": frames,
                "fps": self.fps,
                "resolution": self.resolution,
                "frame_count": self.frame_count,
            }

            # Emit signal that video was loaded successfully
            self.video_loaded.emit(self.current_video_data)

            return True

        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False

    def get_current_video(self):
        """Return the currently loaded video data."""
        return self.current_video_data

    def get_video_info(self):
        """Return information about the currently loaded video."""
        if self.current_video_data is None:
            return None

        return {
            "path": self.current_video_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "resolution": self.resolution,
        }
