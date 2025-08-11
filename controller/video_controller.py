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
    loading_progress = pyqtSignal(int)  # Emits loading progress (0-100%)

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
            self.video_loaded.emit({"success": False, "error": "File not found"})
            return False

        try:
            # Load video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.video_loaded.emit(
                    {"success": False, "error": "Could not open video file"}
                )
                return False

            # Get video properties
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (width, height)

            # Extract frames with progress updates
            frames = []
            progress_interval = max(1, self.frame_count // 100)

            # Emit initial progress
            self.loading_progress.emit(5)

            for idx in range(self.frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

                # Update progress bar every progress_interval frames
                if idx % progress_interval == 0 or idx == self.frame_count - 1:
                    percent = int(
                        5 + (idx + 1) / self.frame_count * 90
                    )  # 5-95% for loading
                    self.loading_progress.emit(percent)

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
            self.loading_progress.emit(100)
            self.video_loaded.emit(
                {
                    "success": True,
                    "frames": frames,
                    "fps": self.fps,
                    "resolution": self.resolution,
                    "frame_count": self.frame_count,
                }
            )

            return True

        except Exception as e:
            print(f"Error loading video: {str(e)}")
            self.video_loaded.emit({"success": False, "error": str(e)})
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
