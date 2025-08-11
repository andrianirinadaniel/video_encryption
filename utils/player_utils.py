#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Player Utilities module
---------------------------
Helper functions for video playback in the UI.
"""

from PyQt5.QtCore import QTimer, QObject, pyqtSignal
import cv2


class VideoPlayer(QObject):
    """
    Simple video player for frame-by-frame playback of video data in the UI.
    """

    # Signals
    frame_changed = pyqtSignal(object, int)  # Current frame and frame index
    playback_finished = pyqtSignal()

    def __init__(self, fps=30):
        """Initialize the video player with default frame rate."""
        super().__init__()
        self.frames = []
        self.fps = fps
        self.current_frame_idx = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

    def load_video(self, video_data):
        """
        Load video data for playback.

        Args:
            video_data: Dictionary containing video frames and metadata
        """
        if video_data and "frames" in video_data:
            self.frames = video_data["frames"]
            self.fps = video_data.get("fps", 30)
            self.current_frame_idx = 0

            # Set timer interval based on fps
            interval = int(1000 / self.fps)  # Convert fps to milliseconds
            self.timer.setInterval(interval)

            # Emit the first frame
            if len(self.frames) > 0:
                self.frame_changed.emit(self.frames[0], 0)

            return True
        return False

    def play(self):
        """Start playing the video."""
        if len(self.frames) > 0 and not self.is_playing:
            self.is_playing = True
            self.timer.start()

    def pause(self):
        """Pause the video playback."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()

    def stop(self):
        """Stop the video playback and reset to the beginning."""
        self.is_playing = False
        self.timer.stop()
        self.current_frame_idx = 0
        if len(self.frames) > 0:
            self.frame_changed.emit(self.frames[0], 0)

    def seek(self, frame_idx):
        """
        Seek to a specific frame in the video.

        Args:
            frame_idx: Index of the frame to seek to
        """
        if 0 <= frame_idx < len(self.frames):
            self.current_frame_idx = frame_idx
            self.frame_changed.emit(self.frames[frame_idx], frame_idx)

    def _next_frame(self):
        """Display the next frame in the sequence."""
        if len(self.frames) == 0:
            self.stop()
            return

        # Advance to next frame
        self.current_frame_idx += 1

        # Check if reached the end of the video
        if self.current_frame_idx >= len(self.frames):
            self.current_frame_idx = 0  # Loop back to the beginning
            self.playback_finished.emit()

        # Emit current frame
        self.frame_changed.emit(
            self.frames[self.current_frame_idx], self.current_frame_idx
        )


def create_video_mosaic(frames, rows=1, cols=3, frame_size=None):
    """
    Create a mosaic of video frames.

    Args:
        frames: List of frames to combine
        rows: Number of rows in the mosaic
        cols: Number of columns in the mosaic
        frame_size: Optional tuple (width, height) to resize frames before combining

    Returns:
        Combined mosaic frame
    """
    import numpy as np

    # Ensure we have enough frames
    if len(frames) == 0:
        return None

    # Fill with blank frames if needed
    while len(frames) < rows * cols:
        # Create blank frame with same shape as first frame
        blank = np.zeros_like(frames[0])
        frames.append(blank)

    # Resize frames if needed
    if frame_size is not None:
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, frame_size)
            resized_frames.append(resized)
        frames = resized_frames

    # Combine frames into rows
    rows_list = []
    for i in range(rows):
        start = i * cols
        end = start + cols
        if end > len(frames):
            end = len(frames)
        row_frames = frames[start:end]
        row = np.hstack(row_frames)
        rows_list.append(row)

    # Combine rows into final mosaic
    mosaic = np.vstack(rows_list)

    return mosaic
