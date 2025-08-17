#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Controller module for the Video Encryption Application
---------------------------------------------------------
Handles the initialization of views and connects UI events to model actions.
"""

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog
import cv2
from view.main_view import MainView
from model.encryption_model import EncryptionModel
from model.neural_network_model import NeuralNetworkModel
from controller.video_controller import VideoController
from controller.metrics_controller import MetricsController
from controller.training_controller import TrainingController


class MainController(QObject):
    """Main controller for the application, orchestrates all other controllers and views."""

    def __init__(self):
        """Initialize the main controller and its dependencies."""
        super().__init__()

        # Initialize models
        self.encryption_model = EncryptionModel()
        self.neural_model = NeuralNetworkModel()

        # Initialize sub-controllers
        self.video_controller = VideoController(self.encryption_model)
        self.metrics_controller = MetricsController()
        self.training_controller = TrainingController(self.neural_model)

        # Initialize the main view
        self.main_view = MainView(self)

        # Connect signals and slots
        self._connect_signals()

    def show_main_view(self):
        """Display the main application window."""
        self.main_view.show()

    def _connect_signals(self):
        """Connect UI signals to appropriate controller methods."""
        # File operations
        self.main_view.load_video_signal.connect(self.video_controller.load_video)

        # Processing operations
        self.main_view.encrypt_signal.connect(self.process_encryption)
        self.main_view.decrypt_signal.connect(self.process_decryption)
        self.main_view.new_training_signal.connect(
            self.training_controller.start_new_training
        )
        self.main_view.real_training_signal.connect(self.process_real_training)

        # Connect model signals to view updates
        self.encryption_model.progress_updated.connect(self.main_view.update_progress)
        self.encryption_model.processing_finished.connect(
            self.main_view.processing_finished
        )
        self.neural_model.training_progress.connect(
            self.main_view.update_training_graphs
        )

        # Connect video controller signals
        self.video_controller.video_loaded.connect(self.handle_video_loaded)
        self.video_controller.loading_progress.connect(self.main_view.update_progress)

        # Connect training controller signals
        self.training_controller.progress_updated.connect(
            self.main_view.update_progress
        )
        self.training_controller.status_updated.connect(
            self.main_view.status_label.setText
        )

        # Connect encryption model signals for video display
        self.encryption_model.video_encrypted.connect(self.display_encrypted_video)
        self.encryption_model.video_decrypted.connect(self.display_decrypted_video)

    def handle_video_loaded(self, result):
        """
        Handle the video loaded event and display the video.

        Args:
            result: Dictionary containing loading result and video data
        """
        if result.get("success", False):
            self.display_original_video(result)
        self.main_view.video_loaded_callback(result)

    def collect_training_frames(self):
        """Collect frames from multiple videos for training."""

        collected_frames = []
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.main_view,
            "Select Training Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )

        if not file_paths:
            # If no videos selected, try to use the currently loaded video
            video_data = self.video_controller.get_current_video()
            if video_data and "frames" in video_data and len(video_data["frames"]) > 0:
                frames = video_data["frames"]
                input_shape = self.neural_model.get_input_shape()
                collected_frames = [
                    cv2.resize(f, (input_shape[1], input_shape[0])) for f in frames
                ]
                self.main_view.status_label.setText(
                    f"Using current video: {len(collected_frames)} frames"
                )
            return collected_frames

        self.main_view.status_label.setText(
            f"Loading {len(file_paths)} videos for training..."
        )
        self.main_view.progress_bar.setValue(0)

        total_files = len(file_paths)
        for i, path in enumerate(file_paths):
            # Update progress
            progress = int((i / total_files) * 100)
            self.main_view.progress_bar.setValue(progress)

            # Extract frames using video controller's method but don't update UI
            import os
            from utils.video_utils import extract_frames_from_video

            if os.path.exists(path):
                result = extract_frames_from_video(path)
                if result:
                    frames, fps, resolution, _ = result
                    # Resize frames to model input shape
                    input_shape = self.neural_model.get_input_shape()
                    resized_frames = [
                        cv2.resize(f, (input_shape[1], input_shape[0])) for f in frames
                    ]
                    collected_frames.extend(resized_frames)
                    self.main_view.status_label.setText(
                        f"Collected {len(collected_frames)} frames so far..."
                    )

        self.main_view.progress_bar.setValue(100)
        self.main_view.status_label.setText(
            f"Training data collection complete: {len(collected_frames)} frames"
        )
        return collected_frames

    def process_real_training(self):
        """Train the decryption model using real video frames from multiple sources."""
        # Collect frames from multiple videos
        frames = self.collect_training_frames()

        if frames and len(frames) > 0:
            self.main_view.status_label.setText(
                f"Training decryption model with {len(frames)} frames..."
            )
            self.training_controller.train_decryption_on_real_video(
                frames, epochs=10, batch_size=32, augment_data=True
            )
            self.main_view.status_label.setText(
                "Decryption model trained on real data."
            )
        else:
            self.main_view.status_label.setText(
                "No videos available for real-data training."
            )

    def process_encryption(self):
        """Handle the encryption process."""
        video_data = self.video_controller.get_current_video()
        if video_data is not None:
            # Start encryption in a separate thread
            self.encryption_model.encrypt_video(video_data, self.neural_model)
            self.main_view.set_ui_busy(True)

    def process_decryption(self):
        """Handle the decryption process."""
        encrypted_data = self.encryption_model.get_encrypted_data()
        if encrypted_data is not None:
            # Start decryption in a separate thread
            self.encryption_model.decrypt_video(encrypted_data, self.neural_model)
            self.main_view.set_ui_busy(True)

    def get_metrics_data(self):
        """Retrieve current metrics data for display."""
        return self.metrics_controller.get_current_metrics()

    def display_original_video(self, video_data):
        """
        Display the original video in the UI.

        Args:
            video_data: Dictionary containing video frames and metadata
        """
        if video_data and "frames" in video_data and len(video_data["frames"]) > 0:
            # Update the main view with the video frames
            self.main_view.update_video_display(
                "original",
                video_data["frames"],
                video_data.get("fps", 30),
            )
        else:
            print("[DEBUG] No frames found in video_data or video_data is None")
            self.main_view.status_label.setText("No frames found in loaded video.")

    def display_encrypted_video(self, encrypted_data):
        """
        Display the encrypted video in the UI.

        Args:
            encrypted_data: Dictionary containing encrypted video frames and metadata
        """
        if (
            encrypted_data
            and "frames" in encrypted_data
            and len(encrypted_data["frames"]) > 0
        ):
            # Set frames in the encrypted video panel
            self.main_view.update_video_display(
                "encrypted",
                encrypted_data["frames"],
                encrypted_data.get("fps", 30),
            )

            # Calculate and display metrics between original and encrypted
            if (
                self.video_controller.current_video_data
                and len(self.video_controller.current_video_data["frames"]) > 0
            ):
                original_frame = self.video_controller.current_video_data["frames"][0]
                encrypted_frame = encrypted_data["frames"][0]
                metrics = self.metrics_controller.calculate_metrics(
                    original_frame, encrypted_frame
                )
                self.main_view.metrics_display.update_metrics(metrics)

    def display_decrypted_video(self, decrypted_data):
        """
        Display the decrypted video in the UI.

        Args:
            decrypted_data: Dictionary containing decrypted video frames and metadata
        """
        if (
            decrypted_data
            and "frames" in decrypted_data
            and len(decrypted_data["frames"]) > 0
        ):
            # Set frames in the decrypted video panel
            self.main_view.update_video_display(
                "decrypted",
                decrypted_data["frames"],
                decrypted_data.get("fps", 30),
            )

            # Calculate and display metrics between original and decrypted
            if (
                self.video_controller.current_video_data
                and len(self.video_controller.current_video_data["frames"]) > 0
            ):
                original_frame = self.video_controller.current_video_data["frames"][0]
                decrypted_frame = decrypted_data["frames"][0]
                metrics = self.metrics_controller.calculate_metrics(
                    original_frame, decrypted_frame
                )
                self.main_view.metrics_display.update_metrics(metrics)
