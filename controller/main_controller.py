#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Controller module for the Video Encryption Application
---------------------------------------------------------
Handles the initialization of views and connects UI events to model actions.
"""

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMessageBox
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
        # Add this connection for the real training button
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

        # Connect encryption model signals for video display
        self.encryption_model.video_encrypted.connect(self.display_encrypted_video)
        self.encryption_model.video_decrypted.connect(self.display_decrypted_video)

    def handle_video_loaded(self, result):
        """Handle the video loaded event from the video controller."""
        if result["success"]:
            # Update the original video player with the loaded frames
            frames = result["frames"]
            fps = result["fps"]
            self.main_view.update_video_display("original", frames, fps)

            # Enable encrypt button after video is loaded successfully
            self.main_view.encrypt_btn.setEnabled(True)

            # Update status
            self.main_view.status_label.setText(f"Video loaded: {len(frames)} frames")
        else:
            # Show error message
            error_msg = result.get("error", "Unknown error")
            QMessageBox.critical(self.main_view, "Error Loading Video", error_msg)

    def process_real_training(self):
        """Train using real video frames."""
        video_data = self.video_controller.get_current_video()

        if (
            not video_data
            or "frames" not in video_data
            or len(video_data["frames"]) == 0
        ):
            QMessageBox.warning(
                self.main_view, "No Video", "Please load a video first!"
            )
            return

        self.main_view.update_progress(10)
        self.main_view.status_label.setText("Training with real video data...")

        # Use your existing train_on_video_frames method
        result = self.neural_model.train_on_video_frames(
            video_data["frames"], epochs=20, batch_size=16, validation_split=0.2
        )

        self.main_view.update_progress(100)
        self.main_view.status_label.setText("Training completed!")

    def process_encryption(self):
        """Handle the encryption process."""
        video_data = self.video_controller.get_current_video()

        if not video_data or "frames" not in video_data:
            self.main_view.status_label.setText("No video loaded for encryption")
            return

        # Set UI to busy state
        self.main_view.set_ui_busy(True)

        try:
            # Get original frames
            original_frames = video_data["frames"]
            fps = video_data.get("fps", 30)

            # Encrypt frames
            encrypted_frames = self.encryption_model.encrypt_video(original_frames)

            # Update UI with encrypted frames
            self.main_view.update_video_display("encrypted", encrypted_frames, fps)

            # Calculate metrics between original and encrypted frames
            metrics = self.metrics_controller.calculate_metrics(
                original_frames, encrypted_frames
            )

            # Update metrics display
            self.main_view.metrics_display.update_metrics(metrics)

            # Store encrypted frames for later use
            self.encrypted_frames = encrypted_frames

            # Update status
            self.main_view.status_label.setText("Video encrypted successfully")

        except Exception as e:
            self.main_view.status_label.setText(f"Error during encryption: {str(e)}")

        finally:
            # Set UI back to ready state
            self.main_view.set_ui_busy(False)

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

    def _on_train_new_model(self):
        """Handle the new training button click."""
        self.new_training_signal.emit(epochs=20, batch_size=32)
        self.status_label.setText("Training new model with extended epochs...")
