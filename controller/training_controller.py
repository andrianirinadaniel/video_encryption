#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Controller module
-------------------------
Handles neural network training and visualization.
"""

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QCoreApplication
import numpy as np
import threading
import cv2
from tensorflow.keras.callbacks import Callback


class TrainingController(QObject):
    # Define signals for updating UI
    progress_updated = pyqtSignal(int)  # Progress percentage (0-100)
    status_updated = pyqtSignal(str)  # Status message

    def train_decryption_on_real_video(
        self, frames, epochs=10, batch_size=32, validation_split=0.1, augment_data=False
    ):
        """
        Train the decryption model to invert the encryption model using real video frames.
        Args:
            frames: List of video frames (numpy arrays)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            augment_data: Whether to use data augmentation to create more training samples
        """
        self.is_training = True
        try:
            # First, update the UI to show we're starting
            self.progress_updated.emit(5)

            # Data augmentation to create more training samples
            if augment_data and len(frames) > 0:
                augmented_frames = []
                total_frames = len(frames)

                print(f"Starting data augmentation on {total_frames} frames...")
                self.status_updated.emit("Starting data augmentation...")

                for i, frame in enumerate(frames):
                    # Update progress for augmentation phase (use first 50% of progress bar)
                    progress = int(
                        5 + (i / total_frames) * 45
                    )  # 5-50% for augmentation
                    self.progress_updated.emit(progress)

                    # Process events to keep UI responsive
                    QCoreApplication.processEvents()

                    # Add original frame
                    augmented_frames.append(frame)

                    # Add flipped version (horizontal)
                    augmented_frames.append(cv2.flip(frame, 1))

                    # Add rotated versions (90, 180, 270 degrees)
                    augmented_frames.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
                    augmented_frames.append(cv2.rotate(frame, cv2.ROTATE_180))
                    augmented_frames.append(
                        cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    )

                    # Add brightness variations
                    bright_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                    dark_frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10)
                    augmented_frames.append(bright_frame)
                    augmented_frames.append(dark_frame)

                msg = f"Data augmentation: Increased training set from {len(frames)} to {len(augmented_frames)} frames"
                print(msg)
                self.status_updated.emit(msg)
                frames = augmented_frames

            self.progress_updated.emit(50)  # 50% progress after augmentation
            self.status_updated.emit(
                f"Training decryption model on {len(frames)} frames..."
            )

            # Train model with potentially augmented data
            # Since we can't pass callbacks directly, we'll use epoch-by-epoch progress updates
            total_epochs = epochs

            # Create a manual progress update function
            def update_progress_after_epoch(epoch, total_epochs):
                # Calculate progress: 50-95% range for training
                progress = int(50 + ((epoch + 1) / total_epochs) * 45)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"Training epoch {epoch+1}/{total_epochs}")

            # Connect to neural model's training_progress signal if available
            try:
                # Temporarily connect to the training progress signal
                self.neural_model.training_progress.connect(
                    lambda data: self.status_updated.emit(
                        f"Epoch {data['epoch']}/{total_epochs}: acc={data['accuracy']:.4f}, loss={data['loss']:.4f}"
                    )
                )
            except Exception as e:
                print(f"Could not connect to training progress signal: {str(e)}")

            # Train model with progress reporting
            history = self.neural_model.train_on_video_frames(
                frames,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )

            # Update progress for each epoch that was completed
            for i in range(total_epochs):
                update_progress_after_epoch(i, total_epochs)

            self.training_history = history.history
            self.progress_updated.emit(100)  # Complete the progress bar
            self.status_updated.emit("Training completed successfully!")
        except Exception as e:
            print(f"Error during real-data training: {str(e)}")
        finally:
            self.is_training = False

    """Controller for neural network training operations."""

    def __init__(self, neural_model):
        """Initialize the training controller with a reference to the neural network model."""
        super().__init__()
        self.neural_model = neural_model
        self.training_thread = None
        self.training_history = None
        self.is_training = False

    @pyqtSlot()
    def start_new_training(self, epochs=10, batch_size=32):
        """
        Start a new training session for the neural network.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.is_training:
            return False

        self.is_training = True

        # Start training in a separate thread to avoid blocking the UI
        self.training_thread = threading.Thread(
            target=self._run_training, args=(epochs, batch_size)
        )
        self.training_thread.daemon = True
        self.training_thread.start()

        return True

    def _run_training(self, epochs, batch_size):
        """
        Run the actual training process in a separate thread.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            # Generate some sample data if needed
            X_train, y_train = self._generate_training_data()
            X_val, y_val = self._generate_validation_data()

            # Reset the neural network models to avoid optimizer variable conflicts
            self.neural_model.reset_models()

            # Train the model
            self.training_history = self.neural_model.train(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
            )

        except Exception as e:
            print(f"Error during training: {str(e)}")

        finally:
            self.is_training = False

    def get_training_history(self):
        """Return the training history for visualization."""
        return self.training_history

    def _generate_training_data(self, num_samples=1000):
        """
        Generate sample training data for the neural network.
        In a real application, this would use actual video frames.

        Args:
            num_samples: Number of training samples to generate

        Returns:
            Tuple of (X_train, y_train) numpy arrays
        """
        # For this example, generating random data
        # In a real application, this would use actual video frame data
        input_shape = self.neural_model.get_input_shape()

        # Generate random input data
        X_train = np.random.random((num_samples,) + input_shape)

        # Generate corresponding output data (encrypted/decrypted version)
        # For this example, just using the same data with some modifications
        y_train = X_train + 0.1 * np.random.random((num_samples,) + input_shape)

        return X_train, y_train

    def _generate_validation_data(self, num_samples=200):
        """
        Generate sample validation data for the neural network.

        Args:
            num_samples: Number of validation samples to generate

        Returns:
            Tuple of (X_val, y_val) numpy arrays
        """
        # Similar to training data generation but for validation
        input_shape = self.neural_model.get_input_shape()

        X_val = np.random.random((num_samples,) + input_shape)
        y_val = X_val + 0.1 * np.random.random((num_samples,) + input_shape)

        return X_val, y_val
