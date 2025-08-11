#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Controller module
-------------------------
Handles neural network training and visualization.
"""

from PyQt5.QtCore import QObject, pyqtSlot
import numpy as np
import threading


class TrainingController(QObject):
    def train_decryption_on_real_video(
        self, frames, epochs=10, batch_size=32, validation_split=0.1
    ):
        """
        Train the decryption model to invert the encryption model using real video frames.
        Args:
            frames: List of video frames (numpy arrays)
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Fraction for validation
        """
        self.is_training = True
        try:
            history = self.neural_model.train_on_video_frames(
                frames,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )
            self.training_history = history.history
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
