#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Controller module
-------------------------
Handles neural network training and visualization.
"""

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import numpy as np
import threading


class TrainingController(QObject):
    training_progress = pyqtSignal(int)  # Progress percentage

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
            self.training_progress.emit(20)
            history = self.neural_model.train_on_video_frames(
                frames,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )
            self.training_history = history.history
            self.training_progress.emit(100)
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
        """Generate more effective training data."""
        input_shape = self.neural_model.get_input_shape()

        # Create varied, structured patterns
        X = np.zeros((num_samples,) + input_shape)
        Y = np.zeros((num_samples,) + input_shape)

        for i in range(num_samples):
            # Generate base image with more variation
            img = np.zeros(input_shape)

            # Create varied patterns (choose one randomly)
            pattern_type = np.random.randint(0, 4)

            if pattern_type == 0:  # Gradients
                for x in range(input_shape[0]):
                    for y in range(input_shape[1]):
                        img[x, y, 0] = x / input_shape[0]  # Red gradient
                        img[x, y, 1] = y / input_shape[1]  # Green gradient
                        img[x, y, 2] = (x + y) / (
                            input_shape[0] + input_shape[1]
                        )  # Blue gradient

            elif pattern_type == 1:  # Checkerboard
                check_size = np.random.randint(4, 16)
                for x in range(input_shape[0]):
                    for y in range(input_shape[1]):
                        if (x // check_size + y // check_size) % 2 == 0:
                            img[x, y, :] = [0.9, 0.9, 0.9]
                        else:
                            img[x, y, :] = [0.1, 0.1, 0.1]

            elif pattern_type == 2:  # Random shapes
                # Add 3-5 random shapes
                for _ in range(np.random.randint(3, 6)):
                    shape_type = np.random.randint(0, 2)
                    color = np.random.random(3)

                    if shape_type == 0:  # Circle
                        cx, cy = np.random.randint(
                            10, input_shape[0] - 10
                        ), np.random.randint(10, input_shape[1] - 10)
                        radius = np.random.randint(5, 15)
                        for x in range(
                            max(0, cx - radius), min(input_shape[0], cx + radius)
                        ):
                            for y in range(
                                max(0, cy - radius), min(input_shape[1], cy + radius)
                            ):
                                if (x - cx) ** 2 + (y - cy) ** 2 < radius**2:
                                    img[x, y, :] = color
                    else:  # Rectangle
                        x1, y1 = (
                            np.random.randint(0, input_shape[0] - 10),
                            np.random.randint(0, input_shape[1] - 10),
                        )
                        w, h = np.random.randint(5, 20), np.random.randint(5, 20)
                        x2, y2 = min(input_shape[0], x1 + w), min(
                            input_shape[1], y1 + h
                        )
                        img[x1:x2, y1:y2, :] = color

            else:  # Noise with structure
                img = np.random.random(input_shape) * 0.1
                # Add some structure
                for c in range(3):
                    freq = np.random.randint(2, 6)
                    phase = np.random.random() * 2 * np.pi
                    for x in range(input_shape[0]):
                        for y in range(input_shape[1]):
                            img[x, y, c] += 0.2 + 0.2 * np.sin(
                                freq * x / input_shape[0] * 2 * np.pi + phase
                            )

            X[i] = img

            # Create target by applying a deterministic transformation
            # This is crucial: don't use random noise, but a specific transformation
            Y[i] = np.clip(1.0 - img, 0, 1)  # Simple inversion

        # Apply more varied augmentation
        for i in range(num_samples):
            # After creating the base image:

            # Random brightness and contrast
            brightness = 0.8 + 0.4 * np.random.random()  # 0.8-1.2
            contrast = 0.8 + 0.4 * np.random.random()  # 0.8-1.2
            img = np.clip((img - 0.5) * contrast + 0.5 + (brightness - 1.0), 0, 1)

            # Random noise
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.05, img.shape)
                img = np.clip(img + noise, 0, 1)

        return X, Y

    def _generate_validation_data(self, num_samples=200):
        """Generate consistent validation data with same approach as training data."""
        input_shape = self.neural_model.get_input_shape()

        # Create structured patterns similar to training data
        X_val = np.zeros((num_samples,) + input_shape)

        for i in range(num_samples):
            # Create structured data with gradients, shapes, etc.
            img = np.zeros(input_shape)

            # Add a gradient (slightly different pattern from training)
            for x in range(input_shape[0]):
                for y in range(input_shape[1]):
                    img[x, y, 0] = 1 - (x / input_shape[0])  # Inverted red gradient
                    img[x, y, 1] = 0.5 * (y / input_shape[1])  # Scaled green gradient
                    img[x, y, 2] = (x * y) / (
                        input_shape[0] * input_shape[1]
                    )  # Different blue pattern

            # Add some shapes (using squares instead of circles for variety)
            sx, sy = np.random.randint(5, input_shape[0] - 15), np.random.randint(
                5, input_shape[1] - 15
            )
            size = np.random.randint(5, 10)
            for x in range(sx, min(input_shape[0], sx + size)):
                for y in range(sy, min(input_shape[1], sy + size)):
                    img[x, y, :] = np.random.random(3)  # Random color square

            X_val[i] = img

        # Use the encryption model to create y_val
        y_val = np.copy(X_val)
        for i in range(num_samples):
            # Apply same encryption as training data
            y_val[i] = self.neural_model.encrypt_frame(X_val[i])

        return y_val, X_val  # Switch X and Y for decryption task
