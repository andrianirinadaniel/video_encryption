#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural Network Model module
--------------------------
Implements the neural network for video encryption/decryption.
"""

from PyQt5.QtCore import QObject, pyqtSignal
import tensorflow as tf
import numpy as np
import os
import cv2


class ProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to emit progress updates during training."""

    def __init__(self, model_instance):
        super().__init__()
        self.model_instance = model_instance

    def on_epoch_end(self, epoch, logs=None):
        """Emit progress signal at the end of each epoch."""
        if logs is None:
            logs = {}

        # Send progress update with current epoch info
        progress_data = {
            "epoch": epoch + 1,
            "accuracy": logs.get("accuracy", 0),
            "loss": logs.get("loss", 0),
            "val_accuracy": logs.get("val_accuracy", 0),
            "val_loss": logs.get("val_loss", 0),
        }

        self.model_instance.training_progress.emit(progress_data)


class NeuralNetworkModel(QObject):
    def train_on_video_frames(
        self, frames, epochs=20, batch_size=16, validation_split=0.2
    ):
        """
        Train the decryption model using real video frames.

        Args:
            frames: List or array of video frames (numpy arrays, shape HxWx3, values 0-255)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        # Preprocess frames
        processed_frames = []
        encrypted_frames = []

        for frame in frames:
            # Convert to float and normalize
            norm_frame = frame.astype(np.float32) / 255.0

            # Resize if needed
            if norm_frame.shape[:2] != self.input_shape[:2]:
                norm_frame = cv2.resize(
                    norm_frame, (self.input_shape[1], self.input_shape[0])
                )

            # Generate encrypted version using the encryption model
            encrypted = self.encrypt_frame(norm_frame)

            processed_frames.append(norm_frame)
            encrypted_frames.append(encrypted)

        # Convert to numpy arrays
        X = np.array(encrypted_frames)  # Encrypted frames (input)
        Y = np.array(processed_frames)  # Original frames (target)

        # Split into training and validation sets
        val_size = int(len(X) * validation_split)
        if val_size > 0:
            X_train, X_val = X[:-val_size], X[-val_size:]
            Y_train, Y_val = Y[:-val_size], Y[-val_size:]
            validation_data = (X_val, Y_val)
        else:
            X_train, Y_train = X, Y
            validation_data = None

        # Train the decryption model only
        print(
            f"Training decryption model with {len(X_train)} frames and {val_size} validation frames"
        )

        # Use early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.decryption_model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[ProgressCallback(self), early_stopping],
        )

        # Save the model after training
        self.save_models()

        return history

    """Model for neural network operations."""

    # Signal for training progress updates
    training_progress = pyqtSignal(dict)  # Dict with epoch, accuracy, loss

    def __init__(self):
        """Initialize the neural network model."""
        super().__init__()
        self.input_shape = (64, 64, 3)  # Default input shape
        self.encryption_model = None
        self.decryption_model = None
        self._build_models()
        self._load_pretrained_models()

    def _build_models(self):
        """Build more advanced neural network models."""
        # Encryption model with residual connections and batch normalization
        enc_input = tf.keras.layers.Input(shape=self.input_shape)

        # Initial convolution
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(enc_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Add residual blocks
        for _ in range(3):
            residual = x
            x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.Activation("relu")(x)

        # Final convolution to output
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        enc_output = tf.keras.layers.Conv2D(
            3, (3, 3), activation="sigmoid", padding="same"
        )(x)

        # Create model
        self.encryption_model = tf.keras.Model(enc_input, enc_output)
        self.encryption_model.compile(
            optimizer="adam",
            loss="mse",  # Mean squared error is good
            metrics=[
                tf.keras.metrics.MeanSquaredError(),
                "mae",
            ],  # Add mean absolute error
        )

        # Decryption model (similar architecture but can be different)
        dec_input = tf.keras.layers.Input(shape=self.input_shape)

        # Encoder layers
        y = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(
            dec_input
        )
        y = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(y)
        y = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(y)
        y = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(y)

        # Transformer layers
        y = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(y)

        # Decoder layers
        y = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(y)
        y = tf.keras.layers.UpSampling2D((2, 2))(y)
        y = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(y)
        y = tf.keras.layers.UpSampling2D((2, 2))(y)

        # Output layer
        dec_output = tf.keras.layers.Conv2D(
            3, (3, 3), activation="sigmoid", padding="same"
        )(y)

        # Create the decryption model
        self.decryption_model = tf.keras.Model(dec_input, dec_output)

        # Create a custom optimizer with explicit configuration
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            name="adam_decryption",
        )

        self.decryption_model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[tf.keras.metrics.MeanSquaredError(), "mae"],
        )

    def _load_pretrained_models(self):
        """Load pre-trained models if available."""
        # Check if models exist
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        # Try modern Keras format first
        encryption_path_keras = os.path.join(models_dir, "encryption_model.keras")
        decryption_path_keras = os.path.join(models_dir, "decryption_model.keras")

        # Fallback paths for HDF5 format
        encryption_path_h5 = os.path.join(models_dir, "encryption_model.h5")
        decryption_path_h5 = os.path.join(models_dir, "decryption_model.h5")

        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Try to load encryption model
        encryption_loaded = False

        # Try Keras format first
        if os.path.exists(encryption_path_keras):
            try:
                self.encryption_model = tf.keras.models.load_model(
                    encryption_path_keras
                )
                print("Loaded pre-trained encryption model (.keras format)")
                encryption_loaded = True
            except Exception as e:
                print(f"Failed to load encryption model (.keras format): {str(e)}")

        # Try H5 format if Keras format failed
        if not encryption_loaded and os.path.exists(encryption_path_h5):
            try:
                custom_objects = {"mse": "mse"}  # Map function names to strings
                self.encryption_model = tf.keras.models.load_model(
                    encryption_path_h5, custom_objects=custom_objects
                )
                print("Loaded pre-trained encryption model (.h5 format)")
            except Exception as e:
                print(f"Failed to load encryption model (.h5 format): {str(e)}")
                # We'll use the default model created in _build_models

        # Try to load decryption model
        decryption_loaded = False

        # Try Keras format first
        if os.path.exists(decryption_path_keras):
            try:
                self.decryption_model = tf.keras.models.load_model(
                    decryption_path_keras
                )
                print("Loaded pre-trained decryption model (.keras format)")
                decryption_loaded = True
            except Exception as e:
                print(f"Failed to load decryption model (.keras format): {str(e)}")

        # Try H5 format if Keras format failed
        if not decryption_loaded and os.path.exists(decryption_path_h5):
            try:
                custom_objects = {"mse": "mse"}  # Map function names to strings
                self.decryption_model = tf.keras.models.load_model(
                    decryption_path_h5, custom_objects=custom_objects
                )
                print("Loaded pre-trained decryption model (.h5 format)")
            except Exception as e:
                print(f"Failed to load decryption model (.h5 format): {str(e)}")
                # We'll use the default model created in _build_models

    def save_models(self):
        """Save the current models."""
        # Create models directory
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save models with modern Keras format
        encryption_path = os.path.join(models_dir, "encryption_model.keras")
        decryption_path = os.path.join(models_dir, "decryption_model.keras")

        try:
            self.encryption_model.save(encryption_path, save_format="keras")
            self.decryption_model.save(decryption_path, save_format="keras")
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {str(e)}")

            # Fallback to HDF5 format if keras format fails
            try:
                encryption_path_h5 = os.path.join(models_dir, "encryption_model.h5")
                decryption_path_h5 = os.path.join(models_dir, "decryption_model.h5")

                self.encryption_model.save(encryption_path_h5, save_format="h5")
                self.decryption_model.save(decryption_path_h5, save_format="h5")
                print("Models saved successfully in H5 format")
            except Exception as e2:
                print(f"Error saving models in H5 format: {str(e2)}")

    def reset_models(self):
        """
        Reset the models completely to solve optimizer variable conflicts.
        This creates fresh models with new weights.
        """
        print("Resetting models to solve optimizer variable conflicts")
        self._build_models()

    def train(self, X_train, y_train, validation_data=None, epochs=10, batch_size=32):
        """Train with learning rate schedule."""

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9
        )

        # Recompile encryption model with lr schedule
        print("Recompiling encryption model with fresh optimizer")
        self.encryption_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="mse",
            metrics=["accuracy"],
        )

        # Train encryption model
        enc_history = self.encryption_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[ProgressCallback(self)],
        )

        # Recompile decryption model with fresh optimizer
        print("Recompiling decryption model with fresh optimizer")
        self.decryption_model.compile(
            optimizer="adam",  # This creates a new Adam optimizer instance
            loss="mse",
            metrics=["accuracy"],
        )

        # Train decryption model (could use different data)
        dec_history = self.decryption_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[ProgressCallback(self)],
        )

        # Save models after training
        self.save_models()

        # Return combined history
        return {"encryption": enc_history.history, "decryption": dec_history.history}

    def encrypt_frame(self, frame):
        """
        Encrypt a single frame using the neural network.

        Args:
            frame: Input video frame

        Returns:
            Encrypted frame
        """
        # Preprocess the frame
        processed_frame = self._preprocess_frame_for_network(frame)

        # Apply the model
        encrypted_frame = self.encryption_model.predict(processed_frame)

        # Postprocess the output
        output_frame = self._postprocess_frame_from_network(encrypted_frame[0])

        return output_frame

    def decrypt_frame(self, frame):
        """
        Decrypt a single frame using the neural network.

        Args:
            frame: Encrypted video frame

        Returns:
            Decrypted frame
        """
        # Preprocess the frame
        processed_frame = self._preprocess_frame_for_network(frame)

        # Apply the model
        decrypted_frame = self.decryption_model.predict(processed_frame)

        # Postprocess the output
        output_frame = self._postprocess_frame_from_network(decrypted_frame[0])

        return output_frame

    def _preprocess_frame_for_network(self, frame):
        """
        Preprocess a frame for input to the neural network.

        Args:
            frame: Input video frame

        Returns:
            Preprocessed frame ready for the neural network
        """
        try:
            # Ensure frame is valid
            if frame is None:
                print("Warning: Frame is None, creating empty frame")
                frame = np.zeros(self.input_shape, dtype=np.uint8)

            # Convert to RGB if grayscale
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # If RGBA, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Resize to the expected input shape
            resized_frame = tf.image.resize(frame, self.input_shape[:2])

            # Normalize pixel values
            normalized_frame = resized_frame / 255.0

            # Add batch dimension
            batched_frame = np.expand_dims(normalized_frame, axis=0)

            return batched_frame

        except Exception as e:
            print(f"Error in preprocessing frame: {str(e)}")
            # Return a default frame if error occurs
            default_frame = np.zeros((1,) + self.input_shape)
            return default_frame

    def _postprocess_frame_from_network(self, frame):
        """
        Postprocess a frame from the neural network output.

        Args:
            frame: Output from neural network

        Returns:
            Postprocessed frame ready for display or further processing
        """
        try:
            if frame is None:
                print("Warning: Frame is None in postprocessing")
                return np.zeros(self.input_shape, dtype=np.uint8)

            # Scale back to 0-255 range
            scaled_frame = (frame * 255.0).astype(np.uint8)

            # Resize to match input shape if needed
            if scaled_frame.shape[:2] != self.input_shape[:2]:
                scaled_frame = cv2.resize(
                    scaled_frame, (self.input_shape[1], self.input_shape[0])
                )

            return scaled_frame

        except Exception as e:
            print(f"Error in postprocessing frame: {str(e)}")
            return np.zeros(self.input_shape, dtype=np.uint8)

    def get_input_shape(self):
        """Return the input shape expected by the neural network."""
        return self.input_shape
