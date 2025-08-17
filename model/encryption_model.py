#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Encryption Model module
----------------------
Handles the core logic for video encryption and decryption.
"""

from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import threading
import cv2
import time


class EncryptionModel(QObject):
    """Model for video encryption and decryption operations."""

    # Signals
    progress_updated = pyqtSignal(int)  # Progress percentage (0-100)
    processing_finished = pyqtSignal(dict)  # Results with metrics
    video_encrypted = pyqtSignal(object)  # Emits encrypted video data
    video_decrypted = pyqtSignal(object)  # Emits decrypted video data

    def __init__(self):
        """Initialize the encryption model."""
        super().__init__()
        self.encrypted_data = None
        self.decrypted_data = None
        self.processing_thread = None
        self.is_processing = False

    def encrypt_video(self, video_data, neural_model):
        """
        Encrypt a video using the neural network model.

        Args:
            video_data: Dictionary containing video frames and metadata
            neural_model: Neural network model for encryption
        """
        if self.is_processing or video_data is None:
            return False

        self.is_processing = True

        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._run_encryption, args=(video_data, neural_model)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

        return True

    def decrypt_video(self, encrypted_data, neural_model):
        """
        Decrypt a video using the neural network model.

        Args:
            encrypted_data: Dictionary containing encrypted video frames and metadata
            neural_model: Neural network model for decryption
        """
        if self.is_processing or encrypted_data is None:
            return False

        self.is_processing = True

        # Start processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._run_decryption, args=(encrypted_data, neural_model)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

        return True

    def _run_encryption(self, video_data, neural_model):
        """
        Run the actual encryption process in a separate thread.

        Args:
            video_data: Dictionary containing video frames and metadata
            neural_model: Neural network model for encryption
        """
        try:
            frames = video_data["frames"]
            total_frames = len(frames)
            encrypted_frames = []

            for i, frame in enumerate(frames):
                # Update progress
                progress = int((i / total_frames) * 100)
                self.progress_updated.emit(progress)

                # Preprocess frame for the neural network
                processed_frame = self._preprocess_frame(frame)

                # Apply neural network encryption
                encrypted_frame = neural_model.encrypt_frame(processed_frame)

                # Apply additional XOR encryption
                # Make sure the key has the same shape as the encrypted frame
                key = self._generate_encryption_key(encrypted_frame.shape)
                xor_encrypted = self._apply_xor_encryption(encrypted_frame, key)

                # Save the encrypted frame
                encrypted_frames.append(xor_encrypted)

                # Simulate processing delay for visualization purposes
                time.sleep(0.01)

            # Store the encrypted data
            self.encrypted_data = {
                "frames": encrypted_frames,
                "fps": video_data["fps"],
                "resolution": video_data["resolution"],
                "frame_count": total_frames,
                "encryption_keys": [
                    self._generate_encryption_key(encrypted_frames[i].shape)
                    for i in range(len(encrypted_frames))
                ],
            }

            # Signal completion
            self.progress_updated.emit(100)
            self.processing_finished.emit({"type": "encryption", "success": True})

            # Emit the encrypted video data
            self.video_encrypted.emit(self.encrypted_data)

        except Exception as e:
            print(f"Error during encryption: {str(e)}")
            self.processing_finished.emit(
                {"type": "encryption", "success": False, "error": str(e)}
            )

        finally:
            self.is_processing = False

    def _run_decryption(self, encrypted_data, neural_model):
        """
        Run the actual decryption process in a separate thread.

        Args:
            encrypted_data: Dictionary containing encrypted video frames and metadata
            neural_model: Neural network model for decryption
        """
        try:
            frames = encrypted_data["frames"]
            keys = encrypted_data.get("encryption_keys", [])
            total_frames = len(frames)
            decrypted_frames = []

            for i, frame in enumerate(frames):
                # Update progress
                progress = int((i / total_frames) * 100)
                self.progress_updated.emit(progress)

                # Apply XOR decryption if keys are available
                if i < len(keys):
                    frame = self._apply_xor_decryption(frame, keys[i])

                # Apply neural network decryption
                decrypted_frame = neural_model.decrypt_frame(frame)

                # Postprocess frame for display
                final_frame = self._postprocess_frame(decrypted_frame)

                # Save the decrypted frame
                decrypted_frames.append(final_frame)

                # Simulate processing delay for visualization purposes
                time.sleep(0.01)

            # Store the decrypted data
            self.decrypted_data = {
                "frames": decrypted_frames,
                "fps": encrypted_data["fps"],
                "resolution": encrypted_data["resolution"],
                "frame_count": total_frames,
            }

            # Signal completion
            self.progress_updated.emit(100)
            self.processing_finished.emit({"type": "decryption", "success": True})

            # Emit the decrypted video data
            self.video_decrypted.emit(self.decrypted_data)

        except Exception as e:
            print(f"Error during decryption: {str(e)}")
            self.processing_finished.emit(
                {"type": "decryption", "success": False, "error": str(e)}
            )

        finally:
            self.is_processing = False

    def get_encrypted_data(self):
        """Return the encrypted video data."""
        return self.encrypted_data

    def get_decrypted_data(self):
        """Return the decrypted video data."""
        return self.decrypted_data

    def _preprocess_frame(self, frame):
        """
        Preprocess a video frame for the neural network.

        Args:
            frame: Input video frame

        Returns:
            Preprocessed frame
        """
        # Resize to the neural network input size if needed
        # normalized_frame = frame.astype('float32') / 255.0
        # For this example, we'll just return the frame as is
        return frame

    def _postprocess_frame(self, frame):
        """
        Postprocess a frame after neural network processing.

        Args:
            frame: Processed frame from neural network

        Returns:
            Postprocessed frame ready for display
        """
        # Convert back to uint8 if needed
        output_frame = (frame * 255.0).astype("uint8")
        # For this example, we'll just return the frame as is
        return frame

    def _generate_encryption_key(self, shape):
        """
        Generate a random encryption key for XOR encryption.

        Args:
            shape: Shape of the frame

        Returns:
            Random encryption key
        """
        return np.random.randint(0, 256, shape, dtype=np.uint8)

    def _apply_xor_encryption(self, frame, key):
        """
        Apply XOR encryption to a frame.

        Args:
            frame: Input frame
            key: Encryption key

        Returns:
            XOR encrypted frame
        """
        return cv2.bitwise_xor(frame, key)

    def _apply_xor_decryption(self, frame, key):
        """
        Apply XOR decryption to a frame (same as encryption since XOR is symmetric).

        Args:
            frame: Encrypted frame
            key: Encryption key used for encryption

        Returns:
            XOR decrypted frame
        """
        return cv2.bitwise_xor(frame, key)
