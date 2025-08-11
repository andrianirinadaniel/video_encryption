#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Controller module
------------------------
Handles calculation and management of video quality metrics.
"""

from PyQt5.QtCore import QObject
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class MetricsController(QObject):
    """Controller for calculating and managing video quality metrics."""

    def __init__(self):
        """Initialize the metrics controller."""
        super().__init__()
        self.current_metrics = {
            "psnr": 0.0,
            "ssim": 0.0,
            "entropy": 0.0,
            "correlation": 0.0,
            "vif": 0.0,
        }

    def calculate_metrics(self, original_frame, processed_frame):
        """
        Calculate quality metrics between original and processed frames.

        Args:
            original_frame: Original video frame
            processed_frame: Encrypted or decrypted frame

        Returns:
            Dictionary containing calculated metrics
        """
        # Convert frames to grayscale for metrics calculation
        if len(original_frame.shape) == 3:
            orig_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_frame

        if len(processed_frame.shape) == 3:
            proc_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        else:
            proc_gray = processed_frame

        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr = peak_signal_noise_ratio(orig_gray, proc_gray)
        except Exception:
            psnr = 0.0

        # Calculate SSIM (Structural Similarity Index)
        try:
            ssim = structural_similarity(orig_gray, proc_gray)
        except Exception:
            ssim = 0.0

        # Calculate Entropy
        entropy = self._calculate_entropy(proc_gray)

        # Calculate Correlation Coefficient
        correlation = self._calculate_correlation(orig_gray, proc_gray)

        # Calculate VIF (Visual Information Fidelity)
        # This is a simplified version, real VIF requires more complex calculations
        vif = self._calculate_simplified_vif(orig_gray, proc_gray)

        # Update and return metrics
        self.current_metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "entropy": entropy,
            "correlation": correlation,
            "vif": vif,
        }

        return self.current_metrics

    def get_current_metrics(self):
        """Return the currently calculated metrics."""
        return self.current_metrics

    def _calculate_entropy(self, image):
        """Calculate Shannon entropy of an image."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _calculate_correlation(self, image1, image2):
        """Calculate correlation coefficient between two images."""
        if image1.shape != image2.shape:
            return 0.0

        # Flatten arrays
        x = image1.flatten()
        y = image2.flatten()

        # Calculate correlation coefficient
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            if np.isnan(correlation):
                return 0.0
            return correlation
        except Exception:
            return 0.0

    def _calculate_simplified_vif(self, image1, image2):
        """
        Calculate simplified Visual Information Fidelity.
        Note: This is not the actual VIF but a simplified approximation.
        """
        # For simplicity, using a combination of SSIM and correlation
        # Real VIF would require wavelet decomposition and statistical modeling
        try:
            ssim = structural_similarity(image1, image2)
            corr = self._calculate_correlation(image1, image2)
            # Simple approximation, real VIF is more complex
            return (ssim * corr) / (ssim + corr + 1e-10)
        except Exception:
            return 0.0
