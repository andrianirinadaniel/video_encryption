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

    def calculate_metrics(self, original_frames, comparison_frames):
        """
        Calculate quality metrics between original and comparison frames.

        Args:
            original_frames: List of original video frames
            comparison_frames: List of frames to compare against original

        Returns:
            Dictionary containing various metrics
        """
        # Sample frames if there are too many
        max_frames = 30  # Limit calculation to reasonable number

        if len(original_frames) > max_frames:
            step = len(original_frames) // max_frames
            frame_indices = list(range(0, len(original_frames), step))[:max_frames]
        else:
            frame_indices = range(len(original_frames))

        # Initialize metrics
        metrics = {
            "psnr": 0.0,
            "ssim": 0.0,
            "entropy": 0.0,
            "correlation": 0.0,
            "vif": 0.0,
        }

        # Calculate metrics across sampled frames
        psnr_values = []
        ssim_values = []
        entropy_orig_values = []
        entropy_comp_values = []
        correlation_values = []
        vif_values = []

        for idx in frame_indices:
            if idx < len(original_frames) and idx < len(comparison_frames):
                orig = original_frames[idx]
                comp = comparison_frames[idx]

                # Ensure same size
                if orig.shape != comp.shape:
                    comp = cv2.resize(comp, (orig.shape[1], orig.shape[0]))

                # Convert to grayscale for some metrics
                orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)

                # Calculate PSNR
                try:
                    psnr = peak_signal_noise_ratio(orig, comp)
                    psnr_values.append(psnr)
                except Exception:
                    pass

                # Calculate SSIM
                try:
                    ssim = structural_similarity(orig_gray, comp_gray, full=True)[0]
                    ssim_values.append(ssim)
                except Exception:
                    pass

                # Calculate entropy - FIX: Changed from self.calculate_entropy to self._calculate_entropy
                entropy_orig = self._calculate_entropy(orig_gray)
                entropy_comp = self._calculate_entropy(comp_gray)
                entropy_orig_values.append(entropy_orig)
                entropy_comp_values.append(entropy_comp)

                # Calculate correlation
                try:
                    correlation = np.corrcoef(orig_gray.flatten(), comp_gray.flatten())[
                        0, 1
                    ]
                    correlation_values.append(correlation)
                except Exception:
                    pass

                # VIF calculation (simplified) - FIX: Changed from self.calculate_vif to self._calculate_simplified_vif
                try:
                    vif = self._calculate_simplified_vif(orig_gray, comp_gray)
                    vif_values.append(vif)
                except Exception:
                    pass

        # Average the metrics
        if psnr_values:
            metrics["psnr"] = sum(psnr_values) / len(psnr_values)
        if ssim_values:
            metrics["ssim"] = sum(ssim_values) / len(ssim_values)
        if entropy_comp_values:
            metrics["entropy"] = sum(entropy_comp_values) / len(entropy_comp_values)
        if correlation_values:
            metrics["correlation"] = sum(correlation_values) / len(correlation_values)
        if vif_values:
            metrics["vif"] = sum(vif_values) / len(vif_values)

        return metrics

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
