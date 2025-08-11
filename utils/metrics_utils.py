#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics Utilities module
----------------------
Helper functions for calculating image and video quality metrics.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def calculate_psnr(original, processed):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Original image (numpy array)
        processed: Processed image (numpy array)

    Returns:
        PSNR value in dB
    """
    try:
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original

        if len(processed.shape) == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed

        # Calculate PSNR
        return peak_signal_noise_ratio(original_gray, processed_gray)
    except Exception as e:
        print(f"Error calculating PSNR: {str(e)}")
        return 0.0


def calculate_ssim(original, processed):
    """
    Calculate Structural Similarity Index between two images.

    Args:
        original: Original image (numpy array)
        processed: Processed image (numpy array)

    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    try:
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original

        if len(processed.shape) == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed

        # Calculate SSIM
        return structural_similarity(original_gray, processed_gray)
    except Exception as e:
        print(f"Error calculating SSIM: {str(e)}")
        return 0.0


def calculate_entropy(image):
    """
    Calculate Shannon entropy of an image.

    Args:
        image: Input image (numpy array)

    Returns:
        Entropy value (higher means more information/randomness)
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()

        # Remove zeros (to avoid log(0))
        hist = hist[hist > 0]

        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    except Exception as e:
        print(f"Error calculating entropy: {str(e)}")
        return 0.0


def calculate_correlation(image1, image2):
    """
    Calculate correlation coefficient between two images.

    Args:
        image1: First image (numpy array)
        image2: Second image (numpy array)

    Returns:
        Correlation coefficient (between -1 and 1)
    """
    try:
        # Ensure images are the same size
        if image1.shape != image2.shape:
            return 0.0

        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        if len(image2.shape) == 3:
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image2_gray = image2

        # Flatten arrays
        x = image1_gray.flatten()
        y = image2_gray.flatten()

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        # Handle NaN values
        if np.isnan(correlation):
            return 0.0

        return correlation
    except Exception as e:
        print(f"Error calculating correlation: {str(e)}")
        return 0.0


def calculate_simplified_vif(image1, image2):
    """
    Calculate a simplified Visual Information Fidelity metric.
    Note: This is not the actual VIF but a simplified approximation.

    Args:
        image1: First image (numpy array)
        image2: Second image (numpy array)

    Returns:
        Simplified VIF value
    """
    try:
        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            image1_gray = image1

        if len(image2.shape) == 3:
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            image2_gray = image2

        # Calculate SSIM and correlation
        ssim = calculate_ssim(image1_gray, image2_gray)
        corr = calculate_correlation(image1_gray, image2_gray)

        # Calculate a simple approximation of VIF
        # Real VIF would require wavelet decomposition and statistical modeling
        return (ssim * corr) / (ssim + corr + 1e-10)
    except Exception as e:
        print(f"Error calculating VIF: {str(e)}")
        return 0.0


def calculate_all_metrics(original, processed):
    """
    Calculate all image quality metrics at once.

    Args:
        original: Original image (numpy array)
        processed: Processed image (numpy array)

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "psnr": calculate_psnr(original, processed),
        "ssim": calculate_ssim(original, processed),
        "entropy": calculate_entropy(processed),
        "correlation": calculate_correlation(original, processed),
        "vif": calculate_simplified_vif(original, processed),
    }

    return metrics
