"""
FeatureExtraction.py

Purpose:
    This script extracts a fixed-length iris feature vector from an enhanced
    normalized iris image. The extracted feature vector is later used for
    iris matching and classification.

Overall Logic:
    1. Take a fixed region of interest (ROI) from the enhanced normalized iris image.
    2. Construct two circularly symmetric spatial filters.
    3. Apply both filters to the ROI to capture texture information at different scales.
    4. Divide each filtered response into non-overlapping 8x8 blocks.
    5. For each block, compute two statistical features:
       - mean
       - average absolute deviation (AAD)
    6. Concatenate all block-level statistics into one 1536-dimensional feature vector.
    7. Apply final L2 normalization to reduce global amplitude variation.

Why this script is needed:
    The iris image itself cannot be used directly for classification because it is too large
    and too sensitive to local variation. This script converts the image into a compact,
    structured, and fixed-length representation that captures local texture patterns.

Key Variables and Parameters:
    size:
        The kernel size of the spatial filter.
    delta_x, delta_y:
        Parameters controlling the horizontal and vertical spread of the Gaussian envelope.
    freq:
        The frequency of the cosine modulation used in the spatial filter.
    enhanced_img:
        The enhanced normalized iris image used as input.
    roi:
        The selected 48x512 region used for feature extraction.
    filter_1, filter_2:
        Two spatial filters with different parameter settings.
    f1, f2:
        Filter responses after applying the two filters to the ROI.
    block:
        An 8x8 local patch extracted from a filtered response.
    mean:
        The average magnitude of the current block.
    aad:
        The average absolute deviation of the current block relative to its mean.
"""

from __future__ import annotations

import cv2
import numpy as np


def _create_spatial_filter(size: int, delta_x: float, delta_y: float, freq: float) -> np.ndarray:
    """
    Create a circularly symmetric spatial filter.

    Logic:
        1. Build a centered 2D coordinate grid.
        2. Compute a Gaussian envelope to localize the filter in space.
        3. Apply a cosine modulation based on radial distance from the center.
        4. Subtract the mean so the filter becomes zero-mean.
        5. Normalize the filter by the sum of absolute values for numerical stability.

    Args:
        size:
            The side length of the square filter kernel.
        delta_x:
            Controls the horizontal spread of the Gaussian envelope.
        delta_y:
            Controls the vertical spread of the Gaussian envelope.
        freq:
            Controls the frequency of the cosine modulation.

    Returns:
        A 2D NumPy array representing the spatial filter.

    Key Variables:
        coords:
            A 1D coordinate array centered at zero.
        x, y:
            2D coordinate matrices used to compute the filter.
        gaussian:
            The Gaussian envelope that controls spatial locality.
        modulation:
            The cosine term that controls texture selectivity.
        kernel:
            The final filter after combining Gaussian and cosine components.
    """
    # Create a centered coordinate grid for the filter kernel
    coords = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    y, x = np.meshgrid(coords, coords, indexing="ij")

    # Gaussian envelope controls the spatial spread of the filter
    gaussian = np.exp(-0.5 * ((x ** 2) / (delta_x ** 2) + (y ** 2) / (delta_y ** 2)))

    # Radial cosine modulation captures local texture patterns at a given frequency
    modulation = np.cos(2.0 * np.pi * freq * np.sqrt(x ** 2 + y ** 2))

    # Combine Gaussian and cosine terms
    kernel = gaussian * modulation

    # Force the filter to have zero mean
    kernel -= np.mean(kernel)

    # Normalize the filter for stable numerical behavior
    kernel /= (np.sum(np.abs(kernel)) + 1e-8)

    return kernel.astype(np.float32)


def extract_features(enhanced_img: np.ndarray, config=None) -> np.ndarray:
    """
    Extract a 1536-dimensional feature vector from the enhanced normalized iris image.

    Main Steps:
        1. Check whether the input image has a valid size and shape.
        2. Select the upper 48x512 region of interest (ROI).
        3. Construct two spatial filters.
        4. Apply both filters to the ROI.
        5. Divide each filtered image into 8x8 blocks.
        6. For each block, compute:
           - mean of the absolute response
           - average absolute deviation (AAD)
        7. Concatenate all block statistics into a 1536D vector.
        8. Apply L2 normalization to the final feature vector.

    Args:
        enhanced_img:
            A 2D enhanced grayscale normalized iris image.
        config:
            Optional configuration input. It is not used in the current implementation.

    Returns:
        A 1536-dimensional NumPy feature vector.

    Why the output dimension is 1536:
        - ROI size is 48x512
        - Each block is 8x8
        - Number of blocks per filtered image:
              (48 / 8) * (512 / 8) = 6 * 64 = 384
        - Two filtered images are used
        - Two features are extracted from each block (mean and AAD)

        Therefore:
              384 * 2 * 2 = 1536

    Key Variables:
        roi:
            The selected 48x512 region used for feature extraction.
        filter_1, filter_2:
            Two spatial filters with different scales.
        f1, f2:
            The filtered outputs of the ROI.
        features:
            A list that stores all block-level statistics before conversion to NumPy format.
        i, j:
            Loop indices used to scan the ROI block by block.
        filtered:
            Refers to one of the two filtered outputs inside the loop.
        block:
            The current 8x8 local patch.
        mean:
            The mean magnitude of the current block.
        aad:
            The average absolute deviation of the current block.
    """
    # Ensure that the input image is a valid 2D grayscale image
    if enhanced_img.ndim != 2:
        raise ValueError("enhanced_img must be a 2D grayscale image")

    # Ensure that the image is large enough for the expected ROI
    if enhanced_img.shape[0] < 48 or enhanced_img.shape[1] < 512:
        raise ValueError(f"Expected at least (48, 512), got {enhanced_img.shape}")

    # Use the upper 48x512 region as the region of interest
    roi = enhanced_img[:48, :512].astype(np.float32)

    # Construct two spatial filters with different parameter settings
    filter_1 = _create_spatial_filter(15, 3.0, 1.5, 1 / 10.0)
    filter_2 = _create_spatial_filter(15, 4.5, 1.5, 1 / 5.5)

    # Apply the filters to the ROI
    f1 = cv2.filter2D(roi, cv2.CV_32F, filter_1, borderType=cv2.BORDER_REFLECT)
    f2 = cv2.filter2D(roi, cv2.CV_32F, filter_2, borderType=cv2.BORDER_REFLECT)

    # Store all extracted block-level statistics here
    features = []

    # Divide the filtered images into non-overlapping 8x8 blocks
    for i in range(0, 48, 8):
        for j in range(0, 512, 8):
            # Process both filtered responses at the same block location
            for filtered in (f1, f2):
                # Use the magnitude of the filter response
                block = np.abs(filtered[i:i + 8, j:j + 8])

                # Compute local statistics for the block
                mean = float(np.mean(block))
                aad = float(np.mean(np.abs(block - mean)))

                # Add both statistics to the feature list
                features.extend([mean, aad])

    # Convert to NumPy array
    features = np.asarray(features, dtype=np.float32)

    # Check final feature dimension
    if features.shape[0] != 1536:
        raise ValueError(f"Feature dimension must be 1536, got {features.shape[0]}")

    # Normalize the feature vector to reduce global amplitude variation
    features /= (np.linalg.norm(features) + 1e-6)

    return features
