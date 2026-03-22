"""
LOGIC:
1. ROI Selection: Focuses on the 10:58 rows of the normalized image to avoid 
   eyelash and eyelid occlusions at the top.
2. Multichannel Filtering: Uses two Gabor-like symmetric filters to capture 
   texture at different scales.
3. Feature Encoding: Calculates Mean and Mean Absolute Deviation (MAD) for 
   8x8 blocks to form a 1536-D vector.
"""

import cv2
import numpy as np

def _create_spatial_filter(size: int, delta_x: float, delta_y: float, freq: float) -> np.ndarray:
    y, x = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    gaussian = (1 / (2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * ((x**2 / delta_x**2) + (y**2 / delta_y**2)))
    modulation = np.cos(2 * np.pi * freq * np.sqrt(x**2 + y**2))
    kernel = gaussian * modulation
    return kernel - np.mean(kernel)

def extract_features(enhanced_img: np.ndarray, config=None) -> np.ndarray:
    roi = enhanced_img[0:48, :]

    filter_1 = _create_spatial_filter(15, 3.0, 1.2, 1/10.0)
    filter_2 = _create_spatial_filter(15, 4.5, 1.8, 1/5.5)

    f1 = cv2.filter2D(roi, cv2.CV_32F, filter_1)
    f2 = cv2.filter2D(roi, cv2.CV_32F, filter_2)

    features = []

    for i in range(0, roi.shape[0], 8):
        for j in range(0, roi.shape[1], 8):
            for block in [f1[i:i+8, j:j+8], f2[i:i+8, j:j+8]]:
                mag = np.abs(block)
                mean = np.mean(mag)
                mad = np.mean(np.abs(mag - mean))
                features.extend([mean, mad])

    features = np.array(features)

    # Normalize
    norm = np.linalg.norm(features) + 1e-6
    features = features / norm

    return features

def extract_templates(enh_img):
    angles = [-9, -6, -3, 0, 3, 6, 9]
    features = []

    for a in angles:
        shift = int(a / 360.0 * enh_img.shape[1])
        rotated = np.roll(enh_img, shift, axis=1)
        feat = extract_features(rotated)
        features.append(feat)

    return features