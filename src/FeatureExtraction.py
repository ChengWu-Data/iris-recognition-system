"""
RESPONSIBILITY:
1. Filter-based feature extraction using a Region of Interest (ROI) from the upper portion (48x512)[cite: 349].
2. Block-wise statistics: Mean and Average Absolute Deviation for 8x8 blocks[cite: 361, 363].
3. Output feature vector: Generate a 1536-dimensional ordered vector[cite: 362, 370].
"""

import cv2
import numpy as np

def _create_spatial_filter(size: int, delta_x: float, delta_y: float, freq: float) -> np.ndarray:
    """Creates an even-symmetric spatial filter modulated by a symmetric sinusoidal function[cite: 331, 333]."""
    y, x = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    
    gaussian = (1 / (2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * ((x**2 / delta_x**2) + (y**2 / delta_y**2)))
    modulation = np.cos(2 * np.pi * freq * np.sqrt(x**2 + y**2))
    
    kernel = gaussian * modulation
    return kernel - np.mean(kernel) # Zero mean to avoid DC response

def extract_features(enhanced_img: np.ndarray) -> np.ndarray:
    # ROI: Upper 48x512 block (avoids eyelashes) [cite: 349, 357]
    roi = enhanced_img[0:48, :]
    
    # Channel 1 and 2 parameters as defined in the paper [cite: 352]
    filter_1 = _create_spatial_filter(size=15, delta_x=3.0, delta_y=1.5, freq=1/6.0)
    filter_2 = _create_spatial_filter(size=15, delta_x=4.5, delta_y=1.5, freq=1/6.0)
    
    filtered_1 = cv2.filter2D(roi, cv2.CV_32F, filter_1)
    filtered_2 = cv2.filter2D(roi, cv2.CV_32F, filter_2)
    
    features = []
    
    # Block-wise statistics: 8x8 blocks [cite: 361]
    # Total blocks = (48/8) * (512/8) * 2 channels = 6 * 64 * 2 = 768 blocks
    # 2 features per block = 1536 vector length [cite: 362]
    for filtered_img in [filtered_1, filtered_2]:
        for i in range(0, 48, 8):
            for j in range(0, 512, 8):
                block = filtered_img[i:i+8, j:j+8]
                
                # Mean of the absolute magnitude [cite: 365]
                m = np.mean(np.abs(block))
                # Average absolute deviation from the mean [cite: 366]
                sigma = np.mean(np.abs(np.abs(block) - m))
                
                features.extend([m, sigma])
                
    return np.array(features, dtype=np.float32) # length 1536
