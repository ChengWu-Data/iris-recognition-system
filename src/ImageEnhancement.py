"""
RESPONSIBILITY:
1. Illumination correction: Estimate background illumination using 16x16 blocks, expand via bicubic 
   interpolation, and subtract from the normalized image[cite: 286, 287, 288].
2. Contrast enhancement: Perform histogram equalization in each 32x32 region[cite: 289].
"""

import cv2
import numpy as np

def enhance_image(norm_img: np.ndarray) -> np.ndarray:
    M, N = norm_img.shape
    
    # 1. Illumination correction
    # Coarse estimate of background using 16x16 block means [cite: 286]
    coarse_bg = np.zeros((M // 16, N // 16), dtype=np.float32)
    for i in range(M // 16):
        for j in range(N // 16):
            block = norm_img[i*16:(i+1)*16, j*16:(j+1)*16]
            coarse_bg[i, j] = np.mean(block)
            
    # Expand via bicubic interpolation [cite: 287]
    bg_illumination = cv2.resize(coarse_bg, (N, M), interpolation=cv2.INTER_CUBIC)
    
    # Subtract to compensate for lighting [cite: 288]
    corrected_img = cv2.normalize(
        norm_img.astype(np.float32) - bg_illumination,
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    # 2. Contrast enhancement
    # Histogram equalization in 32x32 regions (CLAHE is a modern standard equivalent) [cite: 289]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(M//32, N//32))
    enhanced_img = clahe.apply(corrected_img)
    
    return enhanced_img

def compute_quality(enh_img):
    # use ROI (same as feature extraction)
    roi = enh_img[0:48, :]

    # FFT
    F = np.fft.fftshift(np.fft.fft2(roi))
    magnitude = np.abs(F)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy)**2 + (X - cx)**2)

    # frequency bands (paper: (0,6), (6,22), (22,32))
    F1 = magnitude[(r >= 0) & (r < 6)].sum()
    F2 = magnitude[(r >= 6) & (r < 22)].sum()
    F3 = magnitude[(r >= 22) & (r < 32)].sum()

    total_power = F1 + F2 + F3
    ratio = F2 / (F1 + F3 + 1e-6)

    return total_power, ratio

def is_good_quality(enh_img):
    total_power, ratio = compute_quality(enh_img)

    # thresholds
    return (total_power > 1e6) and (ratio > 0.7)