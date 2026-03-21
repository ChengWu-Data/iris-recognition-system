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
