"""
RESPONSIBILITY:
Enhances normalized iris images and computes frequency-based quality scores for sample ranking.

CURRENT DESIGN:
1. Illumination Correction: 
   Estimates coarse background illumination using 16x16 block means 
   and reconstructs the background map via bicubic interpolation.

2. Local Contrast Enhancement:
   Applies CLAHE after illumination correction to improve local iris texture visibility.

3. Quality Measurement:
   Evaluates image quality in the frequency domain using two local iris patches.
   For each patch, the FFT magnitude is summarized into three radial bands:
   - low frequency
   - middle frequency
   - high frequency

4. Quality Descriptors:
   Returns:
   - total_power = F1 + F2 + F3
   - ratio = F2 / (F1 + F3 + eps)

NOTES:
- Quality ranking is more important than hard thresholding in the current system.
- The current feature pipeline uses quality selection primarily for sample ranking.
"""

import cv2
import numpy as np

def enhance_image(norm_img: np.ndarray) -> np.ndarray:
    img = norm_img.astype(np.float32)
    M, N = img.shape

    block_h, block_w = 16, 16
    bg_small = np.zeros((max(1, M // block_h), max(1, N // block_w)), dtype=np.float32)

    for i in range(bg_small.shape[0]):
        for j in range(bg_small.shape[1]):
            r0, r1 = i * block_h, min((i + 1) * block_h, M)
            c0, c1 = j * block_w, min((j + 1) * block_w, N)
            bg_small[i, j] = np.mean(img[r0:r1, c0:c1])

    bg = cv2.resize(bg_small, (N, M), interpolation=cv2.INTER_CUBIC)

    corrected = img - bg
    corrected = corrected - corrected.min()
    if corrected.max() > 0:
        corrected = corrected / corrected.max() * 255.0
    corrected = corrected.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    return enhanced


def _band_energy(magnitude: np.ndarray, r_low: float, r_high: float) -> float:
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    mask = (r >= r_low) & (r < r_high)
    return float(np.sum(magnitude[mask]))


def compute_patch_quality(patch: np.ndarray):
    patch = patch.astype(np.float32)

    F = np.fft.fftshift(np.fft.fft2(patch))
    mag = np.abs(F)

    F1 = _band_energy(mag, 0, 6)
    F2 = _band_energy(mag, 6, 22)
    F3 = _band_energy(mag, 22, 32)

    total_power = F1 + F2 + F3
    ratio = F2 / (F1 + F3 + 1e-6)

    return total_power, ratio


def compute_quality(enh_img: np.ndarray):
    h, w = enh_img.shape

    row_start = 0
    row_end = min(48, h)

    patch_w = 64

    c1_start = max(32, w // 4 - patch_w // 2)
    c2_start = min(w - patch_w - 32, 3 * w // 4 - patch_w // 2)

    patch1 = enh_img[row_start:row_end, c1_start:c1_start + patch_w]
    patch2 = enh_img[row_start:row_end, c2_start:c2_start + patch_w]

    q1_total, q1_ratio = compute_patch_quality(patch1)
    q2_total, q2_ratio = compute_patch_quality(patch2)

    avg_total = (q1_total + q2_total) / 2.0
    avg_ratio = (q1_ratio + q2_ratio) / 2.0

    return avg_total, avg_ratio


def is_good_quality(enh_img: np.ndarray, debug: bool = False) -> bool:
    total_power, ratio = compute_quality(enh_img)

    if debug:
        print(f"[QUALITY] total_power={total_power:.2f}, ratio={ratio:.4f}")

    # Initial thresholds: you will likely need to tune these
    return (total_power > 3.0e6) and (ratio > 0.95)