"""
RESPONSIBILITY:
1. Illumination correction: Estimate background illumination using 16x16 blocks, expand via bicubic 
   interpolation, and subtract from the normalized image[cite: 286, 287, 288].
2. Contrast enhancement: Perform histogram equalization in each 32x32 region[cite: 289].
"""

import cv2
import numpy as np

def enhance_image(norm_img: np.ndarray) -> np.ndarray:
    """
    Enhance normalized iris image.

    Steps:
    1. Estimate background illumination using 16x16 block means
    2. Upsample background map to full image size
    3. Subtract background
    4. Apply CLAHE for local contrast enhancement
    """
    img = norm_img.astype(np.float32)
    M, N = img.shape

    # 1) Estimate coarse background using 16x16 blocks
    block_h, block_w = 16, 16
    bg_small = np.zeros((max(1, M // block_h), max(1, N // block_w)), dtype=np.float32)

    for i in range(bg_small.shape[0]):
        for j in range(bg_small.shape[1]):
            r0, r1 = i * block_h, min((i + 1) * block_h, M)
            c0, c1 = j * block_w, min((j + 1) * block_w, N)
            bg_small[i, j] = np.mean(img[r0:r1, c0:c1])

    # 2) Resize background map back to full size
    bg = cv2.resize(bg_small, (N, M), interpolation=cv2.INTER_CUBIC)

    # 3) Illumination correction
    corrected = img - bg
    corrected = corrected - corrected.min()
    if corrected.max() > 0:
        corrected = corrected / corrected.max() * 255.0
    corrected = corrected.astype(np.uint8)

    # 4) Local contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)

    return enhanced


def _band_energy(magnitude: np.ndarray, r_low: float, r_high: float) -> float:
    """
    Sum FFT magnitude within an annular frequency band [r_low, r_high).
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    mask = (r >= r_low) & (r < r_high)
    return float(np.sum(magnitude[mask]))


def compute_patch_quality(patch: np.ndarray):
    """
    Compute quality descriptor for one local patch.

    Returns:
        total_power: F1 + F2 + F3
        ratio: F2 / (F1 + F3 + eps)
    """
    patch = patch.astype(np.float32)

    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(patch))
    mag = np.abs(F)

    # Three radial bands following the paper's spirit
    F1 = _band_energy(mag, 0, 6)
    F2 = _band_energy(mag, 6, 22)
    F3 = _band_energy(mag, 22, 32)

    total_power = F1 + F2 + F3
    ratio = F2 / (F1 + F3 + 1e-6)

    return total_power, ratio


def compute_quality(enh_img: np.ndarray):
    """
    Compute image quality using two local iris patches and average them.

    Current design:
    - use the upper 48 rows (aligned with your current stable iris ROI)
    - take two horizontal 64-column patches from the middle-left and middle-right

    Returns:
        avg_total_power, avg_ratio
    """
    h, w = enh_img.shape

    # Keep the vertical region consistent with your current stable feature ROI
    row_start = 0
    row_end = min(48, h)

    # Patch width
    patch_w = 64

    # Two horizontal patches, avoiding extreme borders
    # Adjust these later if needed, but this is a good starting point
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
    """
    Rule-based quality decision.

    NOTE:
    These thresholds are starting points only.
    After running a batch once, print/debug the values and retune if needed.
    """
    total_power, ratio = compute_quality(enh_img)

    if debug:
        print(f"[QUALITY] total_power={total_power:.2f}, ratio={ratio:.4f}")

    # Initial thresholds: you will likely need to tune these
    return (total_power > 3.0e6) and (ratio > 0.95)