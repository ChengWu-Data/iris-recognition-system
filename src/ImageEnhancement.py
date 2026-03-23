"""
ImageEnhancement.py

Purpose:
    This script enhances the normalized iris image before feature extraction.
    Its main goal is to reduce the effect of nonuniform illumination and improve
    the visibility of iris texture patterns.

Overall Logic:
    1. Estimate coarse background illumination using 16x16 block means.
    2. Upsample the coarse background estimate using bicubic interpolation.
    3. Subtract the estimated background from the normalized image.
    4. Normalize the corrected image to the 0-255 range.
    5. Apply local histogram equalization to each 32x32 region.

Why this script is needed:
    Even after normalization, the iris image may still have uneven brightness
    caused by illumination conditions or imaging artifacts. If these brightness
    variations are not corrected, the later feature extraction stage may capture
    lighting patterns instead of true iris texture.

Key Variables and Parameters:
    norm_img:
        The normalized iris image used as input.
    M, N:
        The height and width of the normalized image.
    bh, bw:
        The height and width of each block used for background estimation.
    bg_small:
        A coarse low-resolution estimate of the background illumination.
    bg:
        The full-size background image obtained after bicubic interpolation.
    corrected:
        The image after subtracting background illumination.
    out:
        The final enhanced image after local histogram equalization.
    win_h, win_w:
        The height and width of each local window used for histogram equalization.
"""

import cv2
import numpy as np


def enhance_image(norm_img: np.ndarray) -> np.ndarray:
    """
    Enhance the normalized iris image using illumination correction
    and local histogram equalization.

    Main Steps:
        1. Convert the input image to float for stable numerical processing.
        2. Estimate background illumination using the mean intensity of each 16x16 block.
        3. Resize the low-resolution background map to the original image size.
        4. Subtract the background from the image to compensate for uneven illumination.
        5. Rescale the corrected image into the 0-255 range.
        6. Apply histogram equalization locally on each 32x32 block.

    Args:
        norm_img:
            A normalized grayscale iris image.

    Returns:
        An enhanced grayscale image with improved local contrast.

    Key Variables:
        img:
            Floating-point copy of the normalized image.
        M, N:
            Height and width of the image.
        bh, bw:
            Dimensions of the blocks used for background estimation.
        bg_small_h, bg_small_w:
            Number of background blocks along height and width.
        bg_small:
            Coarse background illumination map.
        bg:
            Upsampled full-size background illumination estimate.
        corrected:
            Image after illumination correction.
        out:
            Final enhanced image after local histogram equalization.
        win_h, win_w:
            Block size used for local histogram equalization.
    """
    # Convert the normalized image to float for illumination correction
    img = norm_img.astype(np.float32)
    M, N = img.shape

    # Step 1: Estimate coarse background illumination using 16x16 block means
    bh, bw = 16, 16
    bg_small_h = int(np.ceil(M / bh))
    bg_small_w = int(np.ceil(N / bw))
    bg_small = np.zeros((bg_small_h, bg_small_w), dtype=np.float32)

    for i in range(bg_small_h):
        for j in range(bg_small_w):
            # Determine the current block boundaries
            r0, r1 = i * bh, min((i + 1) * bh, M)
            c0, c1 = j * bw, min((j + 1) * bw, N)

            # Use the block mean as the coarse illumination estimate
            bg_small[i, j] = np.mean(img[r0:r1, c0:c1])

    # Step 2: Upsample the coarse background estimate to full image size
    bg = cv2.resize(bg_small, (N, M), interpolation=cv2.INTER_CUBIC)

    # Step 3: Subtract background illumination
    corrected = img - bg

    # Shift values to start from zero
    corrected -= corrected.min()

    # Rescale to standard grayscale range
    if corrected.max() > 0:
        corrected = corrected / corrected.max() * 255.0
    corrected = corrected.astype(np.uint8)

  
    # Step 4: Apply local histogram equalization on each 32x32 block
    out = np.zeros_like(corrected)
    win_h, win_w = 32, 32

    for i in range(0, M, win_h):
        for j in range(0, N, win_w):
            # Extract the current local region
            block = corrected[i:i + win_h, j:j + win_w]

            # Enhance local contrast within this block
            out[i:i + win_h, j:j + win_w] = cv2.equalizeHist(block)

    return out