"""
RESPONSIBILITY:
Detects the inner boundary (pupil) and outer boundary (iris) of the eye.

INPUT/OUTPUT:
- Input: Raw eye image (grayscale, 320x280).
- Output: Tuple of pupil parameters (x, y, r) and iris parameters (x, y, r).

METHODS TO IMPLEMENT:
1. Coarse pupil center estimation via horizontal/vertical projections.
2. Exact pupil boundary via thresholding and centroid calculation.
3. Outer boundary via edge detection (Canny) and Hough Transform.
"""

import cv2
import numpy as np
from typing import Tuple

def localize_iris(img: np.ndarray, config=None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    # Preprocessing
    blurred = cv2.medianBlur(img, 5)
    
    # 1. Pupil Localization (Darkest region)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (160, 120, 40), (160, 120, 100) # Fallback
    
    pupil_c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(pupil_c)
    pupil_params = (int(cx), int(cy), int(r))

    # 2. Iris Localization (Outer Boundary)
    # Strategy: Try different sensitivity (param2) to find the outer circle
    iris_params = (int(cx), int(cy), int(r * 2.8)) # Default guess
    for p2 in [30, 20, 15]: # Retry from strict to loose
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=100, param2=p2, minRadius=90, maxRadius=125
        )
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            # Find circle closest to pupil center
            best = min(circles, key=lambda c: (c[0]-cx)**2 + (c[1]-cy)**2)
            iris_params = (int(best[0]), int(best[1]), int(best[2]))
            break
            
    return pupil_params, iris_params
