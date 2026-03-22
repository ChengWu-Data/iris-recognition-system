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

"""
LOGIC:
1. Pupil Localization: Uses global thresholding and morphological cleaning to find 
   the darkest region. minEnclosingCircle provides a robust center (xp, yp).
2. Iris Localization: Applies the Circular Hough Transform within a constrained 
   search space relative to the pupil center. Crucially, it allows the iris 
   center (xi, yi) to differ from (xp, yp), capturing the natural eccentricity.

KEY VARIABLES:
- p2: The accumulator threshold for Hough Circles. Lower values are more sensitive.
- minRadius/maxRadius: Constants (90-125) calibrated for CASIA-IrisV1 database.
"""

import cv2
import numpy as np
from typing import Tuple

def localize_iris(img: np.ndarray, config=None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    blurred = cv2.medianBlur(img, 5)
    
    # Inner Boundary
    _, thresh = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        xp, yp, rp = 160, 120, 45
    else:
        pupil_c = max(contours, key=cv2.contourArea)
        (cx_f, cy_f), r_f = cv2.minEnclosingCircle(pupil_c)
        xp, yp, rp = int(cx_f), int(cy_f), int(r_f)
    
    pupil_params = (xp, yp, rp)

    # Outer Boundary
    iris_params = (xp, yp, int(rp * 2.8)) 
    
    for p2 in [30, 25, 20, 15]: 
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=100, param2=p2, minRadius=90, maxRadius=130
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            best = min(circles, key=lambda c: (c[0]-xp)**2 + (c[1]-yp)**2)
            iris_params = (int(best[0]), int(best[1]), int(best[2]))
            break
            
    return pupil_params, iris_params
