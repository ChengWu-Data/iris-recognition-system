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
from typing import Tuple, Dict, Any

def localize_iris(img: np.ndarray, config: Dict[str, Any] = None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    # 1. Coarse localization via projection
    col_sum = np.sum(img, axis=0)
    row_sum = np.sum(img, axis=1)
    cx_approx = int(np.argmin(col_sum))
    cy_approx = int(np.argmin(row_sum))
    
    # 2. Refined pupil localization
    box_size = 60
    y_min, y_max = max(0, cy_approx - box_size), min(img.shape[0], cy_approx + box_size)
    x_min, x_max = max(0, cx_approx - box_size), min(img.shape[1], cx_approx + box_size)
    roi = img[y_min:y_max, x_min:x_max]
    
    # Adaptive thresholding to find pupil
    _, thresh = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback if no pupil contour found to prevent crashing
        cx_pupil, cy_pupil, r_pupil = cx_approx, cy_approx, 30
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            cx_pupil, cy_pupil = cx_approx, cy_approx
        else:
            cx_pupil = int(M["m10"] / M["m00"]) + x_min
            cy_pupil = int(M["m01"] / M["m00"]) + y_min
        
        # Estimate pupil radius
        _, r_pupil_float = cv2.minEnclosingCircle(largest_contour)
        r_pupil = int(r_pupil_float)
        
    pupil_params = (cx_pupil, cy_pupil, r_pupil)
    
    # 3. Outer boundary localization (Canny + Hough)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Restrict Hough circles search to reasonable iris radii (typically 90 to 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=150, param2=30, minRadius=90, maxRadius=150)
                               
    if circles is not None:
        # Choose the circle whose center is closest to the pupil center
        circles_rounded = np.uint16(np.around(circles[0, :]))
        best_circle = min(circles_rounded, key=lambda c: (c[0]-cx_pupil)**2 + (c[1]-cy_pupil)**2)
        iris_params = (int(best_circle[0]), int(best_circle[1]), int(best_circle[2]))
    else:
        # Fallback heuristic if Hough fails
        iris_params = (cx_pupil, cy_pupil, int(r_pupil * 2.5))
        
    return pupil_params, iris_params
