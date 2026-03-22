"""
RESPONSIBILITY:
Localizes the pupil boundary and iris outer boundary from the raw grayscale eye image.

CURRENT DESIGN:
1. Pupil Localization:
   Uses median filtering, global inverse thresholding, and contour extraction
   to detect the darkest inner region. The largest contour is approximated by
   a minimum enclosing circle to estimate pupil center and radius.

2. Iris Localization:
   Uses Circular Hough Transform to detect the iris outer boundary.
   Multiple accumulator thresholds are tested in descending sensitivity order.

3. Outer-Circle Selection:
   Among detected circles, selects the one whose center is closest to the
   estimated pupil center.

4. Numerical Stability Fix:
   Hough circle outputs are converted to signed integer type before distance
   calculations, preventing overflow during center-distance comparisons.

NOTES:
- This version keeps the original simple localization logic because larger
  structural modifications reduced recognition performance.
- The retained improvement is the signed-integer conversion that fixes the
  overflow warning without hurting CRR.
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
