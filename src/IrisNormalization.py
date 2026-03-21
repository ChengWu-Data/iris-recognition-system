"""
LOGIC:
Transforms the circular iris region into a fixed-size 64x512 rectangle.
Unlike simplified models, this implements the non-concentric mapping:
I(r, theta) = (1-r)*P(theta) + r*I(theta)
where P is a point on the pupil boundary and I is on the iris boundary.

KEY VARIABLES:
- radial_res (64): Number of samples along the radial direction.
- angular_res (512): Number of samples around the circumference.
"""

import cv2
import numpy as np

def normalize_iris(img: np.ndarray, pupil_params: tuple, iris_params: tuple, cfg: dict) -> np.ndarray:
    radial_res = cfg.get('radial_res', 64)
    angular_res = cfg.get('angular_res', 512)
    
    xp, yp, rp = pupil_params
    xi, yi, ri = iris_params
    
    norm_img = np.zeros((radial_res, angular_res), dtype=np.uint8)
    theta = np.linspace(0, 2 * np.pi, angular_res)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    for j in range(angular_res):
        x_p = xp + rp * cos_t[j]
        y_p = yp + rp * sin_t[j]
        
        x_i = xi + ri * cos_t[j]
        y_i = yi + ri * sin_t[j]
        
        for i in range(radial_res):
            r = i / radial_res
            curr_x = (1 - r) * x_p + r * x_i
            curr_y = (1 - r) * y_p + r * y_i
            
            if 0 <= curr_x < img.shape[1]-1 and 0 <= curr_y < img.shape[0]-1:
                norm_img[i, j] = img[int(curr_y), int(curr_x)]
                
    return norm_img
