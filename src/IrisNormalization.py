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

    x_p = xp + rp * cos_t
    y_p = yp + rp * sin_t
    x_i = xi + ri * cos_t
    y_i = yi + ri * sin_t

    for j in range(angular_res):
        for i in range(radial_res):
            r = i / radial_res

            curr_x = (1 - r) * x_p[j] + r * x_i[j]
            curr_y = (1 - r) * y_p[j] + r * y_i[j]
            
            if 0 <= curr_x < img.shape[1]-1 and 0 <= curr_y < img.shape[0]-1:

                norm_img[i, j] = img[int(curr_y), int(curr_x)]
                
    return norm_img
