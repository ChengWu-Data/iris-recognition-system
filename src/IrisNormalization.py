import cv2
import numpy as np

def normalize_iris(img, pupil_params, iris_params, cfg):
    """
    Implements Daugman's Rubber Sheet Model to transform the circular 
    iris region into a fixed-size rectangular block.
    
    Args:
        img: Input grayscale image.
        pupil_params: (xp, yp, rp) - pupil center and radius.
        iris_params: (xi, yi, ri) - iris center and radius.
        cfg: Configuration dictionary (contains 'radial_res' and 'angular_res').
        
    Returns:
        norm_img: Normalized rectangular iris image.
    """
    radial_res = cfg.get('radial_res', 64)   # Default: 64
    angular_res = cfg.get('angular_res', 512) # Default: 512
    
    xp, yp, rp = pupil_params
    xi, yi, ri = iris_params
    
    # Create an empty array for the normalized image
    norm_img = np.zeros((radial_res, angular_res), dtype=np.uint8)
    
    # Pre-calculate angles (theta) from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, angular_res)
    
    for j in range(angular_res):
        cos_t = np.cos(theta[j])
        sin_t = np.sin(theta[j])
        
        # Boundary points for pupil and iris
        x_p = xp + rp * cos_t
        y_p = yp + rp * sin_t
        x_i = xi + ri * cos_t
        y_i = yi + ri * sin_t
        
        # Linear interpolation between pupil and iris boundaries
        for i in range(radial_res):
            # r ranges from 0 (pupil) to 1 (iris)
            r = i / radial_res
            
            x = (1 - r) * x_p + r * x_i
            y = (1 - r) * y_p + r * y_i
            
            # Map back to original image coordinates
            # Use basic boundary check
            if 0 <= int(y) < img.shape[0] and 0 <= int(x) < img.shape[1]:
                norm_img[i, j] = img[int(y), int(x)]
            else:
                norm_img[i, j] = 128 # Neutral gray for out-of-bounds
                
    return norm_img
