"""
IrisNormalization.py

Purpose:
    This script maps the annular iris region from Cartesian image coordinates
    into a fixed-size rectangular representation. This step is usually called
    iris normalization or rubber-sheet mapping.

Overall Logic:
    1. Take the pupil circle and iris outer circle as input.
    2. Sample points along both circles for a set of angular positions.
    3. For each angular position, interpolate points from the pupil boundary
       to the iris boundary along the radial direction.
    4. Use bilinear interpolation to estimate pixel values at non-integer locations.
    5. Construct a normalized iris image with fixed dimensions.

Why this script is needed:
    Different iris images may have different pupil sizes, iris sizes, or slight
    geometric variation. Directly comparing raw circular iris regions is difficult.
    This script converts the iris ring into a fixed-size image so that all samples
    can be processed consistently in later enhancement and feature extraction steps.

Key Variables and Parameters:
    pupil_params:
        Tuple containing pupil center coordinates and radius: (xp, yp, rp).
    iris_params:
        Tuple containing iris outer boundary center coordinates and radius: (xi, yi, ri).
    radial_res:
        Number of samples in the radial direction of the normalized image.
    angular_res:
        Number of samples in the angular direction of the normalized image.
    theta:
        Array of angular positions used to trace the iris ring.
    x_p, y_p:
        Cartesian coordinates of the sampled points on the pupil boundary.
    x_i, y_i:
        Cartesian coordinates of the sampled points on the iris outer boundary.
    curr_x, curr_y:
        The current interpolated point between the pupil and iris boundaries.
    norm_img:
        Output normalized iris image.
"""

import numpy as np


def bilinear(img: np.ndarray, x: float, y: float) -> float:
    """
    Compute the bilinearly interpolated pixel value at a non-integer location.

    Logic:
        1. Find the four neighboring pixels surrounding the target point.
        2. Compute the horizontal and vertical fractional offsets.
        3. Use weighted averaging to interpolate the intensity value.

    Args:
        img:
            Input grayscale image.
        x:
            Horizontal coordinate of the target point.
        y:
            Vertical coordinate of the target point.

    Returns:
        Interpolated pixel intensity value.

    Why this function is needed:
        During normalization, the mapped iris coordinates are usually not integers.
        Direct indexing would lose precision and introduce artifacts. Bilinear
        interpolation provides a smoother and more accurate sampling result.

    Key Variables:
        x0, y0:
            Integer floor coordinates of the target point.
        x1, y1:
            Neighboring coordinates to the right and below.
        dx, dy:
            Fractional offsets in the horizontal and vertical directions.
        val:
            Final interpolated pixel value.
    """
    # Integer coordinates surrounding the target point
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, img.shape[1] - 1)
    y1 = min(y0 + 1, img.shape[0] - 1)

    # Fractional offsets from the top-left corner
    dx = x - x0
    dy = y - y0

    # Bilinear interpolation using four neighboring pixels
    val = (
        (1 - dx) * (1 - dy) * float(img[y0, x0]) +
        dx * (1 - dy) * float(img[y0, x1]) +
        (1 - dx) * dy * float(img[y1, x0]) +
        dx * dy * float(img[y1, x1])
    )
    return val


def normalize_iris(img: np.ndarray, pupil_params: tuple, iris_params: tuple, cfg: dict) -> np.ndarray:
    """
    Normalize the iris region into a fixed-size rectangular image.

    Main Steps:
        1. Read the radial and angular resolution from the configuration.
        2. Compute boundary points on the pupil circle and iris outer circle.
        3. For each angular position, interpolate points between the two boundaries.
        4. Sample the image intensity at each interpolated point using bilinear interpolation.
        5. Store the sampled values in the normalized image.

    Args:
        img:
            Input grayscale eye image.
        pupil_params:
            Tuple (xp, yp, rp) representing pupil center and radius.
        iris_params:
            Tuple (xi, yi, ri) representing iris outer boundary center and radius.
        cfg:
            Dictionary containing normalization parameters such as:
            - radial_res
            - angular_res

    Returns:
        A normalized iris image of size (radial_res, angular_res).

    Key Variables:
        radial_res:
            Number of rows in the normalized image.
        angular_res:
            Number of columns in the normalized image.
        xp, yp, rp:
            Pupil center coordinates and radius.
        xi, yi, ri:
            Iris center coordinates and radius.
        theta:
            Array of angular sample positions in radians.
        cos_t, sin_t:
            Cosine and sine of all angular positions.
        x_p, y_p:
            Sampled coordinates on the pupil boundary.
        x_i, y_i:
            Sampled coordinates on the iris outer boundary.
        r:
            Normalized radial position between 0 and 1.
        curr_x, curr_y:
            Current Cartesian coordinates being sampled.
        norm_img:
            Output normalized iris image.
    """
    # Read output image dimensions from the configuration
    radial_res = cfg.get("radial_res", 64)
    angular_res = cfg.get("angular_res", 512)

    # Unpack pupil and iris circle parameters
    xp, yp, rp = pupil_params
    xi, yi, ri = iris_params

    # Allocate the output normalized image
    norm_img = np.zeros((radial_res, angular_res), dtype=np.float32)

    # Generate angular sampling positions.
    # endpoint=False avoids duplicating the first and last angular columns.
    theta = np.linspace(0.0, 2.0 * np.pi, angular_res, endpoint=False, dtype=np.float32)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Compute the pupil boundary points for all angular positions
    x_p = xp + rp * cos_t
    y_p = yp + rp * sin_t

    # Compute the iris outer boundary points for all angular positions
    x_i = xi + ri * cos_t
    y_i = yi + ri * sin_t

    # Sample points between pupil and iris boundaries
    for j in range(angular_res):
        for i in range(radial_res):
            # Radial interpolation factor from pupil boundary (0) to iris boundary (1)
            r = i / (radial_res - 1)

            # Linear interpolation between the two boundaries
            curr_x = (1.0 - r) * x_p[j] + r * x_i[j]
            curr_y = (1.0 - r) * y_p[j] + r * y_i[j]

            # Only sample points that stay inside valid image coordinates
            if 0 <= curr_x < img.shape[1] - 1 and 0 <= curr_y < img.shape[0] - 1:
                norm_img[i, j] = bilinear(img, float(curr_x), float(curr_y))

    # Convert to standard grayscale range
    norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
    return norm_img
