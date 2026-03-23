"""
IrisLocalization.py

Purpose:
    This script detects the pupil boundary and the outer iris boundary from a
    grayscale eye image. These two circular boundaries are required for the
    later normalization step, where the annular iris region is mapped into a
    fixed-size rectangular representation.

Overall Logic:
    1. Perform a coarse pupil center estimation using horizontal and vertical
       projection minima.
    2. Refine the pupil location inside a local 120x120 region using adaptive
       thresholding and contour analysis.
    3. Refine the pupil circle using a constrained Hough Circle Transform.
    4. Detect the outer iris boundary using edge detection and a second
       constrained Hough Circle Transform.
    5. Return both pupil parameters and iris parameters.

Why this script is needed:
    Accurate localization is essential in an iris recognition system. If the
    pupil circle or outer iris circle is incorrect, the following normalization
    and feature extraction stages will also be inaccurate, which can reduce the
    final recognition performance.

Key Variables and Parameters:
    xp, yp:
        Estimated pupil center coordinates.
    rp:
        Estimated pupil radius.
    xp0, yp0:
        Initial coarse pupil center estimated from projection minima.
    pupil_params:
        Final pupil circle parameters in the form (center_x, center_y, radius).
    iris_params:
        Final iris outer circle parameters in the form (center_x, center_y, radius).
    roi:
        Local image region used for adaptive pupil refinement.
    bw:
        Binary image obtained after thresholding the local ROI.
    contours:
        Connected components extracted from the binary pupil candidate region.
    circles_p:
        Pupil circle candidates returned by Hough Circle Transform.
    circles_i:
        Iris outer boundary candidates returned by Hough Circle Transform.
    center_offset:
        Distance between the candidate iris center and the pupil center.
"""

import cv2
import numpy as np
from typing import Tuple


def _clip_roi(x0, y0, x1, y1, w, h):
    """
    Clip a rectangular region so that it stays inside the image boundaries.

    Args:
        x0, y0:
            Top-left corner of the proposed region.
        x1, y1:
            Bottom-right corner of the proposed region.
        w, h:
            Width and height of the full image.

    Returns:
        A valid clipped rectangle (x0, y0, x1, y1).

    Why this function is needed:
        During localization, local search regions are often centered around an
        estimated pupil position. If the estimate is close to the image border,
        the region may partially fall outside the image. This helper prevents
        invalid indexing.
    """
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return x0, y0, x1, y1


def _coarse_pupil_center(img: np.ndarray) -> tuple[int, int]:
    """
    Estimate a coarse pupil center using projection minima.

    Logic:
        1. Convert the image to float.
        2. Compute the mean gray value along each column and each row.
        3. The darkest column and darkest row are selected as the coarse pupil
           center because the pupil is typically the darkest region in the eye image.

    Args:
        img:
            Grayscale image.

    Returns:
        A tuple (xp, yp) representing the coarse pupil center.

    Key Variables:
        col_proj:
            Average gray value for each column.
        row_proj:
            Average gray value for each row.
        xp:
            Column index with the minimum average intensity.
        yp:
            Row index with the minimum average intensity.
    """
    img_f = img.astype(np.float32)
    col_proj = np.mean(img_f, axis=0)
    row_proj = np.mean(img_f, axis=1)

    xp = int(np.argmin(col_proj))
    yp = int(np.argmin(row_proj))
    return xp, yp


def _adaptive_pupil_from_local_roi(img: np.ndarray, xp: int, yp: int) -> tuple[int, int, int]:
    """
    Refine the pupil center and radius inside a local 120x120 region.

    Logic:
        1. Center a local search window around the current pupil estimate.
        2. Apply Otsu thresholding inside that local window.
        3. Use morphological opening and closing to clean the binary region.
        4. Find contours and keep plausible pupil candidates.
        5. Select the best candidate based on area and closeness to the center.
        6. Update the pupil estimate and repeat once more for refinement.

    Args:
        img:
            Grayscale image.
        xp, yp:
            Initial pupil center estimate.

    Returns:
        A tuple (xp, yp, rp) representing the refined pupil center and radius.

    Key Variables:
        roi:
            The 120x120 local search region around the pupil center.
        bw:
            Binary image after Otsu thresholding.
        contours:
            Contours found in the binary pupil candidate map.
        candidates:
            A list of plausible pupil candidates with their scores.
        cx_roi, cy_roi:
            The center of the local ROI, used to prefer candidates near the middle.
        rp:
            Estimated pupil radius.
    """
    h, w = img.shape

    for _ in range(2):
        # Define a local 120x120 search region centered at the current estimate
        x0, y0, x1, y1 = _clip_roi(xp - 60, yp - 60, xp + 60, yp + 60, w, h)
        roi = img[y0:y1, x0:x1]

        # Apply Otsu thresholding inside the local ROI
        _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove small noise and close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        # Find connected pupil candidates
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            candidates = []
            cx_roi = (x1 - x0) / 2.0
            cy_roi = (y1 - y0) / 2.0

            for c in contours:
                area = cv2.contourArea(c)
                if area < 100:
                    continue

                (cx, cy), r = cv2.minEnclosingCircle(c)
                if r < 15 or r > 70:
                    continue

                # Prefer larger regions close to the ROI center
                dist2 = (cx - cx_roi) ** 2 + (cy - cy_roi) ** 2
                score = area - 0.5 * dist2
                candidates.append((score, c))

            if candidates:
                _, best = max(candidates, key=lambda t: t[0])
                M = cv2.moments(best)

                if M["m00"] > 1e-6:
                    cx = int(x0 + M["m10"] / M["m00"])
                    cy = int(y0 + M["m01"] / M["m00"])
                    (_, _), r = cv2.minEnclosingCircle(best)

                    xp, yp, rp = cx, cy, int(r)
                    continue

        # Fallback radius if no stable contour is found
        rp = 45

    return xp, yp, rp


def localize_iris(img: np.ndarray, config=None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Localize both the pupil boundary and the outer iris boundary.

    Main Steps:
        1. Smooth the image with median filtering.
        2. Estimate a coarse pupil center from projection minima.
        3. Refine the pupil center and radius using local adaptive thresholding.
        4. Refine the pupil circle with a local Hough Circle Transform.
        5. Detect the outer iris boundary with a constrained global Hough Circle Transform.
        6. Return both pupil parameters and iris parameters.

    Args:
        img:
            Input grayscale eye image.
        config:
            Optional configuration input. It is not used in the current implementation.

    Returns:
        A tuple:
            - pupil_params = (xp, yp, rp)
            - iris_params  = (xi, yi, ri)

    Key Variables:
        blurred:
            Median-filtered image used for more stable edge detection and projection.
        xp0, yp0:
            Coarse pupil center from projection minima.
        xp, yp, rp:
            Refined pupil center and radius.
        local:
            Local image patch used for pupil Hough refinement.
        edges_local:
            Edge map used for pupil circle detection.
        circles_p:
            Pupil circle candidates from local Hough transform.
        edges:
            Global edge map used for outer iris boundary detection.
        circles_i:
            Iris circle candidates from constrained Hough transform.
        iris_params:
            Final chosen outer iris circle parameters.
    """
    # Ensure the input is a grayscale image
    if img.ndim != 2:
        raise ValueError("Input must be grayscale.")

    h, w = img.shape

    # Median filtering helps suppress impulse noise before localization
    blurred = cv2.medianBlur(img, 5)

    # Step 1: Coarse pupil center from horizontal / vertical projections
    xp0, yp0 = _coarse_pupil_center(blurred)

    # Step 2: Local adaptive threshold refinement of the pupil
    xp, yp, rp = _adaptive_pupil_from_local_roi(blurred, xp0, yp0)

    # Step 3a: Refine pupil circle using local Hough transform
    x0, y0, x1, y1 = _clip_roi(xp - 70, yp - 70, xp + 70, yp + 70, w, h)
    local = blurred[y0:y1, x0:x1]
    edges_local = cv2.Canny(local, 40, 120)

    pupil_params = (xp, yp, rp)

    circles_p = cv2.HoughCircles(
        edges_local,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=30,
        param1=100,
        param2=10,
        minRadius=max(15, rp - 15),
        maxRadius=min(70, rp + 20),
    )

    if circles_p is not None:
        circles_p = np.round(circles_p[0]).astype(int)

        # Choose the pupil candidate closest to the current estimate
        best = min(
            circles_p,
            key=lambda c: (c[0] + x0 - xp) ** 2 + (c[1] + y0 - yp) ** 2 + 0.5 * abs(c[2] - rp)
        )

        pupil_params = (int(best[0] + x0), int(best[1] + y0), int(best[2]))
        xp, yp, rp = pupil_params

    # Step 3b: Detect the outer iris boundary with constrained Hough transform
    edges = cv2.Canny(blurred, 50, 150)

    circles_i = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=120,
        param2=25,
        minRadius=max(80, int(rp * 2.0)),
        maxRadius=min(140, int(rp * 3.4)),
    )

    # Default fallback if no valid iris circle is found
    iris_params = (xp, yp, int(rp * 2.8))

    if circles_i is not None:
        circles_i = np.round(circles_i[0]).astype(int)

        candidates = []
        for c in circles_i:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])

            # The pupil and iris circles are not necessarily concentric,
            # but the offset should still be limited
            center_offset = np.sqrt((cx - xp) ** 2 + (cy - yp) ** 2)
            if center_offset > 25:
                continue

            # The outer iris radius should be significantly larger than the pupil radius
            if r <= rp + 20:
                continue

            # Smaller score indicates a more plausible iris circle
            score = center_offset + 0.3 * abs(r - int(rp * 2.8))
            candidates.append((score, (cx, cy, r)))

        if candidates:
            _, iris_params = min(candidates, key=lambda t: t[0])

    return pupil_params, iris_params
