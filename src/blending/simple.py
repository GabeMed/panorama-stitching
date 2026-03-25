import cv2
import numpy as np


def simple_blend(warped_a, canvas_b, mask_a, mask_b, mask_overlap):
    """Blend warped image A and placed image B using distance-weighted averaging.

    In the overlap region, each pixel is weighted by its distance to the
    respective mask border (via cv2.distanceTransform), giving a smooth
    transition instead of a hard 50/50 split.

    Parameters
    ----------
    warped_a : ndarray (H, W, 3) – canvas-sized warped image A.
    canvas_b : ndarray (H, W, 3) – canvas-sized placed image B.
    mask_a, mask_b, mask_overlap : ndarray (H, W) bool masks.

    Returns
    -------
    result : ndarray (H, W, 3) uint8 blended image.
    """
    mask_a_only = mask_a & ~mask_b
    mask_b_only = mask_b & ~mask_a

    result = np.zeros_like(warped_a)

    # Exclusive regions
    result[mask_a_only] = warped_a[mask_a_only]
    result[mask_b_only] = canvas_b[mask_b_only]

    # Overlap region: distance-weighted blend
    if mask_overlap.any():
        dist_a = cv2.distanceTransform(mask_a.astype(np.uint8), cv2.DIST_L2, 5)
        dist_b = cv2.distanceTransform(mask_b.astype(np.uint8), cv2.DIST_L2, 5)

        total = dist_a + dist_b + 1e-12
        weight_a = dist_a / total
        weight_b = dist_b / total

        blended = (
            warped_a.astype(np.float64) * weight_a[:, :, np.newaxis]
            + canvas_b.astype(np.float64) * weight_b[:, :, np.newaxis]
        )
        result[mask_overlap] = blended[mask_overlap].astype(np.uint8)

    return result
