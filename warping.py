import math

import cv2
import numpy as np


def _compute_canvas_size(H, img_a_shape, img_b_shape):
    """Determine the canvas dimensions and offset needed to fit both images.

    Parameters
    ----------
    H : ndarray (3, 3) – homography mapping image A into image B's frame.
    img_a_shape, img_b_shape : tuple (H, W, ...).

    Returns
    -------
    canvas_size : (height, width)
    offset : (offset_x, offset_y)
    """
    h_a, w_a = img_a_shape[:2]
    h_b, w_b = img_b_shape[:2]

    # Corners of image A
    corners_a = np.array([
        [0,   0  ],
        [w_a, 0  ],
        [w_a, h_a],
        [0,   h_a],
    ], dtype=np.float64)

    # Transform corners through H
    warped_corners = []
    for x, y in corners_a:
        p = H @ np.array([x, y, 1.0])
        warped_corners.append([p[0] / p[2], p[1] / p[2]])

    # Corners of image B (stays in place)
    corners_b = [[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]

    all_corners = np.array(warped_corners + corners_b)

    min_x = math.floor(all_corners[:, 0].min())
    min_y = math.floor(all_corners[:, 1].min())
    max_x = math.ceil(all_corners[:, 0].max())
    max_y = math.ceil(all_corners[:, 1].max())

    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    return (canvas_height, canvas_width), (offset_x, offset_y)


def warp_and_prepare(img_a, img_b, H):
    """Warp image A through H and place image B on a common canvas.

    Parameters
    ----------
    img_a, img_b : ndarray (H, W, 3) BGR images.
    H : ndarray (3, 3)

    Returns
    -------
    warped_a : ndarray – canvas-sized image with A warped.
    canvas_b : ndarray – canvas-sized image with B placed.
    masks : tuple (mask_a, mask_b, mask_overlap)
        Boolean masks of shape (canvas_h, canvas_w).
    offset : (offset_x, offset_y)
    """
    (canvas_h, canvas_w), (offset_x, offset_y) = _compute_canvas_size(
        H, img_a.shape, img_b.shape
    )

    # Translation matrix to shift for negative coordinates
    T_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1       ],
    ], dtype=np.float64)

    # Warp image A
    H_adjusted = T_offset @ H
    warped_a = cv2.warpPerspective(img_a, H_adjusted, (canvas_w, canvas_h))

    # Place image B on canvas
    h_b, w_b = img_b.shape[:2]
    canvas_b = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_b[offset_y : offset_y + h_b, offset_x : offset_x + w_b] = img_b

    # Compute masks
    mask_a = (warped_a > 0).any(axis=2)
    mask_b = np.zeros((canvas_h, canvas_w), dtype=bool)
    mask_b[offset_y : offset_y + h_b, offset_x : offset_x + w_b] = True

    mask_overlap = mask_a & mask_b

    return warped_a, canvas_b, (mask_a, mask_b, mask_overlap), (offset_x, offset_y)
