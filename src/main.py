import sys

import cv2
import numpy as np

from features import detect_and_match
from homography import ransac_homography
from warping import warp_and_prepare
from blending import simple_blend


def crop_black_borders(img):
    """Remove black (zero) borders from the stitched result."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img[y_min : y_max + 1, x_min : x_max + 1]


def stitch(img_a, img_b, blend_mode="simple"):
    """Stitch two images into a panorama.

    Parameters
    ----------
    img_a, img_b : ndarray (H, W, 3) BGR images.
    blend_mode : str – "simple" (default).

    Returns
    -------
    result : ndarray (H, W, 3) stitched panorama.
    """
    # Module 1 — feature detection & matching
    src_pts, dst_pts = detect_and_match(img_a, img_b)
    print(f"[features]   {len(src_pts)} good matches found")

    # Module 2 — homography via RANSAC + DLT
    H, inlier_mask = ransac_homography(src_pts, dst_pts)
    print(f"[homography] {inlier_mask.sum()} inliers out of {len(src_pts)} matches")

    # Module 3 — warping
    warped_a, canvas_b, masks, offset = warp_and_prepare(img_a, img_b, H)
    print(f"[warping]    canvas size {warped_a.shape[1]}x{warped_a.shape[0]}, offset {offset}")

    # Module 4 — blending
    mask_a, mask_b, mask_overlap = masks
    if blend_mode == "simple":
        result = simple_blend(warped_a, canvas_b, mask_a, mask_b, mask_overlap)
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode!r}")

    # Crop black borders
    result = crop_black_borders(result)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <image_a> <image_b> [output]")
        sys.exit(1)

    path_a = sys.argv[1]
    path_b = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "panorama_result.jpg"

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        print(f"Error: cannot read {path_a}")
        sys.exit(1)
    if img_b is None:
        print(f"Error: cannot read {path_b}")
        sys.exit(1)

    result = stitch(img_a, img_b)
    cv2.imwrite(output_path, result)
    print(f"[done]       saved to {output_path}")


if __name__ == "__main__":
    main()
