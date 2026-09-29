"""Synthetic stitching test case with a known ground-truth homography.

A procedural scene (shapes, lines and text on a gradient) is rendered once.
Image B is a plain crop of the right part of the scene. Image A is a
perspective view of the left part, slightly darker, as if taken from a
rotated camera with different exposure. Because both views are generated
from the scene, the true homography A -> B is known exactly.

Usage (writes the README demo figure):
    uv run python src/synthetic.py assets
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

import cv2
import numpy as np

SCENE_SIZE = (1300, 700)  # (width, height)
VIEW_SIZE = (800, 600)

# Scene -> image B: a crop starting at x=500, y=50.
H_SCENE_TO_B = np.array([[1.0, 0.0, -500.0], [0.0, 1.0, -50.0], [0.0, 0.0, 1.0]])

# Quadrilateral of the scene that image A sees (clockwise from top-left).
A_QUAD = np.float32([[0, 30], [840, 0], [880, 690], [30, 660]])


def make_scene(size=SCENE_SIZE, seed=0):
    """Random but deterministic textured scene (BGR uint8)."""
    w, h = size
    rng = np.random.default_rng(seed)

    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    scene = np.stack([60 + 80 * xx, 90 + 60 * yy, 140 - 60 * xx * yy], axis=2).astype(np.uint8)

    def color():
        return tuple(int(c) for c in rng.integers(0, 256, 3))

    for _ in range(140):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 4)
        if kind == 0:
            dx, dy = rng.integers(10, 80, 2)
            cv2.rectangle(scene, (x, y), (x + int(dx), y + int(dy)), color(), -1)
        elif kind == 1:
            cv2.circle(scene, (x, y), int(rng.integers(5, 40)), color(), -1)
        elif kind == 2:
            x2, y2 = x + int(rng.integers(-120, 120)), y + int(rng.integers(-120, 120))
            cv2.line(scene, (x, y), (x2, y2), color(), int(rng.integers(1, 5)))
        else:
            text = "".join(rng.choice(list(string.ascii_uppercase + string.digits), 4))
            cv2.putText(scene, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.6)), color(), 2)

    return cv2.GaussianBlur(scene, (3, 3), 0)


def make_views(scene, exposure_a=0.85):
    """Return (img_a, img_b, H_true) where H_true maps A's pixels into B's frame."""
    w, h = VIEW_SIZE
    view_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H_scene_to_a = cv2.getPerspectiveTransform(A_QUAD, view_rect).astype(np.float64)

    img_a = cv2.warpPerspective(scene, H_scene_to_a, (w, h))
    img_a = np.clip(img_a.astype(np.float64) * exposure_a, 0, 255).astype(np.uint8)
    img_b = cv2.warpPerspective(scene, H_SCENE_TO_B, (w, h))

    H_true = H_SCENE_TO_B @ np.linalg.inv(H_scene_to_a)
    return img_a, img_b, H_true / H_true[2, 2]


def corner_error(H_est, H_true, size=VIEW_SIZE):
    """Mean distance (px) between image A's corners mapped by H_est and by H_true."""
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    p_est, p_true = H_est @ corners, H_true @ corners
    p_est, p_true = p_est[:2] / p_est[2], p_true[:2] / p_true[2]
    return float(np.linalg.norm(p_est - p_true, axis=0).mean())


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 15 + 20, 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def demo_figure(img_a, img_b, panorama, width=1200):
    """Inputs side by side on top, stitched result below, as one image."""
    gap = 10

    def fit(img):
        return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)

    spacer = np.full((img_a.shape[0], gap, 3), 255, np.uint8)
    top = fit(np.hstack([_label(img_a, "input A"), spacer, _label(img_b, "input B")]))
    bottom = _label(fit(panorama), "stitched")
    return np.vstack([top, np.full((gap, width, 3), 255, np.uint8), bottom])


def main(out_dir):
    from features import detect_and_match
    from homography import ransac_homography
    from main import stitch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_a, img_b, H_true = make_views(make_scene())

    src, dst = detect_and_match(img_a, img_b)
    H_ours, inliers = ransac_homography(src, dst, rng=np.random.default_rng(0))
    H_cv, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print(f"matches {len(src)}, inliers {int(inliers.sum())}")
    print(f"corner error vs ground truth: ours {corner_error(H_ours, H_true):.3f} px, "
          f"cv2.findHomography {corner_error(H_cv, H_true):.3f} px")

    panorama = stitch(img_a, img_b, rng=np.random.default_rng(0))
    cv2.imwrite(str(out_dir / "synthetic_demo.jpg"), demo_figure(img_a, img_b, panorama),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out_dir / 'synthetic_demo.jpg'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets")
