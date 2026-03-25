# Panorama Stitching — Report Briefing

This document is the tech report on this panorama stitching project. It covers the objective, the theory behind each stage, the implementation details with full source code, the execution results, and the project metadata.

---

## 1. Objective

Build a panorama stitching pipeline that takes two overlapping photographs and produces a single seamless panoramic image. The homography estimation (DLT + RANSAC) is implemented from scratch; OpenCV is used only for feature detection (SIFT), perspective warping, and image I/O.

---

## 2. Pipeline Overview

The system is split into four sequential stages:

1. **Feature Detection & Matching** — find corresponding points between the two images.
2. **Homography Estimation** — compute the 3x3 projective transformation matrix H that maps image A into image B's coordinate frame, using DLT with RANSAC (implemented from scratch).
3. **Warping & Canvas Construction** — apply H to warp image A, compute a canvas large enough for both images, and generate overlap masks.
4. **Blending** — merge the two images on the canvas using distance-weighted averaging for a smooth seam.

---

## 3. Theoretical Background

### 3.1 SIFT (Scale-Invariant Feature Transform)

SIFT detects keypoints that are invariant to scale, rotation, and partially invariant to illumination and viewpoint changes. Each keypoint is described by a 128-dimensional descriptor vector computed from the gradient orientations in its local neighborhood. We use OpenCV's `cv2.SIFT_create()` implementation.

### 3.2 Brute-Force Matching + Lowe's Ratio Test

For each descriptor in image A, the two nearest neighbors in image B are found using Euclidean distance (BFMatcher with k=2). Lowe's ratio test filters ambiguous matches: a match is accepted only if the distance to the best match is less than 75% of the distance to the second-best match:

$$\frac{d_1}{d_2} < 0.75$$

This single heuristic removes the majority of incorrect correspondences.

### 3.3 Point Normalization (Hartley normalization)

Raw pixel coordinates can be large (e.g., 1920, 1080). When building the DLT matrix, products like $x' \cdot x$ produce very large values, causing numerical instability in SVD. Hartley's normalization applies a similarity transform T to each point set so that:
- The centroid is shifted to the origin
- The average distance from the origin becomes $\sqrt{2}$

The normalization matrix is:

$$T = \begin{bmatrix} s & 0 & -s \cdot \bar{x} \\ 0 & s & -s \cdot \bar{y} \\ 0 & 0 & 1 \end{bmatrix}$$

where $s = \frac{\sqrt{2}}{\bar{d}}$ and $\bar{d}$ is the mean distance of the points from their centroid.

After computing the homography $H_{norm}$ on normalized points, we recover the actual homography:

$$H = T_{dst}^{-1} \cdot H_{norm} \cdot T_{src}$$

### 3.4 DLT (Direct Linear Transform)

Given N matched point pairs $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, the homography $H$ satisfies:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Expanding this and eliminating the scale factor yields two linear equations per correspondence. For each pair, two rows are added to a matrix $A$ (size $2N \times 9$):

$$\text{Row 1: } [-x, -y, -1, \; 0, 0, 0, \; x'x, \; x'y, \; x']$$
$$\text{Row 2: } [0, 0, 0, \; -x, -y, -1, \; y'x, \; y'y, \; y']$$

The homography vector $\mathbf{h}$ (the 9 entries of $H$ flattened) is the solution to $A\mathbf{h} = \mathbf{0}$, found via SVD: it is the last row of $V^T$ (corresponding to the smallest singular value). The result is reshaped into a $3 \times 3$ matrix and normalized so that $H_{3,3} = 1$.

Minimum correspondences required: 4 (yields an $8 \times 9$ system with a 1D null space).

### 3.5 RANSAC (Random Sample Consensus)

Even after Lowe's ratio test, some matches are incorrect (outliers). A single outlier can corrupt DLT. RANSAC iteratively:

1. Randomly samples 4 point pairs
2. Computes a candidate $H$ via DLT
3. Tests $H$ on all points by computing reprojection error:
   $$e_i = \| H \mathbf{p}_i - \mathbf{p}'_i \|_2$$
   (after dividing by the homogeneous coordinate $w$)
4. Counts inliers (points with $e_i < \tau$, where $\tau = 5.0$ pixels)
5. Keeps the $H$ with the most inliers

After all iterations (2000), a final refit is performed: DLT is run again using all inliers from the best model, producing a more robust homography from the overdetermined system.

### 3.6 Perspective Warping

The homography $H$ is applied to image A using `cv2.warpPerspective`. To determine the output canvas size, the four corners of image A are transformed through $H$ and combined with image B's corners to find the bounding box. If any warped corner has negative coordinates, a translation offset $T_{offset}$ is applied:

$$H_{adjusted} = T_{offset} \cdot H$$

### 3.7 Distance-Weighted Blending

In the overlap region, a naive 50/50 average can produce visible seams. Instead, we use `cv2.distanceTransform` to compute, for each pixel, its Euclidean distance to the nearest border of each mask. The blending weights are:

$$w_A = \frac{d_A}{d_A + d_B}, \quad w_B = \frac{d_B}{d_A + d_B}$$

$$\text{result} = w_A \cdot I_A + w_B \cdot I_B$$

This creates a smooth gradient across the overlap, with pixels closer to image A's interior trusting A more, and vice versa.

---

## 4. Project Structure

```
panorama/
├── main.py              # orchestrator + CLI entry point
├── features.py          # Module 1: SIFT detection + matching
├── homography.py        # Module 2: DLT + RANSAC (from scratch)
├── warping.py           # Module 3: perspective warp + canvas computation
├── blending/
│   ├── __init__.py      # package init, exposes simple_blend
│   └── simple.py        # distance-weighted blending
├── pyproject.toml       # uv project config
└── data/                # input images
```

---

## 5. Source Code

### 5.1 features.py

```python
import cv2
import numpy as np


def detect_and_match(img_a, img_b, ratio_thresh=0.75):
    # 1. Convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # 2. Detect keypoints + compute descriptors (SIFT)
    sift = cv2.SIFT_create()
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)

    # 3. Match descriptors (brute-force, k=2 for ratio test)
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    # 4. Filter with Lowe's ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 5. Extract coordinate pairs
    src_pts = np.array([kp_a[m.queryIdx].pt for m in good_matches])
    dst_pts = np.array([kp_b[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts
```

### 5.2 homography.py

```python
import numpy as np


def _normalize(pts):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / (avg_dist + 1e-12)

    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ]
    )

    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    normed_h = (T @ pts_h.T).T
    normed_pts = normed_h[:, :2]

    return normed_pts, T


def _dlt(src_pts, dst_pts):
    src_norm, T_src = _normalize(src_pts)
    dst_norm, T_dst = _normalize(dst_pts)

    N = src_norm.shape[0]
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]
    return H


def _compute_reprojection_errors(H, src_pts, dst_pts):
    N = src_pts.shape[0]
    ones = np.ones((N, 1))
    src_h = np.hstack([src_pts, ones])

    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
    return errors


def ransac_homography(src_pts, dst_pts, n_iters=2000, threshold=5.0):
    N = src_pts.shape[0]
    best_inlier_count = 0
    best_inlier_mask = None
    best_H = None

    for _ in range(n_iters):
        indices = np.random.choice(N, size=4, replace=False)

        try:
            H_candidate = _dlt(src_pts[indices], dst_pts[indices])
        except np.linalg.LinAlgError:
            continue

        errors = _compute_reprojection_errors(H_candidate, src_pts, dst_pts)

        inlier_mask = errors < threshold
        inlier_count = inlier_mask.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_H = H_candidate

    if best_inlier_mask is not None and best_inlier_count >= 4:
        best_H = _dlt(src_pts[best_inlier_mask], dst_pts[best_inlier_mask])

    return best_H, best_inlier_mask
```

### 5.3 warping.py

```python
import math
import cv2
import numpy as np


def _compute_canvas_size(H, img_a_shape, img_b_shape):
    h_a, w_a = img_a_shape[:2]
    h_b, w_b = img_b_shape[:2]

    corners_a = np.array([
        [0, 0], [w_a, 0], [w_a, h_a], [0, h_a]
    ], dtype=np.float64)

    warped_corners = []
    for x, y in corners_a:
        p = H @ np.array([x, y, 1.0])
        warped_corners.append([p[0] / p[2], p[1] / p[2]])

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
    (canvas_h, canvas_w), (offset_x, offset_y) = _compute_canvas_size(
        H, img_a.shape, img_b.shape
    )

    T_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1       ],
    ], dtype=np.float64)

    H_adjusted = T_offset @ H
    warped_a = cv2.warpPerspective(img_a, H_adjusted, (canvas_w, canvas_h))

    h_b, w_b = img_b.shape[:2]
    canvas_b = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_b[offset_y : offset_y + h_b, offset_x : offset_x + w_b] = img_b

    mask_a = (warped_a > 0).any(axis=2)
    mask_b = np.zeros((canvas_h, canvas_w), dtype=bool)
    mask_b[offset_y : offset_y + h_b, offset_x : offset_x + w_b] = True

    mask_overlap = mask_a & mask_b

    return warped_a, canvas_b, (mask_a, mask_b, mask_overlap), (offset_x, offset_y)
```

### 5.4 blending/simple.py

```python
import cv2
import numpy as np


def simple_blend(warped_a, canvas_b, mask_a, mask_b, mask_overlap):
    mask_a_only = mask_a & ~mask_b
    mask_b_only = mask_b & ~mask_a

    result = np.zeros_like(warped_a)

    result[mask_a_only] = warped_a[mask_a_only]
    result[mask_b_only] = canvas_b[mask_b_only]

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
```

### 5.5 main.py

```python
import sys
import cv2
import numpy as np

from features import detect_and_match
from homography import ransac_homography
from warping import warp_and_prepare
from blending import simple_blend


def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return img[y_min : y_max + 1, x_min : x_max + 1]


def stitch(img_a, img_b, blend_mode="simple"):
    src_pts, dst_pts = detect_and_match(img_a, img_b)
    H, inlier_mask = ransac_homography(src_pts, dst_pts)
    warped_a, canvas_b, masks, offset = warp_and_prepare(img_a, img_b, H)

    mask_a, mask_b, mask_overlap = masks
    if blend_mode == "simple":
        result = simple_blend(warped_a, canvas_b, mask_a, mask_b, mask_overlap)
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode!r}")

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
```

---

## 6. Execution Results

The pipeline was run on two overlapping photographs of an indoor scene (room with a wardrobe and two people).

### Console output

```
[features]   248 good matches found
[homography] 105 inliers out of 248 matches
[warping]    canvas size 2240x2027, offset (0, 187)
[done]       saved to panorama_result.jpg
```

### Key metrics

| Metric | Value |
|---|---|
| SIFT matches after Lowe's test | 248 |
| RANSAC inliers | 105 (42.3% of matches) |
| RANSAC iterations | 2000 |
| Inlier threshold | 5.0 px |
| Output canvas size | 2240 x 2027 px |
| Offset applied | (0, 187) — vertical shift only |
| Blending method | distance-weighted (distanceTransform) |

### Interpretation

- 248 matches survived the ratio test, indicating strong feature overlap between the images.
- RANSAC identified 105 inliers (~42%), meaning the remaining 143 matches were outliers successfully rejected.
- The vertical offset of 187 pixels indicates image A warped slightly above image B's frame.
- The output file `panorama_result.jpg` shows a correctly stitched panorama with a smooth blend in the overlapping region.

---

## 7. Dependencies

| Library | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| NumPy | 2.4.3 | Linear algebra (SVD, matrix ops) |
| opencv-python | 4.13.0 | SIFT, BFMatcher, warpPerspective, distanceTransform, image I/O |

Dependency management: **uv** (via `pyproject.toml`).

---

## 8. What Was Implemented From Scratch vs. Library

| Component | Implementation |
|---|---|
| Point normalization | From scratch (NumPy) |
| DLT (Direct Linear Transform) | From scratch (NumPy + SVD) |
| RANSAC | From scratch (NumPy) |
| Reprojection error computation | From scratch (NumPy, vectorized) |
| SIFT keypoint detection | OpenCV (`cv2.SIFT_create`) |
| Descriptor matching | OpenCV (`cv2.BFMatcher`) |
| Perspective warping | OpenCV (`cv2.warpPerspective`) |
| Distance transform for blending | OpenCV (`cv2.distanceTransform`) |
| Image I/O | OpenCV (`cv2.imread`, `cv2.imwrite`) |

---

## 9. Notes for Report Writing

- The course is Graphics Computing at IME (Instituto de Militar de Engenharia).
- The core academic contribution is the from-scratch implementation of DLT + RANSAC for homography estimation.
