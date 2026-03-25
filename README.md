# Panorama Stitching

Panorama stitching pipeline that combines two overlapping photographs into a single seamless image. The homography estimation (DLT + RANSAC) is implemented from scratch using NumPy; OpenCV handles feature detection (SIFT), perspective warping, and image I/O.

## How It Works

1. **Feature Detection & Matching** — SIFT keypoints are detected in both images and matched via brute-force with Lowe's ratio test to filter ambiguous correspondences.
2. **Homography Estimation** — A 3x3 homography matrix is computed from scratch using the Direct Linear Transform (DLT) with Hartley point normalization, wrapped in RANSAC (2000 iterations, 5px threshold) to reject outlier matches.
3. **Warping** — Image A is warped into image B's coordinate frame via the homography. A canvas is computed to fit both images, with translation offsets to handle negative coordinates.
4. **Blending** — The overlap region is blended using distance-weighted averaging (`cv2.distanceTransform`), producing a smooth seam instead of a hard cut.

## Project Structure

```
panorama/
├── main.py          # orchestrator + CLI entry point
├── features.py      # SIFT detection + BFMatcher + Lowe's ratio test
├── homography.py    # point normalization, DLT, RANSAC (from scratch)
├── warping.py       # perspective warp + canvas computation
└── blending/
    ├── __init__.py
    └── simple.py    # distance-weighted blending
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (install with `brew install uv`)

## Usage

Place two overlapping images in a `data/` folder (or anywhere accessible), then run:

```bash
uv run python main.py <image_a> <image_b> [output_path]
```

`output_path` defaults to `panorama_result.jpg` if omitted.

### Example

```bash
uv run python main.py "data/left.jpg" "data/right.jpg" panorama_result.jpg
```

Output:

```
[features]   248 good matches found
[homography] 105 inliers out of 248 matches
[warping]    canvas size 2240x2027, offset (0, 187)
[done]       saved to panorama_result.jpg
```

## Dependencies

Managed automatically by uv via `pyproject.toml`:

- **numpy** — linear algebra (SVD, matrix operations)
- **opencv-python** — SIFT, BFMatcher, warpPerspective, distanceTransform, image I/O
