import cv2
import numpy as np


def detect_and_match(img_a, img_b, ratio_thresh=0.75):
    """Detect keypoints in both images and return matched coordinate pairs.

    Parameters
    ----------
    img_a, img_b : ndarray (H, W, 3) BGR images.
    ratio_thresh : float – Lowe's ratio test threshold.

    Returns
    -------
    src_pts : ndarray (N, 2) – matched coordinates from image A.
    dst_pts : ndarray (N, 2) – matched coordinates from image B.
    """
    # 1. Convert to grayscale
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # 2. Detect keypoints + compute descriptors (SIFT)
    sift = cv2.SIFT_create()
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)

    if desc_a is None or desc_b is None or len(kp_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    # 3. Match descriptors (brute-force, k=2 for ratio test)
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)

    # 4. Filter with Lowe's ratio test (knnMatch can return fewer than 2 neighbours)
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good_matches.append(pair[0])

    # 5. Extract coordinate pairs
    src_pts = np.array([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.array([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    return src_pts, dst_pts
