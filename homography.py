import numpy as np


def _normalize(pts):
    """Compute a normalizing similarity transform for a set of 2D points.

    Shifts the centroid to the origin and scales so the average distance
    from the origin is sqrt(2).

    Parameters
    ----------
    pts : ndarray (N, 2)

    Returns
    -------
    normed_pts : ndarray (N, 2)
    T : ndarray (3, 3) – the normalizing transformation matrix.
    """
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

    # Apply T to each point in homogeneous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # N x 3
    normed_h = (T @ pts_h.T).T  # N x 3
    normed_pts = normed_h[:, :2]

    return normed_pts, T


def _dlt(src_pts, dst_pts):
    """Compute the homography H such that dst ~ H @ src using DLT.

    Includes point normalization for numerical stability.

    Parameters
    ----------
    src_pts, dst_pts : ndarray (N, 2)  with N >= 4.

    Returns
    -------
    H : ndarray (3, 3)
    """
    # Normalize both sets
    src_norm, T_src = _normalize(src_pts)
    dst_norm, T_dst = _normalize(dst_pts)

    N = src_norm.shape[0]
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]

    # Solve Ah = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # De-normalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]
    return H


def _compute_reprojection_errors(H, src_pts, dst_pts):
    """Vectorized reprojection error for all point pairs.

    Parameters
    ----------
    H : ndarray (3, 3)
    src_pts, dst_pts : ndarray (N, 2)

    Returns
    -------
    errors : ndarray (N,)  Euclidean pixel distances.
    """
    N = src_pts.shape[0]
    ones = np.ones((N, 1))
    src_h = np.hstack([src_pts, ones])  # N x 3

    projected = (H @ src_h.T).T  # N x 3
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
    return errors


def ransac_homography(src_pts, dst_pts, n_iters=2000, threshold=5.0):
    """Compute the best homography using RANSAC + DLT.

    Parameters
    ----------
    src_pts, dst_pts : ndarray (N, 2)
    n_iters : int – number of RANSAC iterations.
    threshold : float – inlier distance threshold in pixels.

    Returns
    -------
    H : ndarray (3, 3)
    inlier_mask : ndarray (N,) bool
    """
    N = src_pts.shape[0]
    best_inlier_count = 0
    best_inlier_mask = None
    best_H = None

    for _ in range(n_iters):
        # 1. Random 4-point sample
        indices = np.random.choice(N, size=4, replace=False)

        # 2. Compute candidate H
        try:
            H_candidate = _dlt(src_pts[indices], dst_pts[indices])
        except np.linalg.LinAlgError:
            continue

        # 3. Reprojection errors on ALL points
        errors = _compute_reprojection_errors(H_candidate, src_pts, dst_pts)

        # 4. Count inliers
        inlier_mask = errors < threshold
        inlier_count = inlier_mask.sum()

        # 5. Keep best
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_H = H_candidate

    # 6. Final refit using ALL inliers
    if best_inlier_mask is not None and best_inlier_count >= 4:
        best_H = _dlt(src_pts[best_inlier_mask], dst_pts[best_inlier_mask])

    return best_H, best_inlier_mask
