"""DLT and RANSAC on synthetic correspondences with a known homography."""

import numpy as np
import pytest

from homography import _compute_reprojection_errors, _dlt, _normalize, ransac_homography

H_TRUE = np.array(
    [
        [0.92, -0.08, 310.0],
        [0.05, 1.03, -42.0],
        [1.2e-4, -6e-5, 1.0],
    ]
)


def _project(H, pts):
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def _correspondences(n, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform([0, 0], [1280, 720], size=(n, 2))
    dst = _project(H_TRUE, src) + rng.normal(scale=noise, size=(n, 2))
    return src, dst


def _unit(H):
    return H / H[2, 2]


def test_normalize_centres_and_scales():
    pts = np.random.default_rng(1).uniform(0, 2000, size=(50, 2))
    normed, T = _normalize(pts)
    np.testing.assert_allclose(normed.mean(axis=0), 0, atol=1e-12)
    assert np.linalg.norm(normed, axis=1).mean() == pytest.approx(np.sqrt(2))
    np.testing.assert_allclose(_project(T, pts), normed, atol=1e-12)


def test_dlt_recovers_known_homography_exactly():
    src, dst = _correspondences(30)
    np.testing.assert_allclose(_unit(_dlt(src, dst)), H_TRUE, rtol=1e-9, atol=1e-12)


def test_dlt_minimal_four_points():
    src = np.array([[10.0, 20.0], [1200.0, 40.0], [1150.0, 700.0], [30.0, 650.0]])
    dst = _project(H_TRUE, src)
    H = _dlt(src, dst)
    np.testing.assert_allclose(_project(H, src), dst, atol=1e-8)
    np.testing.assert_allclose(_unit(H), H_TRUE, rtol=1e-8, atol=1e-12)


def test_dlt_is_invariant_to_point_order_and_scale_of_h():
    src, dst = _correspondences(20)
    perm = np.random.default_rng(3).permutation(20)
    np.testing.assert_allclose(_unit(_dlt(src[perm], dst[perm])), _unit(_dlt(src, dst)), rtol=1e-9)


def test_dlt_with_pixel_noise_stays_subpixel():
    src, dst = _correspondences(200, noise=0.5)
    H = _dlt(src, dst)
    clean = _project(H_TRUE, src)
    assert np.sqrt(np.mean(np.sum((_project(H, src) - clean) ** 2, axis=1))) < 0.2


def test_reprojection_errors():
    src, dst = _correspondences(10)
    np.testing.assert_allclose(_compute_reprojection_errors(H_TRUE, src, dst), 0, atol=1e-9)
    shifted = dst + [3.0, 4.0]
    np.testing.assert_allclose(_compute_reprojection_errors(H_TRUE, src, shifted), 5.0, atol=1e-9)


def test_reprojection_error_of_point_at_infinity_is_inf():
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])  # maps x = 0 to w = 0
    errors = _compute_reprojection_errors(H, np.array([[0.0, 5.0], [2.0, 5.0]]), np.zeros((2, 2)))
    assert np.isinf(errors[0]) and np.isfinite(errors[1])


@pytest.mark.parametrize("outlier_fraction", [0.3, 0.5, 0.6])
def test_ransac_is_robust_to_outliers(outlier_fraction):
    n = 300
    src, dst = _correspondences(n, noise=0.7, seed=4)
    rng = np.random.default_rng(5)
    is_outlier = rng.random(n) < outlier_fraction
    dst[is_outlier] = rng.uniform([0, 0], [1600, 900], size=(is_outlier.sum(), 2))

    H, inliers = ransac_homography(src, dst, rng=np.random.default_rng(6))

    clean = _project(H_TRUE, src[~is_outlier])
    rms = np.sqrt(np.mean(np.sum((_project(H, src[~is_outlier]) - clean) ** 2, axis=1)))
    assert rms < 0.5
    # Almost every true inlier is kept; random outliers only survive by landing near the true position.
    assert inliers[~is_outlier].mean() > 0.98
    assert inliers[is_outlier].mean() < 0.02


def test_plain_dlt_breaks_under_the_same_outliers():
    """Guards the test above: without RANSAC the outliers do corrupt the estimate."""
    src, dst = _correspondences(300, noise=0.7, seed=4)
    rng = np.random.default_rng(5)
    is_outlier = rng.random(300) < 0.3
    dst[is_outlier] = rng.uniform([0, 0], [1600, 900], size=(is_outlier.sum(), 2))
    H = _dlt(src, dst)
    err = np.linalg.norm(_project(H, src[~is_outlier]) - _project(H_TRUE, src[~is_outlier]), axis=1)
    assert err.mean() > 10


def test_ransac_is_reproducible_with_a_seed():
    src, dst = _correspondences(100, noise=1.0)
    dst[:30] += 200
    H1, m1 = ransac_homography(src, dst, rng=np.random.default_rng(0))
    H2, m2 = ransac_homography(src, dst, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(H1, H2)
    np.testing.assert_array_equal(m1, m2)


def test_ransac_survives_degenerate_samples():
    # Half the points are collinear, which gives rank-deficient 4-point samples.
    src, dst = _correspondences(40)
    src[:20, 1] = 100.0
    dst[:20] = _project(H_TRUE, src[:20])
    H, inliers = ransac_homography(src, dst, n_iters=300, rng=np.random.default_rng(1))
    np.testing.assert_allclose(_unit(H), H_TRUE, rtol=1e-6, atol=1e-9)
    assert inliers.all()


def test_ransac_needs_four_points():
    src, dst = _correspondences(3)
    with pytest.raises(ValueError, match="at least 4"):
        ransac_homography(src, dst)
