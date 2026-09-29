"""Warping, blending and the full stitch on the synthetic scene."""

import cv2
import numpy as np
import pytest

from blending import simple_blend
from features import detect_and_match
from homography import ransac_homography
from main import crop_black_borders, stitch
from synthetic import VIEW_SIZE, corner_error, make_scene, make_views
from warping import _compute_canvas_size, warp_and_prepare


@pytest.fixture(scope="module")
def views():
    return make_views(make_scene())


def _translation(tx, ty):
    return np.array([[1.0, 0, tx], [0, 1.0, ty], [0, 0, 1.0]])


def test_canvas_for_a_pure_translation():
    (h, w), offset = _compute_canvas_size(_translation(-300, 40), (400, 500, 3), (400, 500, 3))
    assert (w, h) == (800, 440)
    assert offset == (300, 0)


def test_warp_places_both_images_and_masks_overlap():
    img_a = np.full((100, 200, 3), 50, np.uint8)
    img_b = np.full((100, 200, 3), 200, np.uint8)
    warped_a, canvas_b, (mask_a, mask_b, overlap), offset = warp_and_prepare(img_a, img_b, _translation(-150, 0))

    assert warped_a.shape == canvas_b.shape == (100, 350, 3)
    assert offset == (150, 0)
    assert mask_a[:, :200].all() and not mask_a[:, 200:].any()
    assert mask_b[:, 150:].all() and not mask_b[:, :150].any()
    assert overlap.sum() == 100 * 50


def test_black_pixels_inside_image_a_are_still_part_of_a():
    img_a = np.full((100, 200, 3), 120, np.uint8)
    img_a[40:60, 160:180] = 0  # a black patch inside the overlap
    img_b = np.full((100, 200, 3), 120, np.uint8)
    _, _, (mask_a, _, overlap), _ = warp_and_prepare(img_a, img_b, _translation(-150, 0))
    assert mask_a[40:60, 160:180].all()
    assert overlap[40:60, 160:180].all()


def test_distance_weighted_blend():
    img_a = np.full((100, 200, 3), 60, np.uint8)
    img_b = np.full((100, 200, 3), 180, np.uint8)
    warped_a, canvas_b, masks, _ = warp_and_prepare(img_a, img_b, _translation(-150, 0))
    out = simple_blend(warped_a, canvas_b, *masks).astype(int)

    assert (out[:, :150] == 60).all()  # A only
    assert (out[:, 200:] == 180).all()  # B only
    row = out[50, 150:200, 0]
    assert (np.diff(row) >= 0).all()  # smooth ramp from A towards B
    assert row[0] < 90 and row[-1] > 150


def test_crop_black_borders():
    img = np.zeros((50, 60, 3), np.uint8)
    img[10:30, 5:25] = 255
    assert crop_black_borders(img).shape == (20, 20, 3)
    assert crop_black_borders(np.zeros((5, 5, 3), np.uint8)).shape == (5, 5, 3)


def test_matching_returns_consistent_pairs(views):
    img_a, img_b, H_true = views
    src, dst = detect_and_match(img_a, img_b)
    assert src.shape == dst.shape and src.shape[1] == 2
    assert len(src) > 50
    p = np.hstack([src, np.ones((len(src), 1))]) @ H_true.T
    err = np.linalg.norm(p[:, :2] / p[:, 2:3] - dst, axis=1)
    assert np.mean(err < 3) > 0.8  # most ratio-test matches are geometrically correct


def test_matching_with_featureless_image_returns_empty():
    blank = np.zeros((200, 200, 3), np.uint8)
    src, dst = detect_and_match(blank, blank)
    assert src.shape == dst.shape == (0, 2)


def test_estimated_homography_matches_ground_truth(views):
    img_a, img_b, H_true = views
    src, dst = detect_and_match(img_a, img_b)
    H, inliers = ransac_homography(src, dst, rng=np.random.default_rng(0))
    assert corner_error(H, H_true) < 3.0
    H_cv, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    assert corner_error(H, H_true) < corner_error(H_cv, H_true) + 1.5


def test_stitch_end_to_end(views):
    img_a, img_b, _ = views
    pano = stitch(img_a, img_b, rng=np.random.default_rng(0))
    w, h = VIEW_SIZE
    assert pano.dtype == np.uint8
    assert pano.shape[1] > w + 400  # wider than either input
    assert h <= pano.shape[0] <= h + 150


def test_stitch_rejects_non_overlapping_images():
    blank = np.zeros((200, 200, 3), np.uint8)
    with pytest.raises(ValueError, match="matches"):
        stitch(blank, blank)
