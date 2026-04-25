"""Tests for the ObscurationMask primitive."""

import json
from pathlib import Path

import numpy as np
import pytest

from allclear.obscuration import (
    DEFAULT_ALT_STEP_DEG,
    DEFAULT_AZ_STEP_DEG,
    OCCLUDED_THRESHOLD,
    ObscurationMask,
    build_from_observations,
)
from allclear.projection import CameraModel, ProjectionType


@pytest.fixture
def empty_mask():
    return ObscurationMask.empty()


@pytest.fixture
def camera():
    return CameraModel(
        cx=1500.0, cy=1040.0, az0=0.0, alt0=np.pi / 2,
        rho=0.0, f=1000.0, proj_type=ProjectionType.EQUIDISTANT,
    )


def test_empty_mask_is_fully_visible(empty_mask):
    assert np.all(empty_mask.weight == 1.0)
    assert empty_mask.query(180.0, 45.0) == pytest.approx(1.0)
    assert empty_mask.is_visible(180.0, 45.0)


def test_below_horizon_is_obscured(empty_mask):
    # Altitudes below the grid bottom are treated as obscured
    assert empty_mask.query(0.0, -30.0) == pytest.approx(0.0)
    assert not empty_mask.is_visible(0.0, -30.0)


def test_query_vectorized(empty_mask):
    az = np.array([0.0, 90.0, 180.0, 270.0])
    alt = np.array([10.0, 30.0, 60.0, 85.0])
    w = empty_mask.query(az, alt)
    assert w.shape == az.shape
    assert np.all(w == 1.0)


def test_az_wraps_around(empty_mask):
    # Placing an obstruction in the bin covering az≈350 — query at az=-10
    # (= 350 mod 360) should pick it up.
    ai = int(np.digitize(20.0, empty_mask.alt_edges_deg) - 1)
    zi = int(np.digitize(350.5, empty_mask.az_edges_deg) - 1)
    empty_mask.weight[ai, zi] = 0.0
    assert empty_mask.query(-9.5, 20.0) == pytest.approx(0.0)
    assert empty_mask.query(350.5, 20.0) == pytest.approx(0.0)
    # But az=10 deg should still be visible
    assert empty_mask.query(10.0, 20.0) == pytest.approx(1.0)


def test_from_camera_marks_horizon_and_outside_frame(camera):
    mask = ObscurationMask.from_camera(
        camera, image_shape=(2080, 3096), horizon_alt_deg=3.0,
    )
    # Zenith — visible
    assert mask.query(0.0, 89.0) == pytest.approx(1.0)
    # Below horizon cut
    assert mask.query(0.0, 0.0) == pytest.approx(0.0)
    # Just above horizon, but check something inside the circle is visible.
    # At alt 45 deg, any azimuth is well inside a 1000px-focal fisheye.
    assert mask.query(90.0, 45.0) == pytest.approx(1.0)


def test_project_to_pixel_shape_and_center(camera):
    # Image sized so the corners lie outside the f*π/2 fisheye radius.
    # With f=1000, full-hemisphere radius is ~1571 px.  A 2080x3096 image
    # centered at (1500, 1040) has corners ~1905 px from the optical
    # axis, which maps to theta > π/2 — below-horizon / invalid.
    shape = (2080, 3096)
    # Mask a whole altitude band (alt ~ 20°) and verify the projected
    # pixel mask shows those pixels obscured.
    mask = ObscurationMask.empty()
    alt_band = (mask.alt_edges_deg[:-1] >= 19) & (mask.alt_edges_deg[:-1] < 22)
    mask.weight[alt_band, :] = 0.0

    pixel = mask.project_to_pixel(camera, shape)
    assert pixel.shape == shape
    # Corners lie outside the fisheye circle — always 0
    assert pixel[0, 0] == 0.0
    assert pixel[-1, -1] == 0.0
    # A pixel near where alt~20° should show an obscured weight.  With
    # equidistant projection, alt=20° → theta=70° → r = 70*f*π/180 ≈ 1222 px.
    r = np.radians(70.0) * camera.f
    px_x = int(camera.cx + r)
    px_y = int(camera.cy)
    if 0 <= px_x < shape[1] and 0 <= px_y < shape[0]:
        assert pixel[px_y, px_x] == 0.0


def test_pixel_mask_bool(camera):
    mask = ObscurationMask.from_camera(camera, (300, 300))
    boolean = mask.project_to_pixel_mask(camera, (300, 300))
    assert boolean.dtype == bool
    assert boolean.shape == (300, 300)
    # Corners obscured (outside fisheye)
    assert boolean[0, 0]


def test_roundtrip_json(tmp_path):
    mask = ObscurationMask.empty()
    # Put a few measured-ish values, including NaN
    mask.weight[0, 0] = 0.2
    mask.weight[10, 20] = np.nan
    mask.n_visits = np.ones(mask.weight.shape, dtype=np.int64)
    mask.n_frames = 123

    path = tmp_path / "mask.json"
    mask.save(path)
    loaded = ObscurationMask.load(path)

    assert np.allclose(loaded.alt_edges_deg, mask.alt_edges_deg)
    assert np.allclose(loaded.az_edges_deg, mask.az_edges_deg)
    assert loaded.weight[0, 0] == pytest.approx(0.2)
    assert np.isnan(loaded.weight[10, 20])
    assert loaded.n_frames == 123
    assert loaded.n_visits is not None


def test_combined_with_takes_min():
    a = ObscurationMask.empty()
    b = ObscurationMask.empty()
    a.weight[5, 5] = 0.8
    b.weight[5, 5] = 0.2
    a.weight[6, 6] = 0.3
    b.weight[6, 6] = np.nan  # NaN treated as 1
    c = a.combined_with(b)
    assert c.weight[5, 5] == pytest.approx(0.2)
    assert c.weight[6, 6] == pytest.approx(0.3)


def test_combined_with_grid_mismatch_raises():
    a = ObscurationMask.empty(az_step_deg=2.0, alt_step_deg=2.0)
    b = ObscurationMask.empty(az_step_deg=5.0, alt_step_deg=5.0)
    with pytest.raises(ValueError):
        a.combined_with(b)


def test_radial_response_averages_azimuth():
    mask = ObscurationMask.empty()
    # Half the sky (az < 180) fully obscured at alt bin 10
    mid_alt = 10
    half_az = len(mask.az_edges_deg) // 2
    mask.weight[mid_alt, :half_az] = 0.0
    alt_c, mean_w, std_w = mask.radial_response()
    assert alt_c.shape == mean_w.shape == std_w.shape
    assert 0.4 < mean_w[mid_alt] < 0.6
    # Other altitudes untouched
    assert mean_w[0] == pytest.approx(1.0)


def test_build_from_observations_applies_gates():
    rng = np.random.default_rng(0)
    n = 50000
    az = rng.uniform(0, 360, n)
    alt = rng.uniform(10, 85, n)
    vmag = rng.uniform(2.0, 6.0, n)
    clear_frac = rng.uniform(0.5, 1.0, n)  # all clear
    # "Tree" region: never detected in az band [100, 120] AND alt < 30
    in_tree = (az >= 100) & (az <= 120) & (alt < 30)
    # Other regions: always detected (clear sky, star visible)
    detected = np.where(in_tree, 0, 1)

    mask = build_from_observations(
        az_deg=az, alt_deg=alt, detected=detected,
        clear_fraction=clear_frac, vmag=vmag,
        clear_gate=0.5, vmag_min=1.5, vmag_max=6.5,
        min_visits=3, az_step_deg=10.0, alt_step_deg=5.0,
    )
    # Tree region has low weight, clear region at full weight
    tree_w = mask.query(110.0, 20.0)
    clear_w = mask.query(200.0, 60.0)
    assert tree_w < 0.2
    assert clear_w == pytest.approx(1.0)
    assert mask.n_visits is not None
    # Most bins populated
    assert int((mask.n_visits > 0).sum()) > 200


def test_build_from_observations_min_visits_threshold():
    # Only 2 observations per bin — everything should be NaN
    az = np.array([10.0, 10.0])
    alt = np.array([45.0, 45.0])
    det = np.array([1, 0])
    cf = np.array([0.9, 0.9])
    vm = np.array([3.0, 3.0])
    mask = build_from_observations(
        az_deg=az, alt_deg=alt, detected=det, clear_fraction=cf,
        vmag=vm, min_visits=8,
    )
    # Not enough data — weight is NaN (unknown)
    assert np.isnan(mask.weight).all()


def test_query_treats_nan_as_visible():
    mask = ObscurationMask.empty()
    mask.weight[:] = np.nan
    # Unknown bins should default to visible
    assert mask.query(120.0, 40.0) == pytest.approx(1.0)


def test_build_extrapolates_below_lowest_data_as_obscured():
    """Bins below the lowest fitted altitude should be set to weight=0
    so the rendered obscuration ring connects to the horizon instead of
    floating mid-sky.
    """
    # Place observations at alt >= 20 only (nothing below)
    rng = np.random.default_rng(1)
    n = 20000
    az = rng.uniform(0, 360, n)
    alt = rng.uniform(20, 80, n)
    vmag = rng.uniform(2.0, 6.0, n)
    cf = np.full(n, 0.9)
    det = np.ones(n, dtype=np.int32)  # all detected

    mask = build_from_observations(
        az_deg=az, alt_deg=alt, detected=det, clear_fraction=cf,
        vmag=vmag, min_visits=3, az_step_deg=10.0, alt_step_deg=5.0,
    )
    # A direction below the data range must return weight 0 (obscured),
    # not NaN-defaulted-to-1.
    assert mask.query(120.0, 8.0) == pytest.approx(0.0)
    # Within the data range, fully detected stars give weight 1.
    assert mask.query(120.0, 50.0) == pytest.approx(1.0)
