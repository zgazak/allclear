"""Integration tests for ObscurationMask plumbing through the pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from allclear.instrument import InstrumentModel
from allclear.obscuration import ObscurationMask
from allclear.projection import CameraModel, ProjectionType


@pytest.fixture
def camera():
    return CameraModel(
        cx=1000.0, cy=1000.0, az0=0.0, alt0=np.pi / 2,
        rho=0.0, f=700.0, proj_type=ProjectionType.EQUIDISTANT,
    )


def test_instrument_model_saves_loads_obscuration_sidecar(tmp_path, camera):
    inst = InstrumentModel.from_camera_model(
        camera, site_lat=20.7, site_lon=-156.4,
        image_width=2000, image_height=2000,
    )
    mask = ObscurationMask.from_camera(
        camera, image_shape=(2000, 2000))
    mask.weight[5, 5] = 0.0  # plant a marker
    inst.obscuration = mask

    model_path = tmp_path / "model.json"
    inst.save(model_path)

    # Sidecar exists at the expected path
    sidecar = tmp_path / "model_obscuration.json"
    assert sidecar.exists()

    # Load back — sidecar auto-loaded
    loaded = InstrumentModel.load(model_path)
    assert loaded.obscuration is not None
    assert loaded.obscuration.weight[5, 5] == pytest.approx(0.0)


def test_instrument_model_load_without_sidecar(tmp_path, camera):
    inst = InstrumentModel.from_camera_model(
        camera, site_lat=20.7, site_lon=-156.4)
    model_path = tmp_path / "model.json"
    inst.save(model_path)
    loaded = InstrumentModel.load(model_path)
    assert loaded.obscuration is None


def test_fast_solve_accepts_obscuration_and_filters_catalog(camera):
    """Catalog stars in obscured sky must be excluded from matching."""
    from allclear.solver import fast_solve

    # Synthetic catalog: 3 stars, one in an obscured direction
    cat = Table()
    cat["az_deg"] = np.array([0.0, 90.0, 180.0])
    cat["alt_deg"] = np.array([70.0, 70.0, 70.0])
    cat["vmag"] = np.array([3.0, 3.0, 3.0])
    cat["vmag_expected"] = np.array([3.2, 3.2, 3.2])
    cat["airmass"] = np.array([1.06, 1.06, 1.06])

    # Obscuration blocks direction (az=90, alt=70)
    mask = ObscurationMask.empty()
    ai = int(np.digitize(70.5, mask.alt_edges_deg) - 1)
    zi = int(np.digitize(90.5, mask.az_edges_deg) - 1)
    mask.weight[ai, zi] = 0.0

    # Cheap image (no real peaks) — fast_solve will find 0 matches, that's
    # fine; we only care that n_expected respects the obscuration mask.
    image = np.zeros((1500, 1500), dtype=np.float32) + 100
    det = Table()
    det["x"] = np.array([], dtype=np.float64)
    det["y"] = np.array([], dtype=np.float64)
    det["flux"] = np.array([], dtype=np.float64)

    result_no_mask = fast_solve(image, det, cat, camera, refine=False)
    result_with_mask = fast_solve(
        image, det, cat, camera, refine=False, obscuration=mask,
    )
    # One fewer expected star when the mask blocks a direction
    assert result_with_mask.n_expected == result_no_mask.n_expected - 1


def test_compute_transmission_honors_obscuration():
    """Unmatched catalog stars in obscured sky are not probed as clouds."""
    from allclear.transmission import compute_transmission

    camera = CameraModel(
        cx=500.0, cy=500.0, az0=0.0, alt0=np.pi / 2,
        rho=0.0, f=400.0, proj_type=ProjectionType.EQUIDISTANT,
    )

    # 2 catalog stars — one gets matched, the other is unmatched.
    cat = Table()
    cat["az_deg"] = np.array([0.0, 180.0])
    cat["alt_deg"] = np.array([45.0, 45.0])
    cat["vmag_expected"] = np.array([3.0, 3.0])

    # Only the first star is matched (star 0)
    det = Table()
    det["x"] = np.array([500.0 + 400.0 * np.pi / 4 * 0.0])  # at center-ish
    det["y"] = np.array([500.0 - 400.0 * np.pi / 4])
    det["flux"] = np.array([10000.0])
    matched_pairs = [(0, 0)]

    image = np.zeros((1000, 1000), dtype=np.float32) + 50
    # Place a bright "star" around the detection position so peak extraction works
    image[int(det["y"][0]) - 2:int(det["y"][0]) + 3,
          int(det["x"][0]) - 2:int(det["x"][0]) + 3] = 10000

    # Build a mask that blocks az=180, alt=45 (the unmatched star's location)
    mask = ObscurationMask.empty()
    ai = int(np.digitize(45.5, mask.alt_edges_deg) - 1)
    zi = int(np.digitize(180.5, mask.az_edges_deg) - 1)
    mask.weight[ai, zi] = 0.0

    az_u, alt_u, trans_u, _ = compute_transmission(
        det, cat, matched_pairs, camera, image=image,
    )
    az_m, alt_m, trans_m, _ = compute_transmission(
        det, cat, matched_pairs, camera, image=image, obscuration=mask,
    )

    # Without mask: the unmatched obscured star is probed and added to
    # the output.  With mask: it is skipped.
    assert len(trans_m) == len(trans_u) - 1


def test_detect_stars_respects_pixel_mask():
    """Smoke test: detect_stars skips masked pixels."""
    from allclear.detection import detect_stars

    # Image with a single bright star at (100, 100) and another at (200, 200)
    image = np.random.default_rng(0).normal(100, 5, (300, 300))
    image[98:103, 98:103] = 5000
    image[198:203, 198:203] = 5000

    mask = np.zeros(image.shape, dtype=bool)
    mask[190:210, 190:210] = True  # mask out the second star

    det = detect_stars(image, fwhm=3.0, threshold_sigma=3.0,
                        n_brightest=10, mask=mask, box_size=50)

    # Both stars detected without mask
    det_all = detect_stars(image, fwhm=3.0, threshold_sigma=3.0,
                            n_brightest=10, box_size=50)
    assert len(det) < len(det_all) or all(
        not (190 <= x <= 210 and 190 <= y <= 210)
        for x, y in zip(det["x"], det["y"])
    )
