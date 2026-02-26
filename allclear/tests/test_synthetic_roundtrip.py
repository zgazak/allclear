"""End-to-end test: generate → detect → solve → verify."""

import numpy as np
import pytest
from astropy.table import Table

from allclear.projection import CameraModel, ProjectionType
from allclear.synthetic import generate_synthetic_frame
from allclear.detection import detect_stars
from allclear.solver import AllSkySolver


class TestSyntheticRoundtrip:
    """Full pipeline roundtrip on synthetic data."""

    def test_roundtrip_equidistant(self):
        """Blind solve should recover camera params from a synthetic frame."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        # Create catalog-like stars
        rng = np.random.default_rng(42)
        n = 200
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(15, 88, n)
        vmag = np.concatenate([
            rng.uniform(-0.5, 2.0, 30),  # bright
            rng.uniform(2.0, 4.0, 70),   # medium
            rng.uniform(4.0, 5.5, 100),  # faint
        ])
        cat = Table({"az_deg": az_deg, "alt_deg": alt_deg, "vmag": vmag,
                      "vmag_extinct": vmag + 0.1})  # slight extinction
        cat.sort("vmag_extinct")

        image, truth = generate_synthetic_frame(
            camera_model=true_model,
            star_table=cat,
            sky_background=200,
            read_noise=10,
            flux_scale=1e5,
            psf_sigma=2.5,
            seed=42,
        )

        det = detect_stars(image, fwhm=5.0, threshold_sigma=3.0, n_brightest=150)
        assert len(det) >= 20, f"Only {len(det)} stars detected"

        solver = AllSkySolver(nx=3096, ny=2080)
        result = solver.solve(det, cat, initial_f=750.0)

        rm = result.camera_model
        assert result.n_matched >= 10, f"Only {result.n_matched} matches"

        # Check recovered parameters
        assert abs(rm.cx - true_model.cx) / true_model.cx < 0.01, (
            f"cx: {rm.cx:.1f} vs {true_model.cx:.1f}")
        assert abs(rm.cy - true_model.cy) / true_model.cy < 0.01, (
            f"cy: {rm.cy:.1f} vs {true_model.cy:.1f}")
        assert abs(rm.f - true_model.f) / true_model.f < 0.01, (
            f"f: {rm.f:.1f} vs {true_model.f:.1f}")

        # For a zenith-pointing camera, az0 and rho are degenerate
        # (both rotate about the optical axis). Check the combined rotation.
        true_total = (true_model.az0 + true_model.rho) % (2 * np.pi)
        rec_total = (rm.az0 + rm.rho) % (2 * np.pi)
        d_rot = abs((rec_total - true_total + np.pi) % (2 * np.pi) - np.pi)
        assert d_rot < np.radians(2.0), (
            f"combined rotation: {np.degrees(rec_total):.2f} vs "
            f"{np.degrees(true_total):.2f} (diff {np.degrees(d_rot):.2f} deg)")

        assert result.rms_residual < 5.0, (
            f"RMS residual: {result.rms_residual:.2f} pixels")
