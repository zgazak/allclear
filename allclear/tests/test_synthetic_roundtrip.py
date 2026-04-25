"""End-to-end test: generate → detect → solve → verify."""

import numpy as np
import pytest
from astropy.table import Table

from allclear.projection import CameraModel, ProjectionType
from allclear.synthetic import generate_synthetic_frame
from allclear.detection import detect_stars
from allclear.solver import fast_solve


class TestSyntheticRoundtrip:
    """Full pipeline roundtrip on synthetic data."""

    def test_roundtrip_equidistant(self):
        """Solve should recover matches from a synthetic frame."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        rng = np.random.default_rng(42)
        n = 200
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(15, 88, n)
        vmag = np.concatenate([
            rng.uniform(-0.5, 2.0, 30),
            rng.uniform(2.0, 4.0, 70),
            rng.uniform(4.0, 5.5, 100),
        ])
        cat = Table({
            "az_deg": az_deg, "alt_deg": alt_deg,
            "vmag": vmag, "vmag_expected": vmag + 0.1,
        })
        cat.sort("vmag_expected")

        image, truth = generate_synthetic_frame(
            camera_model=true_model, star_table=cat,
            sky_background=200, read_noise=10,
            flux_scale=5e6, psf_sigma=2.5, seed=42,
        )

        det = detect_stars(image, fwhm=5.0, threshold_sigma=3.0,
                           n_brightest=150)
        assert len(det) >= 20, f"Only {len(det)} stars detected"

        result = fast_solve(image, det, cat, true_model,
                            refine=True, guided=True)

        assert result.n_matched >= 30, f"Only {result.n_matched} matches"
        assert result.rms_residual < 3.0, (
            f"RMS residual: {result.rms_residual:.2f} pixels")

    def test_roundtrip_with_clouds(self):
        """Stars behind clouds should still be detectable where clear."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        rng = np.random.default_rng(99)
        n = 200
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(15, 88, n)
        vmag = np.concatenate([
            rng.uniform(-0.5, 2.0, 30),
            rng.uniform(2.0, 4.0, 70),
            rng.uniform(4.0, 5.5, 100),
        ])
        cat = Table({
            "az_deg": az_deg, "alt_deg": alt_deg,
            "vmag": vmag, "vmag_expected": vmag + 0.1,
        })
        cat.sort("vmag_expected")

        cloud_patches = [
            {"az_deg": 90, "alt_deg": 60, "radius_deg": 30, "opacity": 0.95},
        ]

        image, truth = generate_synthetic_frame(
            camera_model=true_model, star_table=cat,
            sky_background=200, read_noise=10,
            flux_scale=5e6, psf_sigma=2.5, seed=99,
            cloud_patches=cloud_patches,
        )

        det = detect_stars(image, fwhm=5.0, threshold_sigma=3.0,
                           n_brightest=150)

        result = fast_solve(image, det, cat, true_model,
                            refine=False, guided=True)

        # Should still match stars in the clear regions
        assert result.n_matched >= 8, f"Only {result.n_matched} matches"
        assert result.rms_residual < 5.0
