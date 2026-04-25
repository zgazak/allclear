"""Tests for the fast solver (known-model matching and refinement)."""

import numpy as np
import pytest
from astropy.table import Table

from allclear.projection import CameraModel, ProjectionType
from allclear.synthetic import generate_synthetic_frame
from allclear.detection import detect_stars
from allclear.solver import fast_solve


class TestFastSolve:
    """Test fast_solve with guided matching on synthetic data."""

    @pytest.fixture
    def synthetic_scene(self):
        """Generate a synthetic frame with known camera model."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.5, alt0=np.pi / 2 - 0.02,
            rho=0.3, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        rng = np.random.default_rng(42)
        n = 200
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(15, 85, n)
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
                           n_brightest=200)

        return true_model, image, det, cat

    def test_exact_model_matches_stars(self, synthetic_scene):
        """With the exact model, fast_solve should find many matches."""
        true_model, image, det, cat = synthetic_scene

        result = fast_solve(image, det, cat, true_model,
                            refine=False, guided=True)

        assert result.n_matched >= 30
        assert result.rms_residual < 3.0

    def test_refine_recovers_perturbed_model(self, synthetic_scene):
        """Given a slightly perturbed model, refine should recover accuracy."""
        true_model, image, det, cat = synthetic_scene

        perturbed = CameraModel(
            cx=true_model.cx + 8,
            cy=true_model.cy - 5,
            az0=true_model.az0 + np.radians(1.0),
            alt0=true_model.alt0 - np.radians(0.5),
            rho=true_model.rho + np.radians(0.8),
            f=true_model.f * 1.01,
            proj_type=true_model.proj_type,
        )

        result = fast_solve(image, det, cat, perturbed,
                            match_radius=15.0, refine=True, guided=True)

        assert result.n_matched >= 20
        assert result.rms_residual < 6.0
        # Refined model should be closer to truth than the perturbed input
        assert abs(result.camera_model.f - true_model.f) < abs(perturbed.f - true_model.f)
