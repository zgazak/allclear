"""Tests for the all-sky camera solver."""

import numpy as np
import pytest
from astropy.table import Table

from allclear.projection import CameraModel, ProjectionType
from allclear.solver import AllSkySolver


class TestRefine:
    """Test that refine() recovers known camera parameters."""

    def test_refine_recovers_params(self):
        """Given a perturbed initial guess and correct matches, refine should
        recover the true model."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.5, alt0=np.pi / 2 - 0.02,
            rho=0.3, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        # Generate fake catalog stars
        rng = np.random.default_rng(42)
        n = 80
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(20, 85, n)
        cat = Table({"az_deg": az_deg, "alt_deg": alt_deg,
                      "vmag_extinct": rng.uniform(1, 5, n)})

        # Project through true model to get "detected" positions
        az_rad = np.radians(az_deg)
        alt_rad = np.radians(alt_deg)
        x_true, y_true = true_model.sky_to_pixel(az_rad, alt_rad)

        # Add small noise
        x_det = x_true + rng.normal(0, 0.5, n)
        y_det = y_true + rng.normal(0, 0.5, n)

        # Filter to in-frame
        mask = ((x_det >= 0) & (x_det < 3096) &
                (y_det >= 0) & (y_det < 2080))
        det = Table({"x": x_det[mask], "y": y_det[mask],
                      "flux": rng.uniform(1000, 10000, np.sum(mask))})

        # Perturbed initial model
        init_model = CameraModel(
            cx=1560, cy=1050, az0=0.55, alt0=np.pi / 2 - 0.01,
            rho=0.35, f=740.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        solver = AllSkySolver()
        result = solver.refine(det, cat, init_model, max_dist=30.0)

        rm = result.camera_model
        assert abs(rm.cx - true_model.cx) < 5.0
        assert abs(rm.cy - true_model.cy) < 5.0
        assert abs(rm.f - true_model.f) / true_model.f < 0.02
        assert result.rms_residual < 3.0
        assert result.n_matched >= 20


class TestCoarseSolve:
    """Test coarse solve on synthetic data."""

    def test_coarse_solve_synthetic(self):
        """Coarse solve should find a reasonable initial model."""
        true_model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )

        rng = np.random.default_rng(42)
        n = 100
        az_deg = rng.uniform(0, 360, n)
        alt_deg = rng.uniform(25, 85, n)
        vmag = rng.uniform(0, 5, n)
        cat = Table({"az_deg": az_deg, "alt_deg": alt_deg,
                      "vmag_extinct": vmag})
        cat.sort("vmag_extinct")

        az_rad = np.radians(az_deg)
        alt_rad = np.radians(alt_deg)
        x_true, y_true = true_model.sky_to_pixel(az_rad, alt_rad)

        x_det = x_true + rng.normal(0, 1.0, n)
        y_det = y_true + rng.normal(0, 1.0, n)

        mask = ((x_det >= 0) & (x_det < 3096) &
                (y_det >= 0) & (y_det < 2080))
        det = Table({"x": x_det[mask], "y": y_det[mask],
                      "flux": 10**(4 - 0.4 * vmag[mask])})
        det.sort("flux")
        det.reverse()

        solver = AllSkySolver()
        model, matches = solver.coarse_solve(det, cat, initial_f=750.0)

        assert model is not None
        assert len(matches) >= 4
        # Focal length should be in the right ballpark
        assert abs(model.f - 750.0) / 750.0 < 0.3
