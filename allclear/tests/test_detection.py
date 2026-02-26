"""Tests for star detection on synthetic frames."""

import numpy as np
import pytest
from astropy.table import Table

from allclear.projection import CameraModel, ProjectionType
from allclear.synthetic import generate_synthetic_frame
from allclear.detection import detect_stars


class TestDetectSyntheticStars:
    """Detect planted stars in synthetic frames."""

    def test_recover_bright_stars(self):
        """Detection should recover >80% of bright (vmag < 3) in-frame stars."""
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )
        # Plant known stars
        rng = np.random.default_rng(123)
        n_stars = 100
        az = rng.uniform(0, 360, n_stars)
        alt = rng.uniform(20, 85, n_stars)
        vmag = rng.uniform(0, 3.0, n_stars)  # bright stars
        stars = Table({"az_deg": az, "alt_deg": alt, "vmag": vmag})

        image, truth = generate_synthetic_frame(
            camera_model=model,
            star_table=stars,
            sky_background=200,
            read_noise=10,
            flux_scale=1e5,
            psf_sigma=2.5,
            seed=42,
        )

        sources = detect_stars(image, fwhm=5.0, threshold_sigma=3.0,
                               n_brightest=300)

        # Match detected to truth (within 5 pixels)
        in_frame = truth[truth["in_frame"]]
        matched = 0
        for row in in_frame:
            dx = sources["x"] - row["x_true"]
            dy = sources["y"] - row["y_true"]
            dist = np.sqrt(dx**2 + dy**2)
            if np.any(dist < 5.0):
                matched += 1

        recovery_rate = matched / len(in_frame)
        assert recovery_rate > 0.80, (
            f"Recovery rate {recovery_rate:.2%} < 80% "
            f"({matched}/{len(in_frame)})"
        )

    def test_no_false_positives_in_empty_region(self):
        """In a blank sky region, detections should be minimal."""
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )
        # Only a few bright stars in one quadrant
        stars = Table({
            "az_deg": [0.0, 10.0, 20.0],
            "alt_deg": [80.0, 75.0, 70.0],
            "vmag": [1.0, 1.5, 2.0],
        })
        image, truth = generate_synthetic_frame(
            camera_model=model,
            star_table=stars,
            sky_background=200,
            read_noise=10,
            flux_scale=1e5,
            seed=99,
        )
        sources = detect_stars(image, fwhm=5.0, threshold_sigma=5.0,
                               n_brightest=50)
        # Should detect roughly the 3 planted stars (maybe a few noise peaks)
        assert len(sources) < 20, f"Too many detections: {len(sources)}"

    def test_detection_positions_accurate(self):
        """Detected positions should be within ~1 pixel of truth for bright stars."""
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )
        stars = Table({
            "az_deg": [0.0, 90.0, 180.0, 270.0, 45.0],
            "alt_deg": [60.0, 60.0, 60.0, 60.0, 80.0],
            "vmag": [0.0, 0.5, 1.0, 0.5, 0.0],
        })
        image, truth = generate_synthetic_frame(
            camera_model=model,
            star_table=stars,
            sky_background=200,
            read_noise=5,
            flux_scale=2e5,
            psf_sigma=2.5,
            seed=77,
        )
        sources = detect_stars(image, fwhm=5.0, threshold_sigma=3.0)

        in_frame = truth[truth["in_frame"]]
        for row in in_frame:
            dx = sources["x"] - row["x_true"]
            dy = sources["y"] - row["y_true"]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = np.min(dist)
            assert min_dist < 2.0, (
                f"Star at ({row['x_true']:.1f}, {row['y_true']:.1f}) "
                f"not matched within 2 px (closest: {min_dist:.2f})"
            )
