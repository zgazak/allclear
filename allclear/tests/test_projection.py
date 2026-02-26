"""Tests for camera projection model."""

import numpy as np
import pytest

from allclear.projection import CameraModel, ProjectionType


ALL_PROJECTIONS = list(ProjectionType)


class TestForwardInverseRoundtrip:
    """sky_to_pixel then pixel_to_sky should recover original coordinates."""

    @pytest.mark.parametrize("proj_type", ALL_PROJECTIONS)
    def test_roundtrip_grid(self, proj_type):
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=proj_type,
        )
        # Grid of sky positions
        az = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        alt = np.linspace(np.radians(15), np.radians(85), 8)
        az_grid, alt_grid = np.meshgrid(az, alt)
        az_flat = az_grid.ravel()
        alt_flat = alt_grid.ravel()

        x, y = model.sky_to_pixel(az_flat, alt_flat)
        az_rec, alt_rec = model.pixel_to_sky(x, y)

        # Normalize azimuth difference to [-pi, pi]
        daz = (az_rec - az_flat + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(daz, 0, atol=1e-6)
        np.testing.assert_allclose(alt_rec, alt_flat, atol=1e-6)

    @pytest.mark.parametrize("proj_type", ALL_PROJECTIONS)
    def test_zenith_maps_to_center(self, proj_type):
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=proj_type,
        )
        x, y = model.sky_to_pixel(0.0, np.pi / 2)
        np.testing.assert_allclose(x, 1548.0, atol=1e-8)
        np.testing.assert_allclose(y, 1040.0, atol=1e-8)


class TestDistortion:
    """Test distortion application and inversion."""

    @pytest.mark.parametrize("proj_type", ALL_PROJECTIONS)
    def test_distortion_roundtrip(self, proj_type):
        model = CameraModel(
            cx=1548, cy=1040, az0=0.3, alt0=np.pi / 2 - 0.05,
            rho=0.1, f=750.0, proj_type=proj_type,
            k1=1e-7, k2=1e-14,
        )
        az = np.array([0.5, 1.0, 2.0, 4.0, 5.5])
        alt = np.array([0.3, 0.5, 0.8, 1.0, 1.2])

        x, y = model.sky_to_pixel(az, alt)
        az_rec, alt_rec = model.pixel_to_sky(x, y)

        daz = (az_rec - az + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(daz, 0, atol=1e-5)
        np.testing.assert_allclose(alt_rec, alt, atol=1e-5)


class TestSymmetry:
    """Symmetry checks for zenith-pointing camera."""

    def test_opposite_azimuths_symmetric(self):
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )
        alt = np.radians(45)
        x1, y1 = model.sky_to_pixel(0.0, alt)
        x2, y2 = model.sky_to_pixel(np.pi, alt)

        # Both should be equidistant from center
        r1 = np.sqrt((x1 - 1548)**2 + (y1 - 1040)**2)
        r2 = np.sqrt((x2 - 1548)**2 + (y2 - 1040)**2)
        np.testing.assert_allclose(r1, r2, atol=1e-8)

    def test_four_cardinal_directions(self):
        model = CameraModel(
            cx=1548, cy=1040, az0=0.0, alt0=np.pi / 2,
            rho=0.0, f=750.0, proj_type=ProjectionType.EQUIDISTANT,
        )
        alt = np.radians(45)
        azimuths = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        radii = []
        for az in azimuths:
            x, y = model.sky_to_pixel(az, alt)
            radii.append(np.sqrt((x - 1548)**2 + (y - 1040)**2))
        np.testing.assert_allclose(radii, radii[0], atol=1e-8)


class TestParamArray:
    """Test parameter vector serialization."""

    @pytest.mark.parametrize("proj_type", ALL_PROJECTIONS)
    def test_roundtrip(self, proj_type):
        model = CameraModel(
            cx=1500, cy=1000, az0=0.5, alt0=1.4,
            rho=0.1, f=800.0, proj_type=proj_type,
            k1=1e-7, k2=2e-14,
        )
        params = model.get_params_array()
        model2 = CameraModel.from_params_array(params, proj_type=proj_type)
        np.testing.assert_allclose(model2.get_params_array(), params)


class TestRotatedCamera:
    """Test with non-trivial az0 and rho."""

    @pytest.mark.parametrize("proj_type", ALL_PROJECTIONS)
    def test_rotated_roundtrip(self, proj_type):
        model = CameraModel(
            cx=1548, cy=1040, az0=1.2, alt0=np.pi / 2 - 0.1,
            rho=0.5, f=750.0, proj_type=proj_type,
        )
        az = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        alt = np.full_like(az, np.radians(50))

        x, y = model.sky_to_pixel(az, alt)
        az_rec, alt_rec = model.pixel_to_sky(x, y)

        daz = (az_rec - az + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(daz, 0, atol=1e-6)
        np.testing.assert_allclose(alt_rec, alt, atol=1e-6)
