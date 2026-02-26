"""Camera projection model for all-sky lenses."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ProjectionType(Enum):
    EQUIDISTANT = "equidistant"      # r = f * theta
    EQUISOLID = "equisolid"          # r = 2f * sin(theta/2)
    STEREOGRAPHIC = "stereographic"  # r = 2f * tan(theta/2)
    ORTHOGRAPHIC = "orthographic"    # r = f * sin(theta)


def _theta_to_r(theta, f, proj_type):
    """Map zenith angle theta to radial distance r for given projection."""
    if proj_type == ProjectionType.EQUIDISTANT:
        return f * theta
    elif proj_type == ProjectionType.EQUISOLID:
        return 2.0 * f * np.sin(theta / 2.0)
    elif proj_type == ProjectionType.STEREOGRAPHIC:
        return 2.0 * f * np.tan(theta / 2.0)
    elif proj_type == ProjectionType.ORTHOGRAPHIC:
        return f * np.sin(theta)
    raise ValueError(f"Unknown projection type: {proj_type}")


def _r_to_theta(r, f, proj_type):
    """Inverse: radial distance r back to zenith angle theta."""
    r_f = r / f
    if proj_type == ProjectionType.EQUIDISTANT:
        return r_f
    elif proj_type == ProjectionType.EQUISOLID:
        return 2.0 * np.arcsin(np.clip(r_f / 2.0, -1, 1))
    elif proj_type == ProjectionType.STEREOGRAPHIC:
        return 2.0 * np.arctan(r_f / 2.0)
    elif proj_type == ProjectionType.ORTHOGRAPHIC:
        return np.arcsin(np.clip(r_f, -1, 1))
    raise ValueError(f"Unknown projection type: {proj_type}")


def _apply_distortion(r, k1, k2):
    """Apply radial distortion: r' = r * (1 + k1*r^2 + k2*r^4)."""
    r2 = r * r
    return r * (1.0 + k1 * r2 + k2 * r2 * r2)


def _invert_distortion(r_dist, k1, k2, iterations=20):
    """Invert radial distortion via Newton's method."""
    r = r_dist.copy() if hasattr(r_dist, 'copy') else np.float64(r_dist)
    for _ in range(iterations):
        r2 = r * r
        r4 = r2 * r2
        f_val = r * (1.0 + k1 * r2 + k2 * r4) - r_dist
        f_prime = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
        # Guard against zero derivative
        f_prime = np.where(np.abs(f_prime) < 1e-15, 1.0, f_prime)
        r = r - f_val / f_prime
    return r


@dataclass
class CameraModel:
    """All-sky camera geometric model.

    Parameters
    ----------
    cx, cy : float
        Optical center in pixel coordinates.
    az0 : float
        Azimuth of the camera boresight (radians). For a zenith-pointing
        camera this is the azimuth corresponding to the "up" direction
        in the image (the rotation of the camera).
    alt0 : float
        Altitude of boresight (radians). Typically ~pi/2 for zenith.
    rho : float
        Camera roll angle (radians). Rotation of the sensor about the
        optical axis.
    f : float
        Focal length in pixels (focal_mm / pixel_mm).
    proj_type : ProjectionType
        Lens projection model.
    k1 : float
        Radial distortion coefficient (3rd order).
    k2 : float
        Radial distortion coefficient (5th order).
    """

    cx: float = 1548.0
    cy: float = 1040.0
    az0: float = 0.0
    alt0: float = np.pi / 2
    rho: float = 0.0
    f: float = 750.0  # ~1.8mm / 0.0024mm
    proj_type: ProjectionType = ProjectionType.EQUIDISTANT
    k1: float = 0.0
    k2: float = 0.0

    # --- Parameter vector interface for optimizer ---

    @staticmethod
    def param_names():
        return ["cx", "cy", "az0", "alt0", "rho", "f", "k1", "k2"]

    def get_params_array(self):
        return np.array([self.cx, self.cy, self.az0, self.alt0,
                         self.rho, self.f, self.k1, self.k2])

    @classmethod
    def from_params_array(cls, params, proj_type=ProjectionType.EQUIDISTANT):
        return cls(
            cx=params[0], cy=params[1], az0=params[2], alt0=params[3],
            rho=params[4], f=params[5], proj_type=proj_type,
            k1=params[6], k2=params[7],
        )

    # --- Projection ---

    def sky_to_pixel(self, az, alt):
        """Project sky coordinates (az, alt in radians) to pixel (x, y).

        Parameters can be scalar or array. Returns (x, y) same shape as input.
        """
        az = np.asarray(az, dtype=np.float64)
        alt = np.asarray(alt, dtype=np.float64)

        # Direction vectors in ground frame (x=E, y=N, z=Up)
        cos_alt = np.cos(alt)
        dx = cos_alt * np.sin(az)
        dy = cos_alt * np.cos(az)
        dz = np.sin(alt)

        # Rotate into camera frame.
        # Camera boresight points at (az0, alt0), roll = rho.
        # Build rotation: first rotate about z by -az0, then about x by
        # -(pi/2 - alt0), then about z by -rho.
        d_cam = self._rotate_to_camera(dx, dy, dz)

        # Camera frame: z_cam = optical axis, x_cam/y_cam = sensor plane
        xc, yc, zc = d_cam

        # Zenith angle from optical axis
        theta = np.arctan2(np.sqrt(xc**2 + yc**2), zc)
        phi = np.arctan2(xc, yc)  # angle in sensor plane

        # Radial distance
        r = _theta_to_r(theta, self.f, self.proj_type)

        # Apply distortion
        r_dist = _apply_distortion(r, self.k1, self.k2)

        # Pixel coordinates
        x = self.cx + r_dist * np.sin(phi)
        y = self.cy + r_dist * np.cos(phi)

        return x, y

    def pixel_to_sky(self, x, y):
        """Inverse projection: pixel (x, y) to sky (az, alt) in radians.

        Returns (az, alt). Points behind the camera get NaN.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        dx_pix = x - self.cx
        dy_pix = y - self.cy

        r_dist = np.sqrt(dx_pix**2 + dy_pix**2)
        phi = np.arctan2(dx_pix, dy_pix)

        # Invert distortion
        r = _invert_distortion(r_dist, self.k1, self.k2)

        # Invert radial projection
        theta = _r_to_theta(r, self.f, self.proj_type)

        # Direction in camera frame
        sin_theta = np.sin(theta)
        xc = sin_theta * np.sin(phi)
        yc = sin_theta * np.cos(phi)
        zc = np.cos(theta)

        # Rotate back to ground frame
        dx, dy, dz = self._rotate_to_ground(xc, yc, zc)

        az = np.arctan2(dx, dy) % (2 * np.pi)
        alt = np.arcsin(np.clip(dz, -1, 1))

        return az, alt

    def _rotate_to_camera(self, dx, dy, dz):
        """Rotate ground-frame direction to camera frame."""
        # Step 1: rotate about z-axis by -az0
        ca, sa = np.cos(-self.az0), np.sin(-self.az0)
        x1 = ca * dx - sa * dy
        y1 = sa * dx + ca * dy
        z1 = dz

        # Step 2: rotate about x-axis by -(pi/2 - alt0) to tilt boresight to z
        tilt = -(np.pi / 2 - self.alt0)
        ct, st = np.cos(tilt), np.sin(tilt)
        x2 = x1
        y2 = ct * y1 - st * z1
        z2 = st * y1 + ct * z1

        # Step 3: rotate about z-axis by -rho (camera roll)
        cr, sr = np.cos(-self.rho), np.sin(-self.rho)
        xc = cr * x2 - sr * y2
        yc = sr * x2 + cr * y2
        zc = z2

        return xc, yc, zc

    def _rotate_to_ground(self, xc, yc, zc):
        """Rotate camera-frame direction back to ground frame."""
        # Inverse of _rotate_to_camera: apply inverse rotations in reverse order

        # Step 3 inverse: rotate about z by +rho
        cr, sr = np.cos(self.rho), np.sin(self.rho)
        x2 = cr * xc - sr * yc
        y2 = sr * xc + cr * yc
        z2 = zc

        # Step 2 inverse: rotate about x by +(pi/2 - alt0)
        tilt = np.pi / 2 - self.alt0
        ct, st = np.cos(tilt), np.sin(tilt)
        x1 = x2
        y1 = ct * y2 - st * z2
        z1 = st * y2 + ct * z2

        # Step 1 inverse: rotate about z by +az0
        ca, sa = np.cos(self.az0), np.sin(self.az0)
        dx = ca * x1 - sa * y1
        dy = sa * x1 + ca * y1
        dz = z1

        return dx, dy, dz
