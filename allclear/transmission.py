"""Cloud transmission mapping from matched star photometry."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class TransmissionMap:
    """Gridded sky transmission map."""
    az_grid: np.ndarray     # (N_az,) in degrees
    alt_grid: np.ndarray    # (N_alt,) in degrees
    transmission: np.ndarray  # (N_alt, N_az)
    zeropoint: float

    def query(self, az_deg, alt_deg):
        """Look up interpolated transmission at a sky position.

        Parameters
        ----------
        az_deg, alt_deg : float
            Azimuth and altitude in degrees.

        Returns
        -------
        float
            Transmission value (0=opaque, 1=clear). NaN if outside grid.
        """
        if alt_deg < float(self.alt_grid[0]) or alt_deg > float(self.alt_grid[-1]):
            return float("nan")
        az_idx = int(np.argmin(np.abs(self.az_grid - (az_deg % 360))))
        alt_idx = int(np.argmin(np.abs(self.alt_grid - alt_deg)))
        return float(self.transmission[alt_idx, az_idx])

    def get_observability_mask(self, threshold=0.7):
        """Return boolean mask where transmission >= threshold."""
        return self.transmission >= threshold

    def to_dict(self):
        """Serialize to a JSON-compatible dict.

        Returns a compact representation with the grid axes and
        transmission values (rounded to 3 decimal places).
        """
        trans = self.transmission.copy()
        trans = np.where(np.isnan(trans), None, np.round(trans, 3))
        return {
            "az_grid_deg": self.az_grid.tolist(),
            "alt_grid_deg": self.alt_grid.tolist(),
            "transmission": [[v for v in row] for row in trans],
            "zeropoint": round(self.zeropoint, 4),
        }

    def to_image(self, cmap="gray_r", vmin=0, vmax=1.2):
        """Render as a matplotlib figure."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        az_rad = np.radians(self.az_grid)
        r = 90.0 - self.alt_grid  # zenith distance
        AZ, R = np.meshgrid(az_rad, r)
        im = ax.pcolormesh(AZ, R, self.transmission, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_yticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"])
        ax.set_title("Sky Transmission")
        plt.colorbar(im, ax=ax, label="Transmission")
        return fig


def compute_transmission(det_table, cat_table, matched_pairs, camera_model,
                         image=None, image_shape=None,
                         reference_zeropoint=None,
                         probe_vmag_limit=5.5, probe_radius=8):
    """Compute per-star transmission from matched photometry.

    When ``image`` is provided, flux is re-measured using local background
    subtraction at each star position. This correctly handles bright
    obstructions (domes, telescopes) where the guided matcher may "match"
    stars against structure glow rather than real starlight.

    Unmatched in-frame catalog stars are also probed: if no point source
    is detected, they are marked as zero transmission.

    Parameters
    ----------
    det_table : Table
        Detected sources with columns: x, y, flux.
    cat_table : Table
        Catalog stars with columns: vmag_extinct.
    matched_pairs : list of (det_idx, cat_idx)
        Matched detection–catalog pairs.
    camera_model : CameraModel
        Solved camera model.
    image : ndarray (ny, nx), optional
        Raw image data. Enables local-background flux measurement for
        matched stars and pixel probing for unmatched catalog stars.
    image_shape : tuple (ny, nx), optional
        Fallback if ``image`` not provided — unmatched stars get trans=0.
    reference_zeropoint : float, optional
        Global photometric zeropoint from instrument-fit (clear-sky
        calibration). When provided, transmission is measured on an
        absolute scale. When None, a per-frame zeropoint is computed
        (relative scale only).
    probe_vmag_limit : float
        Magnitude limit for probing unmatched stars (default 5.5).
    probe_radius : int
        Half-size of the box for peak/background measurement.

    Returns
    -------
    az_deg : ndarray
        Azimuth of each star (degrees).
    alt_deg : ndarray
        Altitude of each star (degrees).
    transmission : ndarray
        Transmission at each star location.
    zeropoint : float
        Instrumental zeropoint (reference if provided, else per-frame).
    """
    if len(matched_pairs) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0

    cat_vmag = np.array([float(cat_table["vmag_extinct"][ci]) for di, ci in matched_pairs])
    cat_az = np.array([float(cat_table["az_deg"][ci]) for di, ci in matched_pairs])
    cat_alt = np.array([float(cat_table["alt_deg"][ci]) for di, ci in matched_pairs])

    # Measure flux using forced photometry at model-predicted positions.
    # This avoids measuring flux at wrong-star positions in dense fields.
    if image is not None:
        r = probe_radius
        ny, nx = image.shape

        # Get predicted positions for each matched catalog star
        all_az_rad = np.radians(
            np.asarray(cat_table["az_deg"], dtype=np.float64))
        all_alt_rad = np.radians(
            np.asarray(cat_table["alt_deg"], dtype=np.float64))
        px_all, py_all = camera_model.sky_to_pixel(all_az_rad, all_alt_rad)

        det_flux = np.zeros(len(matched_pairs), dtype=np.float64)
        for k, (di, ci) in enumerate(matched_pairs):
            # Use MODEL-PREDICTED position, not matched detection position
            x = float(px_all[ci])
            y = float(py_all[ci])
            xi, yi = int(round(x)), int(round(y))
            if xi < r or xi >= nx - r or yi < r or yi >= ny - r:
                det_flux[k] = max(0, float(det_table["flux"][di]))
                continue
            box = image[yi - r:yi + r + 1, xi - r:xi + r + 1].astype(np.float64)
            # Local background from box edge pixels
            edge = np.concatenate([
                box[0, :], box[-1, :], box[1:-1, 0], box[1:-1, -1]
            ])
            local_bg = float(np.median(edge))
            peak = float(np.max(box))
            det_flux[k] = max(0, peak - local_bg)
    else:
        det_flux = np.array([float(det_table["flux"][di]) for di, ci in matched_pairs])

    # Instrumental magnitude
    valid = det_flux > 0
    inst_mag = np.full_like(det_flux, np.nan)
    inst_mag[valid] = -2.5 * np.log10(det_flux[valid])

    # Compute per-frame zeropoint: zp = cat_vmag - inst_mag
    # This is the magnitude of a source producing 1 count (positive, ~15).
    if np.sum(valid) > 0:
        offsets = cat_vmag[valid] - inst_mag[valid]
        frame_zeropoint = float(np.median(offsets))
    else:
        frame_zeropoint = 0.0

    # Use reference zeropoint if available (absolute scale);
    # otherwise fall back to per-frame (relative scale)
    if reference_zeropoint is not None and reference_zeropoint != 0.0:
        zeropoint = reference_zeropoint
        # Auto-correct old-convention negative zeropoints (pre-v0.3)
        if zeropoint < 0:
            zeropoint = -zeropoint
    else:
        zeropoint = frame_zeropoint

    # Transmission per star: trans = 10^(-0.4 * (inst_mag + zp - cat_vmag))
    # For a clear-sky reference star: inst_mag + zp ≈ cat_vmag → trans ≈ 1.0
    transmission = np.full_like(det_flux, np.nan)
    transmission[valid] = 10 ** (-0.4 * (inst_mag[valid] + zeropoint - cat_vmag[valid]))

    # Probe unmatched catalog stars for zero-transmission regions
    if image is not None or image_shape is not None:
        if image is not None:
            ny_img, nx_img = image.shape
        else:
            ny_img, nx_img = image_shape

        all_az_deg = np.asarray(cat_table["az_deg"], dtype=np.float64)
        all_alt_deg = np.asarray(cat_table["alt_deg"], dtype=np.float64)
        all_vmag = np.asarray(cat_table["vmag_extinct"], dtype=np.float64)

        all_az_rad = np.radians(all_az_deg)
        all_alt_rad = np.radians(all_alt_deg)
        px, py = camera_model.sky_to_pixel(all_az_rad, all_alt_rad)

        margin = probe_radius + 2
        in_frame = (np.isfinite(px) & np.isfinite(py)
                    & (px >= margin) & (px < nx_img - margin)
                    & (py >= margin) & (py < ny_img - margin)
                    & (all_alt_deg > 3.0))
        bright = all_vmag < probe_vmag_limit

        matched_cat_idx = set(ci for _, ci in matched_pairs)
        unmatched_mask = np.ones(len(cat_table), dtype=bool)
        for ci in matched_cat_idx:
            unmatched_mask[ci] = False

        candidates = np.where(in_frame & bright & unmatched_mask)[0]

        if len(candidates) > 0:
            zero_az = []
            zero_alt = []

            if image is not None:
                rr = probe_radius
                for ci in candidates:
                    xi = int(round(px[ci]))
                    yi = int(round(py[ci]))
                    box = image[yi - rr:yi + rr + 1, xi - rr:xi + rr + 1]
                    peak = float(np.max(box))
                    edge = np.concatenate([
                        box[0, :], box[-1, :], box[1:-1, 0], box[1:-1, -1]
                    ])
                    local_bg = float(np.median(edge))
                    if (peak - local_bg) < 500:
                        zero_az.append(all_az_deg[ci])
                        zero_alt.append(all_alt_deg[ci])
            else:
                zero_az = list(all_az_deg[candidates])
                zero_alt = list(all_alt_deg[candidates])

            n_zero = len(zero_az)
            if n_zero > 0:
                cat_az = np.concatenate([cat_az, np.array(zero_az)])
                cat_alt = np.concatenate([cat_alt, np.array(zero_alt)])
                transmission = np.concatenate([
                    transmission, np.zeros(n_zero)
                ])

    return cat_az, cat_alt, transmission, zeropoint


def interpolate_transmission(az_deg, alt_deg, transmission,
                             n_az=180, n_alt=45, smoothing=5.0):
    """Interpolate per-star transmission onto a regular (az, alt) grid.

    Parameters
    ----------
    az_deg, alt_deg : ndarray
        Star positions in degrees.
    transmission : ndarray
        Measured transmission at each star.
    n_az, n_alt : int
        Grid resolution.
    smoothing : float
        RBF smoothing parameter.

    Returns
    -------
    TransmissionMap
    """
    valid = np.isfinite(transmission) & np.isfinite(az_deg) & np.isfinite(alt_deg)
    if np.sum(valid) < 3:
        az_grid = np.linspace(0, 360, n_az, endpoint=False)
        alt_grid = np.linspace(5, 90, n_alt)
        return TransmissionMap(
            az_grid=az_grid, alt_grid=alt_grid,
            transmission=np.full((n_alt, n_az), np.nan),
            zeropoint=0.0,
        )

    az_v = az_deg[valid]
    alt_v = alt_deg[valid]
    trans_v = transmission[valid]

    # Convert to Cartesian on unit sphere for RBF (avoids az wrapping issues)
    az_rad = np.radians(az_v)
    alt_rad = np.radians(alt_v)
    x = np.cos(alt_rad) * np.sin(az_rad)
    y = np.cos(alt_rad) * np.cos(az_rad)
    z = np.sin(alt_rad)
    points = np.column_stack([x, y, z])

    rbf = RBFInterpolator(points, trans_v, smoothing=smoothing,
                          kernel="thin_plate_spline")

    # Build grid
    az_grid = np.linspace(0, 360, n_az, endpoint=False)
    alt_grid = np.linspace(5, 90, n_alt)
    AZ, ALT = np.meshgrid(az_grid, alt_grid)
    az_r = np.radians(AZ.ravel())
    alt_r = np.radians(ALT.ravel())
    xg = np.cos(alt_r) * np.sin(az_r)
    yg = np.cos(alt_r) * np.cos(az_r)
    zg = np.sin(alt_r)
    grid_pts = np.column_stack([xg, yg, zg])

    trans_grid = rbf(grid_pts).reshape(ALT.shape)
    trans_grid = np.clip(trans_grid, 0, 2.0)

    zeropoint = float(np.median(trans_v))

    return TransmissionMap(
        az_grid=az_grid, alt_grid=alt_grid,
        transmission=trans_grid, zeropoint=zeropoint,
    )
