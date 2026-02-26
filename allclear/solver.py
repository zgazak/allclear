"""Fast solve mode: process frames using a known instrument model.

For blind camera characterization, see ``strategies.instrument_fit_pipeline``.
This module handles the fast path: given a saved instrument model, project
the catalog, match sources, optionally refine small pointing drifts, and
compute transmission.
"""

from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table
from scipy.optimize import least_squares

from .projection import CameraModel, ProjectionType
from .matching import match_sources


@dataclass
class SolveResult:
    """Result of a fast solve against a known instrument model."""
    camera_model: CameraModel
    matched_pairs: list          # list of (det_idx, cat_idx)
    rms_residual: float          # pixels
    n_matched: int
    n_expected: int = 0          # catalog stars in frame
    match_fraction: float = 0.0
    guided_det_table: Table = field(default=None, repr=False)
    status: str = "ok"           # ok | low_matches | camera_shifted | cloudy
    status_detail: str = ""


def fast_solve(image, det_table, cat_table, camera_model,
               match_radius=10.0, refine=True, guided=True):
    """Solve a frame using a known camera model.

    Uses guided matching (finding bright peaks at projected catalog
    positions) rather than relying on DAOStarFinder detections, which
    are often contaminated by dome/obstruction artifacts.

    Parameters
    ----------
    image : ndarray (ny, nx)
        Raw image data.
    det_table : Table
        Detected sources (columns: x, y, flux). Used as fallback only.
    cat_table : Table
        Visible catalog stars (columns: az_deg, alt_deg, vmag_extinct).
    camera_model : CameraModel
        Known instrument model.
    match_radius : float
        Search radius in pixels for guided matching.
    refine : bool
        If True, refine small pointing offsets (Δaz, Δalt, Δρ, Δf).
    guided : bool
        If True, use guided matching against pixel data (more robust).

    Returns
    -------
    SolveResult
    """
    ny, nx = image.shape
    cat_az = np.radians(np.asarray(cat_table["az_deg"], dtype=np.float64))
    cat_alt = np.radians(np.asarray(cat_table["alt_deg"], dtype=np.float64))
    vmag = np.asarray(cat_table["vmag_extinct"], dtype=np.float64)

    model = CameraModel(
        cx=camera_model.cx, cy=camera_model.cy,
        az0=camera_model.az0, alt0=camera_model.alt0,
        rho=camera_model.rho, f=camera_model.f,
        proj_type=camera_model.proj_type,
        k1=camera_model.k1, k2=camera_model.k2,
    )

    # Count expected in-frame stars (brighter than mag 5)
    cat_x, cat_y = model.sky_to_pixel(cat_az, cat_alt)
    in_frame = (np.isfinite(cat_x) & np.isfinite(cat_y)
                & (cat_x >= 0) & (cat_x < nx)
                & (cat_y >= 0) & (cat_y < ny)
                & (vmag < 5.0))
    n_expected = int(np.sum(in_frame))

    # --- Guided matching against pixel data ---
    background = float(np.median(image))

    # Use stars up to mag 5.5 for guided matching
    bright_mask = vmag < 5.0
    if np.sum(bright_mask) < 20:
        bright_mask = vmag < 6.0
    bright_idx = np.where(bright_mask)[0]
    bright_az = cat_az[bright_mask]
    bright_alt = cat_alt[bright_mask]

    matches = _guided_match(image, model, bright_az, bright_alt,
                            search_radius=int(match_radius),
                            min_peak=background + 1500,
                            background=background)

    # Refine pointing with initial matches
    if refine and len(matches) >= 6:
        det_x_m = np.array([m[1] for m in matches])
        det_y_m = np.array([m[2] for m in matches])
        cat_az_m = np.array([bright_az[m[0]] for m in matches])
        cat_alt_m = np.array([bright_alt[m[0]] for m in matches])

        model = _refine_pointing(model, det_x_m, det_y_m,
                                 cat_az_m, cat_alt_m)

        # Re-match with refined model at tighter radius
        matches = _guided_match(image, model, bright_az, bright_alt,
                                search_radius=max(5, int(match_radius * 0.7)),
                                min_peak=background + 1500,
                                background=background)

    # Build matched pairs (det_idx, cat_idx) using catalog indices
    # and a synthetic detection table from guided-match centroids
    guided_det_x = []
    guided_det_y = []
    guided_flux = []
    matched_pairs = []

    for i, (local_idx, dx, dy, peak_val) in enumerate(matches):
        cat_idx = int(bright_idx[local_idx])  # map back to full catalog index
        guided_det_x.append(dx)
        guided_det_y.append(dy)
        guided_flux.append(peak_val - background)
        matched_pairs.append((i, cat_idx))

    # Build synthetic detection table from guided centroids
    guided_det = Table()
    if len(guided_det_x) > 0:
        guided_det["x"] = np.array(guided_det_x, dtype=np.float64)
        guided_det["y"] = np.array(guided_det_y, dtype=np.float64)
        guided_det["flux"] = np.array(guided_flux, dtype=np.float64)
    else:
        guided_det["x"] = np.array([], dtype=np.float64)
        guided_det["y"] = np.array([], dtype=np.float64)
        guided_det["flux"] = np.array([], dtype=np.float64)

    n_matched = len(matched_pairs)
    match_frac = n_matched / max(n_expected, 1)

    # Compute RMS from guided matches
    if n_matched > 0:
        gx = np.array(guided_det_x)
        gy = np.array(guided_det_y)
        cat_idx_arr = np.array([ci for _, ci in matched_pairs])
        px, py = model.sky_to_pixel(cat_az[cat_idx_arr], cat_alt[cat_idx_arr])
        rms = float(np.sqrt(np.mean((px - gx) ** 2 + (py - gy) ** 2)))
    else:
        rms = 999.0

    # Diagnose status
    status = "ok"
    detail = ""
    if n_matched < 5:
        status = "low_matches"
        detail = (f"Only {n_matched} matches (expected ~{n_expected}). "
                  "Check if camera has shifted or lens is obstructed.")
    elif match_frac < 0.15:
        # Check if matched stars show uniform dimming (clouds)
        # vs. scatter (camera shifted)
        if n_matched >= 3 and len(guided_flux) > 0:
            fluxes = np.array(guided_flux)
            cat_mags = vmag[np.array([ci for _, ci in matched_pairs])]
            if np.all(fluxes > 0):
                inst_mags = -2.5 * np.log10(fluxes)
                offsets = inst_mags - cat_mags
                spread = float(np.std(offsets))
                if spread < 1.5:
                    status = "cloudy"
                    detail = (f"Match fraction {match_frac:.0%} — "
                              "likely heavy cloud cover.")
                else:
                    status = "camera_shifted"
                    detail = (f"Match fraction {match_frac:.0%} with high "
                              "scatter. Camera may have shifted — "
                              "consider re-running instrument-fit.")

    return SolveResult(
        camera_model=model,
        matched_pairs=matched_pairs,
        rms_residual=rms,
        n_matched=n_matched,
        n_expected=n_expected,
        match_fraction=match_frac,
        guided_det_table=guided_det,
        status=status,
        status_detail=detail,
    )


def _refine_pointing(model, det_x, det_y, cat_az, cat_alt):
    """Refine small pointing offsets (Δaz, Δalt, Δρ, Δf).

    Only adjusts pointing and roll within tight bounds — does NOT
    re-solve the full camera geometry.
    """
    proj_type = model.proj_type

    p0 = np.array([
        model.cx, model.cy, model.az0, model.alt0,
        model.rho, model.f,
    ])
    bounds_lo = [
        p0[0] - 20, p0[1] - 20,
        p0[2] - np.radians(3), p0[3] - np.radians(3),
        p0[4] - np.radians(2), p0[5] * 0.98,
    ]
    bounds_hi = [
        p0[0] + 20, p0[1] + 20,
        p0[2] + np.radians(3), p0[3] + np.radians(3),
        p0[4] + np.radians(2), p0[5] * 1.02,
    ]

    def residuals(params):
        m = CameraModel(
            cx=params[0], cy=params[1], az0=params[2],
            alt0=params[3], rho=params[4], f=params[5],
            proj_type=proj_type,
            k1=model.k1, k2=model.k2,
        )
        px, py = m.sky_to_pixel(cat_az, cat_alt)
        return np.concatenate([px - det_x, py - det_y])

    result = least_squares(
        residuals, p0,
        bounds=(bounds_lo, bounds_hi),
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=500,
    )

    params = result.x
    return CameraModel(
        cx=params[0], cy=params[1], az0=params[2],
        alt0=params[3], rho=params[4], f=params[5],
        proj_type=proj_type,
        k1=model.k1, k2=model.k2,
    )


def _guided_match(image, model, cat_az, cat_alt, search_radius, min_peak,
                  background):
    """Find the bright peak nearest each projected catalog star.

    Returns list of (cat_idx, det_x, det_y, peak_val).
    """
    ny, nx = image.shape
    matches = []
    px, py = model.sky_to_pixel(cat_az, cat_alt)
    r = search_radius

    for i in range(len(cat_az)):
        xi, yi = int(round(px[i])), int(round(py[i]))
        if xi - r < 0 or xi + r >= nx or yi - r < 0 or yi + r >= ny:
            continue

        box = image[yi - r:yi + r + 1, xi - r:xi + r + 1]
        max_val = float(np.max(box))
        if max_val < min_peak:
            continue

        # Centroid around the peak pixel
        box_float = box.astype(np.float64) - background
        box_float[box_float < 0] = 0

        max_pos = np.unravel_index(np.argmax(box), box.shape)
        my, mx = max_pos
        sr = 3
        sy0 = max(0, my - sr)
        sy1 = min(box.shape[0], my + sr + 1)
        sx0 = max(0, mx - sr)
        sx1 = min(box.shape[1], mx + sr + 1)
        sub = box_float[sy0:sy1, sx0:sx1]

        yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
        total = float(np.sum(sub))
        if total > 0:
            cy_c = float(np.sum(yy * sub) / total)
            cx_c = float(np.sum(xx * sub) / total)
        else:
            cy_c, cx_c = float(my), float(mx)

        det_x = xi - r + cx_c
        det_y = yi - r + cy_c
        matches.append((i, det_x, det_y, max_val))

    return matches
