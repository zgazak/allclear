"""Fast solve mode: process frames using a known instrument model.

For blind camera characterization, see ``strategies.instrument_fit_pipeline``.
This module handles the fast path: given a saved instrument model, project
the catalog, match sources, optionally refine small pointing drifts, and
compute transmission.

Supports both fixed-mount cameras (tight refinement) and rotating mounts
(coarse rotation scan followed by refinement).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table
from scipy.optimize import least_squares

from .projection import CameraModel, ProjectionType
from .matching import match_sources

log = logging.getLogger(__name__)

# Minimum matches to consider a solve successful enough to refine
_MIN_MATCHES_REFINE = 6
# Minimum matches from the coarse scan to accept a rotation candidate
_MIN_MATCHES_COARSE = 8


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
               match_radius=10.0, refine=True, guided=True,
               refit_rotation=False):
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
    refit_rotation : bool
        If True, allow wide rotation search when tight solve fails.
        Use for cameras on rotating mounts or platforms that may have
        been physically moved.  When False (default), only small
        pointing adjustments are attempted.

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

    # Count expected in-frame stars
    cat_x, cat_y = model.sky_to_pixel(cat_az, cat_alt)
    in_frame = (np.isfinite(cat_x) & np.isfinite(cat_y)
                & (cat_x >= 0) & (cat_x < nx)
                & (cat_y >= 0) & (cat_y < ny)
                & (vmag < 5.5))
    n_expected = int(np.sum(in_frame))

    # --- Guided matching against pixel data ---
    background = float(np.median(image))

    # Use stars up to mag 5.5 for guided matching
    bright_mask = vmag < 5.5
    if np.sum(bright_mask) < 20:
        bright_mask = vmag < 6.5
    bright_idx = np.where(bright_mask)[0]
    bright_az = cat_az[bright_mask]
    bright_alt = cat_alt[bright_mask]

    # For solve mode with a known model, we can use a much lower
    # detection threshold than instrument-fit (which uses 15-sigma).
    # With tight search radius and known geometry, false matches from
    # noise are rare.  3-sigma detects most visible stars.
    noise_est = max(30.0, np.sqrt(abs(background)))
    min_peak = background + 3.0 * noise_est

    matches = _guided_match(image, model, bright_az, bright_alt,
                            search_radius=int(match_radius),
                            min_peak=min_peak,
                            background=background)
    original_model = model
    original_n = len(matches)

    # --- Rotation recovery ---
    if refit_rotation:
        delta_rho = _find_rotation_offset(image, model, bright_az, bright_alt,
                                          background=background,
                                          min_peak=min_peak)
        if delta_rho is not None:
            model = CameraModel(
                cx=model.cx, cy=model.cy,
                az0=model.az0, alt0=model.alt0,
                rho=model.rho + delta_rho, f=model.f,
                proj_type=model.proj_type,
                k1=model.k1, k2=model.k2,
            )
            for iteration, sr in enumerate([
                int(match_radius),
                max(5, int(match_radius * 0.7)),
            ]):
                rot_matches = _guided_match(
                    image, model, bright_az, bright_alt,
                    search_radius=sr,
                    min_peak=min_peak, background=background,
                )
                if len(rot_matches) < _MIN_MATCHES_REFINE:
                    break
                det_x_r = np.array([m[1] for m in rot_matches])
                det_y_r = np.array([m[2] for m in rot_matches])
                cat_az_r = np.array([bright_az[m[0]] for m in rot_matches])
                cat_alt_r = np.array([bright_alt[m[0]] for m in rot_matches])
                # Sigma-clip before refining
                px_pre, py_pre = model.sky_to_pixel(cat_az_r, cat_alt_r)
                resid = np.sqrt((px_pre - det_x_r)**2 +
                                (py_pre - det_y_r)**2)
                good = resid < max(10.0, np.median(resid) + 2.5 * np.std(resid))
                if np.sum(good) >= _MIN_MATCHES_REFINE:
                    model = _refine_pointing(
                        model, det_x_r[good], det_y_r[good],
                        cat_az_r[good], cat_alt_r[good],
                        wide=(iteration == 0),
                    )

            # Check: did rotation recovery actually improve things?
            rot_final = _guided_match(
                image, model, bright_az, bright_alt,
                search_radius=int(match_radius),
                min_peak=min_peak, background=background,
            )
            if len(rot_final) > original_n:
                matches = rot_final
                log.info("Rotation recovery (Δρ=%.1f°): %d→%d matches",
                         np.degrees(delta_rho), original_n, len(rot_final))
                refine = False  # already refined
            else:
                # Rotation recovery made things worse — revert
                model = original_model
                log.info("Rotation recovery reverted (no improvement)")

    # Refine pointing with initial matches — ONLY if it improves
    if refine and len(matches) >= _MIN_MATCHES_REFINE:
        det_x_m = np.array([m[1] for m in matches])
        det_y_m = np.array([m[2] for m in matches])
        cat_az_m = np.array([bright_az[m[0]] for m in matches])
        cat_alt_m = np.array([bright_alt[m[0]] for m in matches])

        # Sigma-clip before refining
        px_pre, py_pre = model.sky_to_pixel(cat_az_m, cat_alt_m)
        resid = np.sqrt((px_pre - det_x_m)**2 + (py_pre - det_y_m)**2)
        good = resid < max(8.0, np.median(resid) + 2.5 * np.std(resid))
        if np.sum(good) >= _MIN_MATCHES_REFINE:
            refined = _refine_pointing(
                model, det_x_m[good], det_y_m[good],
                cat_az_m[good], cat_alt_m[good])

            # Check: did refine improve things?
            refined_matches = _guided_match(
                image, refined, bright_az, bright_alt,
                search_radius=int(match_radius),
                min_peak=min_peak, background=background)

            if len(refined_matches) >= len(matches):
                model = refined
                matches = refined_matches

    # Final neighborhood-verified match: confirm each match by checking
    # that its catalog neighbors also have detectable peaks.  This rejects
    # wrong-star matches in dense fields (Milky Way).
    verified = _neighborhood_verified_match(
        image, model, bright_az, bright_alt, vmag[bright_mask],
        search_radius=max(3, min(5, int(match_radius))),
        min_peak=min_peak, background=background,
        n_neighbors=3, confirm_radius=5, min_confirmed=2,
    )
    if len(verified) >= 10:
        matches = verified

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


def _find_rotation_offset(image, model, cat_az, cat_alt, background,
                           min_peak, r_tol_frac=0.03, n_bins=360):
    """Find the rotation offset by matching stars by radial distance.

    When a camera rotates around its optical axis, every star moves in
    an arc at constant radius from (cx, cy).  This function:

    1. Projects catalog stars to get predicted (r, PA) in polar coords
       around the optical center.
    2. Finds bright peaks in the image (detections), filtered to a
       "star zone" that excludes dome edges and zenith artifacts.
    3. Matches catalog→detection by radial distance (same star = same r
       regardless of rotation).
    4. Histograms the PA difference (ΔPA = PA_det - PA_pred) — the peak
       is the rotation offset Δρ.

    This is a 1-parameter solve: no optimizer, no degeneracy, works at
    any tilt angle.

    Returns
    -------
    float or None
        Rotation offset in radians, or None if no clear peak found.
    """
    from .detection import detect_stars
    from scipy.ndimage import uniform_filter1d

    ny, nx = image.shape
    cx, cy = model.cx, model.cy

    # Project catalog stars to pixel coords
    pred_x, pred_y = model.sky_to_pixel(cat_az, cat_alt)
    valid = (np.isfinite(pred_x) & np.isfinite(pred_y)
             & (pred_x >= 20) & (pred_x < nx - 20)
             & (pred_y >= 20) & (pred_y < ny - 20))
    pred_x, pred_y = pred_x[valid], pred_y[valid]

    # Polar coords of predicted positions relative to optical center
    pred_r = np.sqrt((pred_x - cx) ** 2 + (pred_y - cy) ** 2)
    pred_pa = np.arctan2(pred_x - cx, pred_y - cy)

    # Detect bright sources in the image
    det = detect_stars(image, fwhm=5.0, threshold_sigma=5.0, n_brightest=500)
    if len(det) < 10:
        return None
    det_x = np.asarray(det["x"], dtype=np.float64)
    det_y = np.asarray(det["y"], dtype=np.float64)

    # Polar coords of detections
    det_r = np.sqrt((det_x - cx) ** 2 + (det_y - cy) ** 2)
    det_pa = np.arctan2(det_x - cx, det_y - cy)

    # Filter to "star zone": exclude dome edges and near-center artifacts.
    # Use the range of catalog star radii to define the zone.
    r_min = max(float(np.percentile(pred_r, 10)), 50.0)
    r_max = float(np.percentile(pred_r, 95))
    star_zone = (det_r > r_min) & (det_r < r_max)
    det_r_z = det_r[star_zone]
    det_pa_z = det_pa[star_zone]

    if len(det_r_z) < 10:
        log.info("Rotation recovery: too few star-zone detections (%d)",
                 len(det_r_z))
        return None

    # Match by radial distance: for each catalog star, find detections
    # at similar radius and collect the PA differences
    delta_pa_all = []
    for i in range(len(pred_r)):
        if pred_r[i] < r_min or pred_r[i] > r_max:
            continue
        r_tol = max(8.0, pred_r[i] * r_tol_frac)
        nearby = np.abs(det_r_z - pred_r[i]) < r_tol
        if not np.any(nearby):
            continue
        dpa = det_pa_z[nearby] - pred_pa[i]
        dpa = (dpa + np.pi) % (2 * np.pi) - np.pi
        delta_pa_all.extend(dpa.tolist())

    if len(delta_pa_all) < 50:
        log.info("Rotation recovery: too few radial matches (%d)",
                 len(delta_pa_all))
        return None

    delta_pa_all = np.array(delta_pa_all)

    # Histogram to find the dominant rotation offset
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, edges = np.histogram(delta_pa_all, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth the histogram to find robust peak
    smooth = uniform_filter1d(hist.astype(float), size=5, mode="wrap")
    peak_idx = int(np.argmax(smooth))
    delta_rho = float(bin_centers[peak_idx])

    # Check that the peak is significantly above the background
    median_count = float(np.median(smooth))
    peak_count = float(smooth[peak_idx])
    if peak_count < median_count * 2.0:
        log.info("Rotation recovery: no clear peak (%.0f vs median %.0f)",
                 peak_count, median_count)
        return None

    log.info("Rotation recovery: Δρ = %.1f° (peak=%.0f, median=%.0f, "
             "ratio=%.1fx)", np.degrees(delta_rho), peak_count,
             median_count, peak_count / max(median_count, 1))
    return delta_rho


def _refine_pointing(model, det_x, det_y, cat_az, cat_alt, wide=False):
    """Refine pointing offsets.

    Fits (cx, cy, alt0, combined_rotation, f) with az0 fixed to break
    the near-zenith az0/rho degeneracy.  The combined rotation
    (az0 + rho) is what matters physically; splitting between az0 and
    rho is arbitrary for near-zenith cameras.

    Parameters
    ----------
    wide : bool
        If True, use wider bounds (camera may have been moved/adjusted).
    """
    proj_type = model.proj_type
    fixed_az0 = model.az0

    # Parameterize as (cx, cy, alt0, rho, f) with az0 fixed
    p0 = np.array([
        model.cx, model.cy, model.alt0, model.rho, model.f,
    ])

    if wide:
        bounds_lo = [
            p0[0] - 50, p0[1] - 50,
            p0[2] - np.radians(10),
            p0[3] - np.radians(10), p0[4] * 0.95,
        ]
        bounds_hi = [
            p0[0] + 50, p0[1] + 50,
            p0[2] + np.radians(10),
            p0[3] + np.radians(10), p0[4] * 1.05,
        ]
    else:
        # Allow enough room for seasonal thermal drift:
        # cx/cy: ±15px, alt0: ±5° (tilt), rho: ±5° (rotation), f: ±1%
        bounds_lo = [
            p0[0] - 15, p0[1] - 15,
            p0[2] - np.radians(5),
            p0[3] - np.radians(5), p0[4] * 0.99,
        ]
        bounds_hi = [
            p0[0] + 15, p0[1] + 15,
            p0[2] + np.radians(5),
            p0[3] + np.radians(5), p0[4] * 1.01,
        ]

    def residuals(params):
        m = CameraModel(
            cx=params[0], cy=params[1], az0=fixed_az0,
            alt0=params[2], rho=params[3], f=params[4],
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
        cx=params[0], cy=params[1], az0=fixed_az0,
        alt0=params[2], rho=params[3], f=params[4],
        proj_type=proj_type,
        k1=model.k1, k2=model.k2,
    )


# Use the same _guided_match from strategies.py (with nearest-peak matching)
from .strategies import _guided_match


def _neighborhood_verified_match(image, model, cat_az, cat_alt, cat_vmag,
                                  search_radius, min_peak, background,
                                  n_neighbors=3, confirm_radius=5,
                                  min_confirmed=2, max_offset_spread=4.0):
    """Guided matching with neighborhood offset-consistency verification.

    For each candidate match, also guided-match its nearest catalog
    neighbors.  If the offsets (predicted→detected) are consistent
    across the group (i.e., the pattern shifted as a unit), the match
    is confirmed.  If the offsets are random, it's a wrong-star match
    in a dense field.

    Parameters
    ----------
    max_offset_spread : float
        Maximum spread (std) of neighbor offsets to accept as consistent.
    """
    ny, nx = image.shape

    # Project all catalog stars and do guided match
    px_all, py_all = model.sky_to_pixel(cat_az, cat_alt)

    matches = _guided_match(image, model, cat_az, cat_alt,
                            search_radius, min_peak, background)
    if len(matches) < 10:
        return matches

    # Build lookup: cat_idx → match result for fast neighbor checking
    match_by_cat = {}
    for cat_idx, det_x, det_y, peak_val in matches:
        match_by_cat[cat_idx] = (det_x, det_y, peak_val)

    # Build KDTree of predicted positions for neighbor lookup
    from scipy.spatial import KDTree
    valid = (np.isfinite(px_all) & np.isfinite(py_all) &
             (px_all >= 10) & (px_all < nx - 10) &
             (py_all >= 10) & (py_all < ny - 10))
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < n_neighbors + 1:
        return matches
    cat_tree = KDTree(np.column_stack([px_all[valid_idx],
                                        py_all[valid_idx]]))

    verified = []
    for cat_idx, det_x, det_y, peak_val in matches:
        pred_x = float(px_all[cat_idx])
        pred_y = float(py_all[cat_idx])
        my_dx = det_x - pred_x
        my_dy = det_y - pred_y

        if not valid[cat_idx]:
            continue

        # Find nearest catalog neighbors
        dists_nn, idx_nn = cat_tree.query(
            [pred_x, pred_y],
            k=min(n_neighbors + 1, len(valid_idx)))
        if np.isscalar(idx_nn):
            idx_nn = np.array([idx_nn])
            dists_nn = np.array([dists_nn])

        # Check if neighbors have consistent offsets
        neighbor_offsets_dx = []
        neighbor_offsets_dy = []
        for nn_tree_idx, nn_dist in zip(idx_nn, dists_nn):
            if nn_dist < 1.0:   # skip self
                continue
            if nn_dist > 200:   # too far
                continue
            nn_cat_idx = int(valid_idx[nn_tree_idx])

            if nn_cat_idx in match_by_cat:
                nn_det_x, nn_det_y, _ = match_by_cat[nn_cat_idx]
                nn_pred_x = float(px_all[nn_cat_idx])
                nn_pred_y = float(py_all[nn_cat_idx])
                neighbor_offsets_dx.append(nn_det_x - nn_pred_x)
                neighbor_offsets_dy.append(nn_det_y - nn_pred_y)

        if len(neighbor_offsets_dx) < min_confirmed:
            # Not enough neighbors matched — can't verify.
            # Keep the match if it's a tight one (likely correct).
            if abs(my_dx) < 2.0 and abs(my_dy) < 2.0:
                verified.append((cat_idx, det_x, det_y, peak_val))
            continue

        # Check offset consistency: the center star and its neighbors
        # should all have similar (dx, dy) offsets if correctly matched.
        all_dx = np.array([my_dx] + neighbor_offsets_dx)
        all_dy = np.array([my_dy] + neighbor_offsets_dy)
        spread = np.sqrt(np.std(all_dx)**2 + np.std(all_dy)**2)

        if spread < max_offset_spread:
            verified.append((cat_idx, det_x, det_y, peak_val))

    return verified
