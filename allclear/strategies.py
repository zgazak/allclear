"""Instrument-fit solving strategies for blind camera characterization.

Implements six strategies that are combined into a robust pipeline:

1. Radial density profile  → rough f estimate, projection ranking
2. Azimuthal star-count correlation → roll angle ρ
3. Brightness-ordered anchoring → candidate bright-star assignments
4. Center-outward progressive matching
5. RANSAC refinement
6. Residual vector field diagnostics
"""

import logging

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import KDTree

from .projection import CameraModel, ProjectionType
from .matching import match_sources

log = logging.getLogger(__name__)


def _adaptive_min_peak_offset(background, p999):
    """Compute adaptive min_peak_offset from image statistics.

    Uses the lower of a noise-based threshold (15-sigma) and a
    dynamic-range-based threshold (50% of p99.9-median).  Floored at 200
    and capped at 3000.  This scales correctly for both high-background
    images (dome glow, Haleakala) and low-background deep CCD images
    (APICAM at Paranal).
    """
    noise_est = max(30.0, np.sqrt(abs(background)))
    snr_based = 15.0 * noise_est
    range_based = 0.5 * (p999 - background)
    return max(200.0, min(3000.0, min(snr_based, range_based)))


# ---------------------------------------------------------------------------
# Strategy 1: Radial density profile → rough f, projection ranking
# ---------------------------------------------------------------------------

def radial_density_profile(det_x, det_y, cx, cy, n_bins=30):
    """Compute the observed radial density of detections.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detection pixel positions.
    cx, cy : float
        Assumed image center.
    n_bins : int
        Number of radial bins.

    Returns
    -------
    r_centers : ndarray
        Bin center radii in pixels.
    density : ndarray
        Sources per pixel per unit radius (dn/dr normalized by annulus area).
    """
    r = np.sqrt((det_x - cx) ** 2 + (det_y - cy) ** 2)
    r_max = np.percentile(r, 98)
    edges = np.linspace(0, r_max, n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    # Annulus area = pi * (r2^2 - r1^2)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    areas[areas < 1] = 1
    density = counts / areas
    return r_centers, density


def estimate_focal_length_from_density(det_x, det_y, cx, cy,
                                       n_cat_stars=500):
    """Estimate focal length from the radial density fall-off.

    Stars are roughly uniform on the sphere.  For equidistant projection,
    the density per unit pixel area goes as sin(r/f)/f at radius r.  The
    radius where density peaks or where the cumulative reaches 50% gives
    a constraint on f.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detection positions.
    cx, cy : float
        Image center.
    n_cat_stars : int
        Approximate number of catalog stars above the horizon.

    Returns
    -------
    f_est : float
        Estimated focal length in pixels.
    proj_scores : dict
        {projection_name: fit_score} — lower is better.
    """
    r = np.sqrt((det_x - cx) ** 2 + (det_y - cy) ** 2)
    r_max = np.percentile(r, 95)
    r_centers, density = radial_density_profile(det_x, det_y, cx, cy)

    # Normalize density
    density_norm = density / (np.sum(density) + 1e-10)

    best_f = r_max / (np.pi / 2)  # default: r_max maps to 90°
    best_score = 1e10
    proj_scores = {}

    for proj_type in ProjectionType:
        for f_try in np.linspace(max(200, r_max * 0.4), r_max * 1.5, 50):
            # Predicted density for uniform-on-sphere stars
            theta = np.clip(r_centers / f_try, 0, np.pi / 2 - 0.01)
            if proj_type == ProjectionType.EQUIDISTANT:
                # dr/dtheta = f, solid angle element ~ sin(theta) dtheta dphi
                pred = np.sin(theta) / f_try
            elif proj_type == ProjectionType.EQUISOLID:
                pred = np.sin(theta) / (2 * f_try * np.cos(theta / 2))
            elif proj_type == ProjectionType.STEREOGRAPHIC:
                cos_half = np.cos(theta / 2)
                pred = np.sin(theta) / (2 * f_try) * cos_half ** 2
            elif proj_type == ProjectionType.ORTHOGRAPHIC:
                pred = np.sin(theta) / (f_try * np.cos(theta) + 1e-10)
            else:
                continue

            pred = np.abs(pred)
            pred_norm = pred / (np.sum(pred) + 1e-10)
            score = np.sum((density_norm - pred_norm) ** 2)

            if proj_type.value not in proj_scores or score < proj_scores[proj_type.value]:
                proj_scores[proj_type.value] = float(score)

            if score < best_score:
                best_score = score
                best_f = float(f_try)

    return best_f, proj_scores


# ---------------------------------------------------------------------------
# Strategy 2: Azimuthal star-count correlation → roll angle
# ---------------------------------------------------------------------------

def azimuthal_correlation(det_x, det_y, cx, cy,
                          cat_az_deg, cat_alt_deg,
                          n_bins=72, alt_min=15.0):
    """Cross-correlate azimuthal source distributions to find roll angle.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detection pixel positions.
    cx, cy : float
        Image center.
    cat_az_deg, cat_alt_deg : ndarray
        Catalog star positions (degrees).
    n_bins : int
        Number of azimuthal bins (360/n_bins degrees each).
    alt_min : float
        Minimum altitude for catalog stars (degrees).

    Returns
    -------
    best_rho_deg : float
        Best-fit roll angle (degrees) — the offset that aligns the
        detected azimuthal distribution with the catalog one.
    correlation : ndarray
        Cross-correlation as a function of roll offset (length n_bins).
    offsets_deg : ndarray
        Roll offsets corresponding to ``correlation`` (degrees).
    """
    # Observed: position angle from image center (measured from +x axis)
    pa = np.degrees(np.arctan2(det_y - cy, det_x - cx)) % 360
    obs_hist, _ = np.histogram(pa, bins=n_bins, range=(0, 360))
    obs_hist = obs_hist.astype(np.float64)

    # Catalog: azimuth histogram for stars above alt_min
    mask = cat_alt_deg >= alt_min
    cat_az_use = cat_az_deg[mask] % 360
    cat_hist, _ = np.histogram(cat_az_use, bins=n_bins, range=(0, 360))
    cat_hist = cat_hist.astype(np.float64)

    # Normalize
    obs_hist -= np.mean(obs_hist)
    cat_hist -= np.mean(cat_hist)

    obs_norm = np.sqrt(np.sum(obs_hist ** 2)) + 1e-10
    cat_norm = np.sqrt(np.sum(cat_hist ** 2)) + 1e-10

    # Circular cross-correlation
    correlation = np.zeros(n_bins)
    for shift in range(n_bins):
        correlation[shift] = np.sum(
            obs_hist * np.roll(cat_hist, shift)
        ) / (obs_norm * cat_norm)

    best_idx = np.argmax(correlation)
    offsets_deg = np.arange(n_bins) * (360.0 / n_bins)
    best_rho_deg = offsets_deg[best_idx]

    return best_rho_deg, correlation, offsets_deg


def compact_arc_rho_search(image, cat_az, cat_alt, cx, cy, f,
                           n_bins=24, r_min=200, r_max=1100,
                           min_peak_offset=1000, compact_threshold=1.5,
                           n_brightest=500):
    """Find roll angle by correlating compact-detection azimuthal profile
    with projected catalog profile.

    Detects sources, filters to compact point sources (rejecting dome
    emission), bins by position angle, and for each trial rho projects
    the catalog through the camera model and correlates the azimuthal
    profiles.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az, cat_alt : ndarray
        Catalog star positions (radians).
    cx, cy : float
        Optical center.
    f : float
        Focal length estimate (pixels).
    n_bins : int
        Number of azimuthal bins.
    r_min, r_max : float
        Radial ring bounds — excludes zenith pile-up and horizon noise.
    min_peak_offset : float
        Minimum brightness above background for a detection to be
        considered.
    compact_threshold : float
        Inner/outer brightness ratio threshold for point sources.
    n_brightest : int
        Number of sources to detect for compactness filtering.

    Returns
    -------
    best_rho_rad : float
        Best roll angle (radians).
    correlations : ndarray
        Correlation at each 1-degree rho step.
    """
    from .detection import detect_stars
    ny, nx = image.shape
    background = float(np.median(image))
    min_peak = background + min_peak_offset

    det = detect_stars(image, n_brightest=n_brightest)
    det_x = np.asarray(det["x"], dtype=np.float64)
    det_y = np.asarray(det["y"], dtype=np.float64)

    # Filter to compact point sources in the radial ring
    ir = 3
    outr = 7
    ring_x = []
    ring_y = []
    for i in range(len(det_x)):
        xi, yi = int(round(det_x[i])), int(round(det_y[i]))
        r = np.sqrt((det_x[i] - cx) ** 2 + (det_y[i] - cy) ** 2)
        if r < r_min or r > r_max:
            continue
        if xi - outr < 0 or xi + outr >= nx or yi - outr < 0 or yi + outr >= ny:
            continue
        inner = image[yi - ir:yi + ir + 1, xi - ir:xi + ir + 1]
        outer = image[yi - outr:yi + outr + 1, xi - outr:xi + outr + 1]
        inner_mean = float(np.mean(inner))
        outer_mean = float(np.mean(outer))
        if outer_mean <= 0 or inner_mean < min_peak:
            continue
        if inner_mean / outer_mean < compact_threshold:
            continue
        ring_x.append(det_x[i])
        ring_y.append(det_y[i])

    ring_x = np.array(ring_x)
    ring_y = np.array(ring_y)

    if len(ring_x) < 10:
        log.warning(f"  Only {len(ring_x)} compact sources in ring, "
                    "arc profiling may be unreliable")

    # Observed azimuthal profile
    pa_det = np.degrees(np.arctan2(ring_y - cy, ring_x - cx)) % 360
    obs_hist, _ = np.histogram(pa_det, bins=n_bins, range=(0, 360))
    obs_hist = obs_hist.astype(np.float64)
    obs_n = obs_hist - np.mean(obs_hist)
    obs_norm = np.sqrt(np.sum(obs_n ** 2)) + 1e-10

    # Scan rho at 1-degree steps
    correlations = np.zeros(360)
    for rho_deg in range(360):
        model = CameraModel(
            cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
            rho=np.radians(rho_deg), f=f,
            proj_type=ProjectionType.EQUIDISTANT,
        )
        px, py = model.sky_to_pixel(cat_az, cat_alt)
        r_cat = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        in_ring = ((r_cat > r_min) & (r_cat < r_max) &
                   (px >= 0) & (px < nx) & (py >= 0) & (py < ny))
        pa_cat = np.degrees(
            np.arctan2(py[in_ring] - cy, px[in_ring] - cx)
        ) % 360
        cat_hist, _ = np.histogram(pa_cat, bins=n_bins, range=(0, 360))
        cat_hist = cat_hist.astype(np.float64)
        cat_n = cat_hist - np.mean(cat_hist)
        cat_norm = np.sqrt(np.sum(cat_n ** 2)) + 1e-10
        correlations[rho_deg] = np.sum(obs_n * cat_n) / (obs_norm * cat_norm)

    best_rho_deg = int(np.argmax(correlations))
    best_corr = correlations[best_rho_deg]

    log.info(f"  Compact arc profiling: {len(ring_x)} compact sources, "
             f"rho={best_rho_deg}° (corr={best_corr:.3f})")

    return np.radians(best_rho_deg), correlations


# ---------------------------------------------------------------------------
# Strategy 3: Brightness-ordered matching
# ---------------------------------------------------------------------------

def brightness_anchor(image, cat_az_rad, cat_alt_rad, cat_vmag,
                      f_est, rho_est, cx, cy,
                      n_bright_cat=12, n_bright_det=8):
    """Try assigning brightest detections to brightest catalog stars.

    For each candidate assignment of the brightest detected source to
    a bright catalog star, fit a quick model and score by pixel-brightness.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az_rad, cat_alt_rad : ndarray
        Catalog positions (radians).
    cat_vmag : ndarray
        Extinction-corrected magnitudes.
    f_est : float
        Focal length estimate.
    rho_est : float
        Roll angle estimate (radians).
    cx, cy : float
        Image center.
    n_bright_cat : int
        Number of brightest catalog stars to try.
    n_bright_det : int
        Number of brightest detected sources to try.

    Returns
    -------
    best_model : CameraModel
        Best-scoring model from brightness anchoring.
    best_score : float
        Pixel brightness score.
    """
    ny, nx = image.shape
    background = float(np.median(image))
    image_sub = (image.astype(np.float64) - background)

    # Brightest catalog stars
    order = np.argsort(cat_vmag)
    bright_cat_idx = order[:n_bright_cat]

    # Find brightest peaks in the image
    det_peaks = _find_bright_peaks(image, background, n_peaks=n_bright_det)

    best_model = None
    best_score = -1.0

    for det_x, det_y, _ in det_peaks:
        for ci in bright_cat_idx:
            star_az = cat_az_rad[ci]
            star_alt = cat_alt_rad[ci]

            # Given this assignment, what az0 makes this star land at (det_x, det_y)?
            # For a zenith-pointing equidistant camera:
            #   r = f * (pi/2 - alt)
            #   pixel_pa = rho + az  (roughly)
            # So az0 ~ az_star - pixel_angle_of_detection
            r_det = np.sqrt((det_x - cx) ** 2 + (det_y - cy) ** 2)
            pa_det = np.arctan2(det_x - cx, det_y - cy)  # position angle

            # Expected r from this star
            zenith_dist = np.pi / 2 - star_alt
            if zenith_dist < 0.01:
                continue  # star at zenith — can't determine pa

            f_trial = r_det / zenith_dist if zenith_dist > 0.05 else f_est

            # The star should appear at pa = rho + (az_star - az0)
            # So az0 = az_star - (pa_det - rho)
            az0_trial = star_az - (pa_det - rho_est)

            model = CameraModel(
                cx=cx, cy=cy, az0=az0_trial, alt0=np.pi / 2,
                rho=rho_est, f=f_trial,
                proj_type=ProjectionType.EQUIDISTANT,
            )

            # Score: sum of brightness at projected catalog positions
            # Use top 50 brightest catalog stars
            top50 = order[:50]
            px, py = model.sky_to_pixel(cat_az_rad[top50], cat_alt_rad[top50])
            score = _brightness_score(image_sub, px, py, nx, ny, box_half=5)

            if score > best_score:
                best_score = score
                best_model = model

    if best_model is None:
        best_model = CameraModel(
            cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
            rho=rho_est, f=f_est,
            proj_type=ProjectionType.EQUIDISTANT,
        )

    return best_model, best_score


def _find_bright_peaks(image, background, n_peaks=10, min_sep=50):
    """Find the brightest isolated peaks in the image.

    Returns list of (x, y, peak_val) sorted by brightness.
    """
    ny, nx = image.shape
    threshold = background + 5000
    margin = 20

    # Find local maxima above threshold
    peaks = []
    # Use a coarse grid to find candidates
    step = 5
    for y0 in range(margin, ny - margin, step):
        for x0 in range(margin, nx - margin, step):
            val = float(image[y0, x0])
            if val < threshold:
                continue
            # Check if local max in 11x11 box
            box = image[max(0, y0 - 5):y0 + 6, max(0, x0 - 5):x0 + 6]
            if val >= np.max(box):
                peaks.append((x0, y0, val))

    peaks.sort(key=lambda p: -p[2])

    # Deduplicate by min_sep
    kept = []
    for x, y, v in peaks:
        too_close = False
        for kx, ky, _ in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 < min_sep ** 2:
                too_close = True
                break
        if not too_close:
            kept.append((x, y, v))
        if len(kept) >= n_peaks:
            break

    return kept


# ---------------------------------------------------------------------------
# Strategy 4: RANSAC matching
# ---------------------------------------------------------------------------

def ransac_refine(image, cat_az_rad, cat_alt_rad, model,
                  n_iterations=500, inlier_threshold=15.0,
                  min_inliers=8, fix_az0=False):
    """RANSAC refinement of camera model against image pixel data.

    Uses guided matching to generate candidate pairs, then RANSAC
    to find a robust consensus set.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az_rad, cat_alt_rad : ndarray
        Catalog positions (radians).
    model : CameraModel
        Initial model.
    n_iterations : int
        RANSAC iterations.
    inlier_threshold : float
        Pixel distance for inlier classification.
    min_inliers : int
        Minimum inliers for a valid model.
    fix_az0 : bool
        If True, fix az0 during fitting (breaks near-zenith degeneracy).

    Returns
    -------
    best_model : CameraModel
        Model with most inliers.
    n_inliers : int
        Number of inliers.
    rms : float
        RMS of inlier residuals.
    """
    ny, nx = image.shape
    background = float(np.median(image))
    p999 = float(np.percentile(image, 99.9))
    adaptive_mpo = _adaptive_min_peak_offset(background, p999)

    # Generate candidate matches via guided matching at generous radius
    matches = _guided_match(image, model, cat_az_rad, cat_alt_rad,
                            search_radius=20, min_peak=background + adaptive_mpo,
                            background=background,
                            alt_min_rad=np.radians(15),
                            alt_max_rad=np.radians(75))

    if len(matches) < min_inliers:
        return model, len(matches), 999.0

    cat_idx = np.array([m[0] for m in matches])
    det_x = np.array([m[1] for m in matches])
    det_y = np.array([m[2] for m in matches])
    match_az = cat_az_rad[cat_idx]
    match_alt = cat_alt_rad[cat_idx]

    best_model = model
    best_n_inliers = 0
    best_inlier_mask = np.zeros(len(matches), dtype=bool)

    rng = np.random.RandomState(42)

    for _ in range(n_iterations):
        # Sample 4 random pairs
        sample = rng.choice(len(matches), size=min(4, len(matches)),
                            replace=False)

        # Fit model to sample
        try:
            sample_model = _fit_model_to_pairs(
                det_x[sample], det_y[sample],
                match_az[sample], match_alt[sample],
                model, fix_az0=fix_az0,
            )
        except Exception:
            continue

        # Score: project all catalog matches and count inliers
        px, py = sample_model.sky_to_pixel(match_az, match_alt)
        residuals = np.sqrt((px - det_x) ** 2 + (py - det_y) ** 2)
        inlier_mask = residuals < inlier_threshold
        n_inliers = int(np.sum(inlier_mask))

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inlier_mask
            best_model = sample_model

    # Re-fit on all inliers
    if best_n_inliers >= min_inliers:
        inl = best_inlier_mask
        try:
            best_model = _fit_model_to_pairs(
                det_x[inl], det_y[inl],
                match_az[inl], match_alt[inl],
                best_model, fix_az0=fix_az0,
            )
        except Exception:
            pass

        px, py = best_model.sky_to_pixel(match_az[inl], match_alt[inl])
        rms = float(np.sqrt(np.mean(
            (px - det_x[inl]) ** 2 + (py - det_y[inl]) ** 2
        )))
    else:
        rms = 999.0

    return best_model, best_n_inliers, rms


# ---------------------------------------------------------------------------
# Strategy 5: Residual vector field diagnostics
# ---------------------------------------------------------------------------

def diagnose_residuals(det_x, det_y, proj_x, proj_y, cx, cy):
    """Analyze residual vectors for systematic patterns.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detected positions.
    proj_x, proj_y : ndarray
        Projected catalog positions.
    cx, cy : float
        Optical center.

    Returns
    -------
    diagnostics : dict
        Keys: 'mean_dx', 'mean_dy' (translation),
        'radial_trend' (slope of radial residual vs r),
        'tangential_mean' (mean tangential residual — roll error),
        'pattern' (string description of dominant pattern).
    """
    dx = proj_x - det_x
    dy = proj_y - det_y

    # Radial distance from center
    r = np.sqrt((det_x - cx) ** 2 + (det_y - cy) ** 2)
    r[r < 1] = 1

    # Radial and tangential decomposition
    ux = (det_x - cx) / r  # unit radial vector
    uy = (det_y - cy) / r
    radial_resid = dx * ux + dy * uy
    tangential_resid = -dx * uy + dy * ux

    mean_dx = float(np.mean(dx))
    mean_dy = float(np.mean(dy))
    mean_radial = float(np.mean(radial_resid))
    mean_tangential = float(np.mean(tangential_resid))

    # Radial trend: linear fit of radial residual vs r
    if len(r) > 3:
        coeffs = np.polyfit(r, radial_resid, 1)
        radial_slope = float(coeffs[0])
    else:
        radial_slope = 0.0

    # Diagnose dominant pattern
    translation = np.sqrt(mean_dx ** 2 + mean_dy ** 2)
    pattern = "clean"
    if translation > 3.0:
        pattern = "translation (cx/cy offset)"
    elif abs(mean_radial) > 2.0:
        pattern = "radial (focal length error)"
    elif abs(mean_tangential) > 2.0:
        pattern = "tangential (roll error)"
    elif abs(radial_slope) > 0.005:
        pattern = "radial_gradient (distortion k1 error)"

    return {
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "mean_radial": mean_radial,
        "mean_tangential": mean_tangential,
        "radial_slope": radial_slope,
        "pattern": pattern,
    }


def apply_residual_corrections(model, diagnostics):
    """Apply corrections based on residual diagnostics.

    Returns a corrected CameraModel.
    """
    cx = model.cx - diagnostics["mean_dx"]
    cy = model.cy - diagnostics["mean_dy"]

    # Roll correction from tangential residual
    # Tangential residual of T pixels at mean radius R → delta_rho ~ T/R
    mean_r = max(model.f * 0.5, 100)  # rough mean radius
    drho = -diagnostics["mean_tangential"] / mean_r

    # Focal length correction from mean radial residual
    # Radial residual of R_err at mean radius R → delta_f/f ~ R_err/R
    df = -diagnostics["mean_radial"] / mean_r * model.f

    return CameraModel(
        cx=cx, cy=cy,
        az0=model.az0, alt0=model.alt0,
        rho=model.rho + drho,
        f=model.f + df,
        proj_type=model.proj_type,
        k1=model.k1, k2=model.k2,
    )


# ---------------------------------------------------------------------------
# Strategy 6: Center-outward progressive matching
# ---------------------------------------------------------------------------

def center_outward_refine(image, cat_az_rad, cat_alt_rad, model,
                          n_rings=4, fix_az0=False):
    """Refine model by matching from center outward in annular rings.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az_rad, cat_alt_rad : ndarray
        Catalog positions (radians).
    model : CameraModel
        Initial model estimate.
    n_rings : int
        Number of annular rings.
    fix_az0 : bool
        If True, fix az0 during fitting (breaks near-zenith degeneracy).

    Returns
    -------
    refined_model : CameraModel
        Refined model.
    n_matched : int
        Total matches.
    rms : float
        RMS residual.
    """
    ny, nx = image.shape
    background = float(np.median(image))
    p999 = float(np.percentile(image, 99.9))
    _mpo = _adaptive_min_peak_offset(background, p999)
    max_r = min(nx, ny) * 0.5

    current = model
    all_det_x = []
    all_det_y = []
    all_az = []
    all_alt = []

    for ring in range(n_rings):
        r_inner = ring * max_r / n_rings
        r_outer = (ring + 1) * max_r / n_rings

        # Select catalog stars that project into this annulus
        px, py = current.sky_to_pixel(cat_az_rad, cat_alt_rad)
        r = np.sqrt((px - current.cx) ** 2 + (py - current.cy) ** 2)
        in_ring = (r >= r_inner) & (r < r_outer)
        in_ring &= np.isfinite(px) & np.isfinite(py)
        in_ring &= (px >= 0) & (px < nx) & (py >= 0) & (py < ny)

        if np.sum(in_ring) < 3:
            continue

        ring_az = cat_az_rad[in_ring]
        ring_alt = cat_alt_rad[in_ring]

        # Guided match in this ring
        search_r = max(8, int(20 - 3 * ring))
        matches = _guided_match(image, current, ring_az, ring_alt,
                                search_r, background + _mpo, background)

        for m in matches:
            all_det_x.append(m[1])
            all_det_y.append(m[2])
            all_az.append(ring_az[m[0]])
            all_alt.append(ring_alt[m[0]])

        # Re-fit with all matches so far
        if len(all_det_x) >= 6:
            dx = np.array(all_det_x)
            dy = np.array(all_det_y)
            az = np.array(all_az)
            alt = np.array(all_alt)
            try:
                current = _fit_model_to_pairs(dx, dy, az, alt, current,
                                              fix_az0=fix_az0)
            except Exception:
                pass

    n_matched = len(all_det_x)
    if n_matched > 0:
        dx = np.array(all_det_x)
        dy = np.array(all_det_y)
        az = np.array(all_az)
        alt = np.array(all_alt)
        px, py = current.sky_to_pixel(az, alt)
        rms = float(np.sqrt(np.mean((px - dx) ** 2 + (py - dy) ** 2)))
    else:
        rms = 999.0

    return current, n_matched, rms


# ---------------------------------------------------------------------------
# Guided matching and fitting helpers (shared across strategies)
# ---------------------------------------------------------------------------

def _brightness_score(image_sub, px, py, nx, ny, box_half=5):
    """Sum of peak pixel values in boxes at projected positions."""
    total = 0.0
    r = box_half
    for i in range(len(px)):
        xi, yi = int(round(px[i])), int(round(py[i]))
        if r <= xi < nx - r and r <= yi < ny - r:
            box = image_sub[yi - r:yi + r + 1, xi - r:xi + r + 1]
            val = float(np.max(box))
            if val > 0:
                total += val
    return total


def _point_source_score(image_sub, px, py, nx, ny, inner_r=3, outer_r=8):
    """Score based on point-source contrast (peak vs annulus).

    Unlike _brightness_score, this rejects extended emission (dome,
    lens reflections) by requiring the peak to be compact — the inner
    box must be significantly brighter than the surrounding annulus.

    Parameters
    ----------
    image_sub : ndarray
        Background-subtracted image.
    px, py : ndarray
        Projected x, y positions.
    nx, ny : int
        Image dimensions.
    inner_r : int
        Half-size of inner (peak) box.
    outer_r : int
        Half-size of outer (annulus) box.

    Returns
    -------
    float
        Sum of contrast scores for positions with point-source peaks.
    """
    total = 0.0
    r = outer_r
    for i in range(len(px)):
        xi, yi = int(round(px[i])), int(round(py[i]))
        if r <= xi < nx - r and r <= yi < ny - r:
            outer_box = image_sub[yi - r:yi + r + 1, xi - r:xi + r + 1]
            inner_box = image_sub[
                yi - inner_r:yi + inner_r + 1,
                xi - inner_r:xi + inner_r + 1,
            ]
            inner_peak = float(np.max(inner_box))
            if inner_peak <= 0:
                continue
            # Annulus: outer box minus inner box contribution
            outer_sum = float(np.sum(outer_box))
            inner_sum = float(np.sum(inner_box))
            outer_npix = outer_box.size - inner_box.size
            if outer_npix <= 0:
                continue
            annulus_mean = (outer_sum - inner_sum) / outer_npix
            # Point-source contrast: peak must be much brighter than annulus
            contrast = inner_peak - annulus_mean
            if contrast > annulus_mean * 0.5 and contrast > 500:
                total += contrast
    return total


def _guided_match(image, model, cat_az, cat_alt, search_radius, min_peak,
                  background, alt_min_rad=None, alt_max_rad=None):
    """Find the bright peak nearest each projected catalog star.

    In dense star fields, the *brightest* pixel in the search box may
    belong to a different star.  Instead, find the local maximum that
    is closest to the predicted position and above ``min_peak``.

    Parameters
    ----------
    alt_min_rad, alt_max_rad : float or None
        If set, skip catalog stars outside this altitude range (radians).

    Returns list of (cat_idx, det_x, det_y, peak_val).
    """
    ny, nx = image.shape
    matches = []
    px, py = model.sky_to_pixel(cat_az, cat_alt)
    r = search_radius

    for i in range(len(cat_az)):
        # Altitude filtering
        if alt_min_rad is not None and cat_alt[i] < alt_min_rad:
            continue
        if alt_max_rad is not None and cat_alt[i] > alt_max_rad:
            continue

        xi, yi = int(round(px[i])), int(round(py[i]))
        if xi - r < 0 or xi + r >= nx or yi - r < 0 or yi + r >= ny:
            continue

        box = image[yi - r:yi + r + 1, xi - r:xi + r + 1]
        max_val = float(np.max(box))
        if max_val < min_peak:
            continue

        box_float = box.astype(np.float64) - background
        box_float[box_float < 0] = 0

        # Find local maxima (pixels brighter than all 4 neighbours)
        # and pick the one nearest to the box center (predicted pos).
        padded = np.pad(box_float, 1, mode='constant',
                        constant_values=0)
        is_max = (box_float > padded[:-2, 1:-1]) & \
                 (box_float > padded[2:, 1:-1]) & \
                 (box_float > padded[1:-1, :-2]) & \
                 (box_float > padded[1:-1, 2:]) & \
                 (box > min_peak)

        peak_ys, peak_xs = np.where(is_max)
        if len(peak_ys) == 0:
            # Fallback: just use the global maximum
            max_pos = np.unravel_index(np.argmax(box), box.shape)
            my, mx = max_pos
        else:
            # Pick the local maximum closest to the center (predicted pos)
            dists = (peak_xs - r) ** 2 + (peak_ys - r) ** 2
            nearest = np.argmin(dists)
            my, mx = int(peak_ys[nearest]), int(peak_xs[nearest])

        # Sub-pixel centroid around the chosen peak
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

        # Point-source check: the peak must be significantly brighter
        # than the local background (edge pixels of a 15×15 box).
        # Rejects dome glow, illuminated clouds, and telescope
        # structure which have similar peak and edge brightness.
        abs_x = xi - r + mx
        abs_y = yi - r + my
        edge_r = 7
        if (abs_x - edge_r >= 0 and abs_x + edge_r < nx and
                abs_y - edge_r >= 0 and abs_y + edge_r < ny):
            peak_val = float(image[abs_y, abs_x])
            edge_box = image[abs_y - edge_r:abs_y + edge_r + 1,
                             abs_x - edge_r:abs_x + edge_r + 1]
            edge_mask_arr = np.ones(edge_box.shape, dtype=bool)
            edge_mask_arr[2:-2, 2:-2] = False
            local_bg = float(np.median(edge_box[edge_mask_arr]))
            if peak_val - local_bg < min_peak - background:
                continue

            # Compactness check in bright regions: when the local
            # background is significantly above the global background
            # (moonlit clouds, bright nebulosity), structured
            # backgrounds can mimic point-source contrast.  Require
            # the peak to drop sharply to its 4 direct neighbors —
            # real stars have steep gradients; cloud glow is smooth.
            if local_bg > 1.5 * background:
                if (abs_x >= 1 and abs_x < nx - 1 and
                        abs_y >= 1 and abs_y < ny - 1):
                    neighbors = np.array([
                        float(image[abs_y - 1, abs_x]),
                        float(image[abs_y + 1, abs_x]),
                        float(image[abs_y, abs_x - 1]),
                        float(image[abs_y, abs_x + 1]),
                    ], dtype=np.float64)
                    neighbor_med = float(np.median(neighbors))
                    drop = peak_val - neighbor_med
                    noise_local = max(1.0, np.sqrt(abs(neighbor_med)))
                    if drop / noise_local < 3.0:
                        continue

        det_x = xi - r + cx_c
        det_y = yi - r + cy_c
        matches.append((i, det_x, det_y, float(box[my, mx])))

    return matches


def _fit_model_to_pairs(det_x, det_y, cat_az, cat_alt, model,
                        fit_distortion=False, fix_az0=False,
                        horizon_r=None, horizon_weight=1.0):
    """Least-squares fit of camera model to matched pairs.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detected pixel positions.
    cat_az, cat_alt : ndarray
        Catalog sky positions (radians).
    model : CameraModel
        Initial model.
    fit_distortion : bool
        If True, also fit k1 and k2.
    fix_az0 : bool
        If True, fix az0 at its initial value. This breaks the az0/rho
        degeneracy for near-zenith cameras, forcing all rotation into rho.
    horizon_r : float or None
        If set, add a penalty for the model's horizon distance deviating
        from this observed value.  Breaks the f/k1 degeneracy.
    horizon_weight : float
        Weight of the horizon constraint relative to star residuals.

    Returns
    -------
    CameraModel
        Refined model.
    """
    proj_type = model.proj_type
    fixed_az0 = model.az0

    # Compute physically meaningful distortion bounds based on focal length.
    # At r = f (zenith angle ~1 rad), we want |k1*r^2| < 0.3 and |k2*r^4| < 0.15
    # Fisheye lenses typically have barrel distortion (k1 ≤ 0) with equidistant
    # projection, but lenses following equisolid or stereographic mappings
    # appear slightly pincushion relative to equidistant.  Allow small positive
    # k1 (up to 10% of k1_max) to handle these cases.
    f0 = model.f
    k1_max = 0.3 / (f0 * f0)          # typically ~2.5e-7 for f=1100
    k1_hi = k1_max * 0.1              # allow slight pincushion
    k2_max = 0.15 / (f0 ** 4)          # typically ~1e-13 for f=1100

    if fit_distortion:
        if fix_az0:
            # Fit: cx, cy, alt0, rho, f, k1, k2  (az0 fixed)
            p0 = np.array([
                model.cx, model.cy, model.alt0,
                model.rho, model.f, model.k1, model.k2,
            ])
            bounds_lo = [
                p0[0] - 200, p0[1] - 200,
                p0[2] - 0.3,
                p0[3] - 0.5, p0[4] * 0.7,
                -k1_max, -k2_max,
            ]
            bounds_hi = [
                p0[0] + 200, p0[1] + 200,
                p0[2] + 0.3,
                p0[3] + 0.5, p0[4] * 1.3,
                k1_hi, k2_max,
            ]

            def residuals(params):
                m = CameraModel(
                    cx=params[0], cy=params[1], az0=fixed_az0,
                    alt0=params[2], rho=params[3], f=params[4],
                    proj_type=proj_type, k1=params[5], k2=params[6],
                )
                px, py = m.sky_to_pixel(cat_az, cat_alt)
                res = np.concatenate([px - det_x, py - det_y])
                if horizon_r is not None:
                    from .projection import _theta_to_r, _apply_distortion
                    r_h = _theta_to_r(np.pi / 2, params[4], proj_type)
                    r_hd = _apply_distortion(r_h, params[5], params[6])
                    # Add horizon penalty (scaled like pixel residuals)
                    n_stars = len(det_x)
                    h_pen = (r_hd - horizon_r) * horizon_weight
                    res = np.append(res, [h_pen] * max(1, n_stars // 4))
                return res
        else:
            p0 = np.array([
                model.cx, model.cy, model.az0, model.alt0,
                model.rho, model.f, model.k1, model.k2,
            ])
            bounds_lo = [
                p0[0] - 200, p0[1] - 200,
                p0[2] - 0.5, p0[3] - 0.3,
                p0[4] - 0.5, p0[5] * 0.7,
                -k1_max, -k2_max,
            ]
            bounds_hi = [
                p0[0] + 200, p0[1] + 200,
                p0[2] + 0.5, p0[3] + 0.3,
                p0[4] + 0.5, p0[5] * 1.3,
                k1_hi, k2_max,
            ]

            def residuals(params):
                m = CameraModel(
                    cx=params[0], cy=params[1], az0=params[2],
                    alt0=params[3], rho=params[4], f=params[5],
                    proj_type=proj_type, k1=params[6], k2=params[7],
                )
                px, py = m.sky_to_pixel(cat_az, cat_alt)
                res = np.concatenate([px - det_x, py - det_y])
                if horizon_r is not None:
                    from .projection import _theta_to_r, _apply_distortion
                    r_h = _theta_to_r(np.pi / 2, params[5], proj_type)
                    r_hd = _apply_distortion(r_h, params[6], params[7])
                    n_stars = len(det_x)
                    h_pen = (r_hd - horizon_r) * horizon_weight
                    res = np.append(res, [h_pen] * max(1, n_stars // 4))
                return res
    else:
        if fix_az0:
            # Fit: cx, cy, alt0, rho, f  (az0 fixed)
            p0 = np.array([
                model.cx, model.cy, model.alt0,
                model.rho, model.f,
            ])
            bounds_lo = [
                p0[0] - 200, p0[1] - 200,
                p0[2] - 0.3,
                p0[3] - 0.5, p0[4] * 0.7,
            ]
            bounds_hi = [
                p0[0] + 200, p0[1] + 200,
                p0[2] + 0.3,
                p0[3] + 0.5, p0[4] * 1.3,
            ]

            def residuals(params):
                m = CameraModel(
                    cx=params[0], cy=params[1], az0=fixed_az0,
                    alt0=params[2], rho=params[3], f=params[4],
                    proj_type=proj_type,
                )
                px, py = m.sky_to_pixel(cat_az, cat_alt)
                res = np.concatenate([px - det_x, py - det_y])
                if horizon_r is not None:
                    from .projection import _theta_to_r
                    r_h = _theta_to_r(np.pi / 2, params[4], proj_type)
                    n_stars = len(det_x)
                    h_pen = (r_h - horizon_r) * horizon_weight
                    res = np.append(res, [h_pen] * max(1, n_stars // 4))
                return res
        else:
            p0 = np.array([
                model.cx, model.cy, model.az0, model.alt0,
                model.rho, model.f,
            ])
            bounds_lo = [
                p0[0] - 200, p0[1] - 200,
                p0[2] - 0.5, p0[3] - 0.3,
                p0[4] - 0.5, p0[5] * 0.7,
            ]
            bounds_hi = [
                p0[0] + 200, p0[1] + 200,
                p0[2] + 0.5, p0[3] + 0.3,
                p0[4] + 0.5, p0[5] * 1.3,
            ]

            def residuals(params):
                m = CameraModel(
                    cx=params[0], cy=params[1], az0=params[2],
                    alt0=params[3], rho=params[4], f=params[5],
                    proj_type=proj_type,
                )
                px, py = m.sky_to_pixel(cat_az, cat_alt)
                res = np.concatenate([px - det_x, py - det_y])
                if horizon_r is not None:
                    from .projection import _theta_to_r
                    r_h = _theta_to_r(np.pi / 2, params[5], proj_type)
                    n_stars = len(det_x)
                    h_pen = (r_h - horizon_r) * horizon_weight
                    res = np.append(res, [h_pen] * max(1, n_stars // 4))
                return res

    # When fitting distortion with a horizon constraint, use linear (ordinary
    # least squares) loss so the horizon penalty is not downweighted.
    # The soft_l1 loss with f_scale=5 treats horizon penalty terms (~50-100 px)
    # as outliers and essentially ignores them, breaking the f/k1 constraint.
    # Without a horizon constraint, keep soft_l1 for robustness against
    # occasional bad matches.
    if fit_distortion and horizon_r is not None:
        loss_fn = "linear"
        f_scale_val = 1.0
    else:
        loss_fn = "soft_l1"
        f_scale_val = 5.0

    result = least_squares(
        residuals, p0,
        bounds=(bounds_lo, bounds_hi),
        loss=loss_fn,
        f_scale=f_scale_val,
        max_nfev=3000,
    )

    params = result.x
    if fit_distortion and fix_az0:
        return CameraModel(
            cx=params[0], cy=params[1], az0=fixed_az0,
            alt0=params[2], rho=params[3], f=params[4],
            proj_type=proj_type, k1=params[5], k2=params[6],
        )
    elif fit_distortion:
        return CameraModel(
            cx=params[0], cy=params[1], az0=params[2],
            alt0=params[3], rho=params[4], f=params[5],
            proj_type=proj_type, k1=params[6], k2=params[7],
        )
    elif fix_az0:
        return CameraModel(
            cx=params[0], cy=params[1], az0=fixed_az0,
            alt0=params[2], rho=params[3], f=params[4],
            proj_type=proj_type, k1=model.k1, k2=model.k2,
        )
    else:
        return CameraModel(
            cx=params[0], cy=params[1], az0=params[2],
            alt0=params[3], rho=params[4], f=params[5],
            proj_type=proj_type, k1=model.k1, k2=model.k2,
        )


# ---------------------------------------------------------------------------
# Model quality comparison
# ---------------------------------------------------------------------------

def _is_better(n_new, rms_new, n_old, rms_old, rms_threshold=7.0,
               rms_reject=15.0):
    """Decide if a new model is better than the current best.

    Uses a quality score that balances match count with RMS.  A model
    with many matches but terrible RMS (likely false positives) does NOT
    beat a model with fewer matches but good RMS.
    """
    if n_old == 0:
        return n_new > 0

    # Hard reject: RMS above reject threshold is never acceptable
    # (unless the old model also has terrible RMS)
    if rms_new > rms_reject and rms_old <= rms_reject:
        return False

    # If new RMS is much worse, reject — high-RMS results are likely
    # contaminated with false matches from dense star fields.
    # But skip this check when the old model has too few matches to be
    # reliable (< 50 matches can easily be coincidental in dense fields).
    if n_old >= 50:
        if rms_new > rms_threshold and rms_new > rms_old * 1.3:
            return False

    # Quality score: n_matched / (1 + rms²) — quadratic RMS penalty
    # discourages trading precision for match count.
    score_new = n_new / (1.0 + rms_new * rms_new)
    score_old = n_old / (1.0 + rms_old * rms_old)
    return score_new > score_old


# ---------------------------------------------------------------------------
# Blind pattern-matching solve
# ---------------------------------------------------------------------------


def pattern_match_solve(det_x, det_y, cat_az, cat_alt, cat_vmag,
                        cx, cy, nx, ny,
                        f_range=(700, 1400), n_bright_det=25,
                        vmag_limit=3.0, match_radius=15.0,
                        verbose=False, rho_hint=None,
                        alt0_range=None):
    """Blind camera model solve using hypothesis-and-verify.

    For each bright detection × bright catalog star, hypothesizes they
    match, computes implied (f, rho), builds a trial model, and counts
    how many catalog stars land near detections.  The hypothesis with
    the most matches wins.

    Parameters
    ----------
    det_x, det_y : ndarray
        Detection pixel positions (sorted by brightness, brightest first).
    cat_az, cat_alt : ndarray
        Catalog star positions in radians.
    cat_vmag : ndarray
        Catalog apparent magnitudes (extinction-corrected).
    cx, cy : float
        Approximate optical center (e.g. from horizon circle).
    nx, ny : int
        Image dimensions.
    f_range : tuple
        (f_min, f_max) to consider.
    n_bright_det : int
        Number of brightest detections to try as anchor.
    vmag_limit : float
        Magnitude limit for bright catalog stars to try as anchor.
    match_radius : float
        Pixel radius for verification matching.
    verbose : bool
        Print progress messages.

    Returns
    -------
    model : CameraModel or None
        Best camera model, or None if no solution found.
    n_matched : int
        Number of matched stars.
    rms : float
        RMS residual in pixels.
    diagnostics : dict
        Solver diagnostics.
    """
    diag = {}
    f_min, f_max = f_range

    # Build alt0 grids from range.
    if alt0_range is None:
        alt0_range = (85, 93)
    alt0_lo, alt0_hi = alt0_range
    alt0_coarse = list(range(alt0_lo, alt0_hi + 1, 2))
    if 90 not in alt0_coarse and alt0_lo <= 90 <= alt0_hi:
        alt0_coarse.append(90)
        alt0_coarse.sort()
    alt0_fine = list(range(max(80, alt0_lo - 2), min(95, alt0_hi + 3)))

    # --- Phase 1: Build KDTree for verification ---
    n_verify_det = min(200, len(det_x))
    all_det_xy = np.column_stack([det_x[:n_verify_det],
                                  det_y[:n_verify_det]])
    det_tree = KDTree(all_det_xy)

    # Verification catalog: vmag < 5 (more stars for counting)
    verify_mask = cat_vmag < 5.0
    verify_az = cat_az[verify_mask]
    verify_alt = cat_alt[verify_mask]

    # Anchor detections: bright, not too close to center (PA ill-defined)
    n_det = min(n_bright_det, len(det_x))
    dx = det_x[:n_det]
    dy = det_y[:n_det]
    det_r = np.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
    det_pa = np.arctan2(dx - cx, dy - cy)

    # Anchor catalog: bright stars not near zenith
    bright_mask = cat_vmag < vmag_limit
    if np.sum(bright_mask) < 15:
        bright_mask = cat_vmag < vmag_limit + 1.0
    bright_idx = np.where(bright_mask)[0]
    mag_order = np.argsort(cat_vmag[bright_idx])
    bright_idx = bright_idx[mag_order[:min(50, len(mag_order))]]
    bcat_az = cat_az[bright_idx]
    bcat_alt = cat_alt[bright_idx]
    bcat_za = np.pi / 2 - bcat_alt  # zenith angle
    n_cat = len(bright_idx)

    if verbose:
        log.info(f"Pattern match: {n_det} anchor dets, "
                 f"{n_cat} anchor cat stars, "
                 f"{np.sum(verify_mask)} verify stars")

    # --- Phase 2: Hypothesis-and-verify ---
    # For each center candidate × detection × catalog star, hypothesize
    # they match, compute implied (f, rho) with az0=0, build a trial
    # model, and count how many catalog stars project near detections.
    # The hypothesis with the most matches wins.
    best_count = 0
    best_f = None
    best_rho = None
    best_alt0 = np.pi / 2
    best_cx = cx
    best_cy = cy
    n_hypotheses = 0

    # Center search grid: offsets from horizon-detected center.
    # Scale with image size — large-sensor cameras can have the
    # optical center 200+ px from the horizon center.
    cx_span = max(100, min(250, nx // 16))
    cy_span = max(175, min(350, ny // 12))
    cx_offsets = np.arange(-cx_span, cx_span + 1, 50)
    cy_offsets = np.arange(-cy_span, cy_span + 1, 50)

    for dcx in cx_offsets:
        for dcy in cy_offsets:
            cx_try = cx + dcx
            cy_try = cy + dcy
            # Recompute radii and PAs for this center
            dr = np.sqrt((dx - cx_try) ** 2 + (dy - cy_try) ** 2)
            dpa = np.arctan2(dx - cx_try, dy - cy_try)

            for i in range(n_det):
                if dr[i] < 50:
                    continue
                for j in range(n_cat):
                    if bcat_za[j] < np.radians(3.0):
                        continue

                    implied_f = dr[i] / bcat_za[j]
                    if implied_f < f_min or implied_f > f_max:
                        continue

                    # phi = az + az0 + rho; with az0=0: rho = PA - az
                    implied_rho = (dpa[i] - bcat_az[j]) % (2 * np.pi)

                    # If a rotation hint is given, skip hypotheses
                    # that are far from the hint.
                    if rho_hint is not None:
                        drho = abs((implied_rho - rho_hint + np.pi)
                                   % (2 * np.pi) - np.pi)
                        if drho > np.radians(30):
                            continue

                    n_hypotheses += 1

                    # Test with a few alt0 values
                    for alt0_deg in alt0_coarse:
                        alt0 = np.radians(alt0_deg)
                        m = CameraModel(
                            cx=cx_try, cy=cy_try, az0=0.0,
                            alt0=alt0,
                            rho=implied_rho, f=implied_f,
                            proj_type=ProjectionType.EQUIDISTANT,
                        )
                        vx, vy = m.sky_to_pixel(verify_az, verify_alt)
                        in_frame = ((vx > 0) & (vx < nx) &
                                    (vy > 0) & (vy < ny))
                        if np.sum(in_frame) < 3:
                            continue
                        v_xy = np.column_stack([vx[in_frame],
                                                vy[in_frame]])
                        vdists, _ = det_tree.query(v_xy)
                        count = int(np.sum(
                            vdists < match_radius))
                        if count > best_count:
                            best_count = count
                            best_f = implied_f
                            best_rho = implied_rho
                            best_alt0 = alt0
                            best_cx = cx_try
                            best_cy = cy_try

    diag["n_hypotheses"] = n_hypotheses
    diag["best_count"] = best_count

    if verbose:
        if best_f is not None:
            log.info(f"  Hypotheses tested: {n_hypotheses}, "
                     f"best: {best_count} matches at "
                     f"f={best_f:.0f}, "
                     f"rho={np.degrees(best_rho):.1f}°, "
                     f"alt0={np.degrees(best_alt0):.1f}°, "
                     f"cx={best_cx:.0f}, cy={best_cy:.0f}")
        else:
            log.info(f"  Hypotheses tested: {n_hypotheses}, no valid match")

    if best_count < 5 or best_f is None:
        if verbose:
            log.info("  Too few matches, pattern match failed")
        return None, 0, 999.0, diag

    # --- Phase 2b: Fine-tune around best hypothesis ---
    fine_best_count = best_count
    fine_best_f = best_f
    fine_best_rho = best_rho
    fine_best_alt0 = best_alt0
    fine_best_cx = best_cx
    fine_best_cy = best_cy

    for dcx in np.arange(-50, 75, 25):
        for dcy in np.arange(-50, 75, 25):
            cx_try = best_cx + dcx
            cy_try = best_cy + dcy
            for df in np.arange(-30, 35, 5):
                f_try = best_f + df
                if f_try < f_min or f_try > f_max:
                    continue
                for drho in np.arange(-3, 3.5, 0.5):
                    rho_try = best_rho + np.radians(drho)
                    for alt0_deg in alt0_fine:
                        alt0 = np.radians(alt0_deg)
                        m = CameraModel(
                            cx=cx_try, cy=cy_try, az0=0.0,
                            alt0=alt0,
                            rho=rho_try, f=f_try,
                            proj_type=ProjectionType.EQUIDISTANT,
                        )
                        vx, vy = m.sky_to_pixel(
                            verify_az, verify_alt)
                        in_frame = ((vx > 0) & (vx < nx) &
                                    (vy > 0) & (vy < ny))
                        if np.sum(in_frame) < 3:
                            continue
                        v_xy = np.column_stack([vx[in_frame],
                                                vy[in_frame]])
                        vdists, _ = det_tree.query(v_xy)
                        count = int(np.sum(
                            vdists < match_radius))
                        if count > fine_best_count:
                            fine_best_count = count
                            fine_best_f = f_try
                            fine_best_rho = rho_try
                            fine_best_alt0 = alt0
                            fine_best_cx = cx_try
                            fine_best_cy = cy_try

    diag["fine_best_count"] = fine_best_count

    if verbose:
        log.info(f"  Fine-tuned: {fine_best_count} matches at "
                 f"f={fine_best_f:.0f}, "
                 f"rho={np.degrees(fine_best_rho):.1f}°, "
                 f"alt0={np.degrees(fine_best_alt0):.1f}°, "
                 f"cx={fine_best_cx:.0f}, cy={fine_best_cy:.0f}")

    # --- Phase 3: Iterative least-squares refinement ---
    seed_model = CameraModel(
        cx=fine_best_cx, cy=fine_best_cy,
        az0=0.0, alt0=fine_best_alt0,
        rho=fine_best_rho, f=fine_best_f,
        proj_type=ProjectionType.EQUIDISTANT,
    )

    wide_mask = cat_vmag < 6.0
    wide_az = cat_az[wide_mask]
    wide_alt = cat_alt[wide_mask]

    current_model = seed_model
    n_matched = 0
    rms = 999.0

    for iteration in range(8):
        px, py = current_model.sky_to_pixel(wide_az, wide_alt)
        in_frame = ((px > 0) & (px < nx) & (py > 0) & (py < ny))

        if np.sum(in_frame) < 5:
            break

        cat_xy_if = np.column_stack([px[in_frame], py[in_frame]])
        az_if = wide_az[in_frame]
        alt_if = wide_alt[in_frame]

        dists, det_idxs = det_tree.query(cat_xy_if)
        mr = max(8.0, match_radius * (0.85 ** iteration))
        matched = dists < mr

        if np.sum(matched) < 6:
            break

        m_det_x = all_det_xy[det_idxs[matched], 0]
        m_det_y = all_det_xy[det_idxs[matched], 1]
        m_cat_az = az_if[matched]
        m_cat_alt = alt_if[matched]

        # Sigma-clip
        px_m, py_m = current_model.sky_to_pixel(m_cat_az, m_cat_alt)
        res = np.sqrt((px_m - m_det_x) ** 2 + (py_m - m_det_y) ** 2)
        med_r = np.median(res)
        mad_r = np.median(np.abs(res - med_r)) * 1.4826
        clip = max(5.0, med_r + 3.0 * mad_r)
        good = res < clip

        if np.sum(good) < 6:
            break

        fit_dist = iteration >= 3
        try:
            current_model = _fit_model_to_pairs(
                m_det_x[good], m_det_y[good],
                m_cat_az[good], m_cat_alt[good],
                current_model,
                fix_az0=False, fit_distortion=fit_dist,
            )
        except Exception:
            break

        n_matched = int(np.sum(good))
        px_f, py_f = current_model.sky_to_pixel(
            m_cat_az[good], m_cat_alt[good])
        rms = float(np.sqrt(np.mean(
            (px_f - m_det_x[good]) ** 2 +
            (py_f - m_det_y[good]) ** 2
        )))

        if verbose and (iteration < 3 or iteration == 7):
            log.info(f"  Refine iter {iteration}: {n_matched} stars, "
                     f"RMS={rms:.1f}px, f={current_model.f:.0f}")

    diag["final_n_matched"] = n_matched
    diag["final_rms"] = rms

    if verbose:
        log.info(f"  Final: {n_matched} matches, RMS={rms:.1f}px, "
                 f"f={current_model.f:.0f}, "
                 f"rho={np.degrees(current_model.rho) % 360:.1f}, "
                 f"az0={np.degrees(current_model.az0):.1f}, "
                 f"alt0={np.degrees(current_model.alt0):.1f}, "
                 f"k1={current_model.k1:.2e}")

    return current_model, n_matched, rms, diag


# ---------------------------------------------------------------------------
# Combined instrument-fit pipeline
# ---------------------------------------------------------------------------

def guided_refine(image, cat_az, cat_alt, model, n_iterations=12,
                  bright_mag_limit=None, min_peak_offset=None,
                  alt_min_deg=None, alt_max_deg=None, fix_az0=False,
                  fit_distortion=False,
                  horizon_r=None, horizon_weight=1.0,
                  initial_search_radius=30):
    """Iterative guided-match refinement of a camera model.

    Parameters
    ----------
    image : ndarray
        Raw image data.
    cat_az, cat_alt : ndarray
        Catalog star positions (radians).
    model : CameraModel
        Initial camera model.
    n_iterations : int
        Number of refinement iterations.
    bright_mag_limit : ignored (kept for API compat)
    min_peak_offset : float or None
        Minimum peak above background for guided matching.
        If None, computed adaptively from image dynamic range.
    alt_min_deg, alt_max_deg : float or None
        If set, only match stars within this altitude range (degrees).
        Use to exclude near-zenith (degenerate) and near-horizon (noisy)
        regions during initial fitting.
    fix_az0 : bool
        If True, fix az0 during fitting to break the az0/rho degeneracy
        near zenith. All rotation goes into rho instead.
    fit_distortion : bool
        If True, jointly fit k1 and k2 distortion coefficients along
        with geometric parameters.
    horizon_r : float or None
        Observed horizon radius in pixels. Passed to _fit_model_to_pairs
        to constrain f (and break f/k1 degeneracy when fit_distortion=True).
    horizon_weight : float
        Weight of the horizon constraint relative to star residuals.

    Returns
    -------
    model : CameraModel
        Refined model.
    n_matched : int
        Number of guided matches in final iteration.
    rms : float
        RMS residual in final iteration.
    """
    ny, nx = image.shape
    background = float(np.median(image))
    current = CameraModel(
        cx=model.cx, cy=model.cy, az0=model.az0, alt0=model.alt0,
        rho=model.rho, f=model.f, proj_type=model.proj_type,
        k1=model.k1, k2=model.k2,
    )
    n_matched = 0
    rms = 999.0

    if min_peak_offset is None:
        p999 = float(np.percentile(image, 99.9))
        min_peak_offset = _adaptive_min_peak_offset(background, p999)

    alt_min_rad = np.radians(alt_min_deg) if alt_min_deg is not None else None
    alt_max_rad = np.radians(alt_max_deg) if alt_max_deg is not None else None

    for iteration in range(n_iterations):
        search_r = max(10, int(initial_search_radius * (0.90 ** iteration)))
        min_peak = background + min_peak_offset

        matches = _guided_match(image, current, cat_az, cat_alt,
                                search_r, min_peak, background,
                                alt_min_rad=alt_min_rad,
                                alt_max_rad=alt_max_rad)
        if len(matches) < 6:
            break

        det_x_m = np.array([m[1] for m in matches])
        det_y_m = np.array([m[2] for m in matches])
        cat_az_m = np.array([cat_az[m[0]] for m in matches])
        cat_alt_m = np.array([cat_alt[m[0]] for m in matches])

        # Sigma-clip outlier matches — essential in dense star fields
        # where the nearest local maximum may be a different star.
        if len(matches) > 10:
            px_pre, py_pre = current.sky_to_pixel(cat_az_m, cat_alt_m)
            resid = np.sqrt((px_pre - det_x_m) ** 2 +
                            (py_pre - det_y_m) ** 2)
            med_r = np.median(resid)
            clip_threshold = max(10.0, med_r + 2.5 * np.std(resid))
            good = resid < clip_threshold
            if np.sum(good) >= 6:
                det_x_m = det_x_m[good]
                det_y_m = det_y_m[good]
                cat_az_m = cat_az_m[good]
                cat_alt_m = cat_alt_m[good]

        try:
            current = _fit_model_to_pairs(
                det_x_m, det_y_m, cat_az_m, cat_alt_m, current,
                fix_az0=fix_az0, fit_distortion=fit_distortion,
                horizon_r=horizon_r, horizon_weight=horizon_weight,
            )
        except Exception:
            break

        n_matched = len(det_x_m)
        px, py = current.sky_to_pixel(cat_az_m, cat_alt_m)
        rms = float(np.sqrt(np.mean(
            (px - det_x_m) ** 2 + (py - det_y_m) ** 2
        )))

    return current, n_matched, rms


def detection_refine(det_table, cat_az, cat_alt, model, n_iterations=12,
                     alt_min_deg=None, alt_max_deg=None, fix_az0=False,
                     fit_distortion=False,
                     horizon_r=None, horizon_weight=1.0,
                     max_match_dist=20.0):
    """Iterative refinement using pre-detected star positions.

    Unlike guided_refine which searches for peaks in the raw image
    (making detected positions dependent on the model), this matches
    pre-detected sources (from DAOStarFinder) to projected catalog
    positions.  The detection positions are fixed and model-independent,
    which prevents the optimizer from creating self-reinforcing errors.

    Parameters
    ----------
    det_table : astropy Table
        Detection table with 'x' and 'y' columns.
    cat_az, cat_alt : ndarray
        Catalog star positions (radians).
    model : CameraModel
        Initial camera model.
    n_iterations : int
        Number of refinement iterations.
    alt_min_deg, alt_max_deg : float or None
        Altitude range for catalog stars to include.
    fix_az0, fit_distortion : bool
        Passed to _fit_model_to_pairs.
    horizon_r, horizon_weight : float
        Horizon constraint parameters.
    max_match_dist : float
        Initial max match distance in pixels (shrinks each iteration).

    Returns
    -------
    model, n_matched, rms
    """
    det_x = np.asarray(det_table["x"], dtype=np.float64)
    det_y = np.asarray(det_table["y"], dtype=np.float64)
    det_xy = np.column_stack([det_x, det_y])

    current = CameraModel(
        cx=model.cx, cy=model.cy, az0=model.az0, alt0=model.alt0,
        rho=model.rho, f=model.f, proj_type=model.proj_type,
        k1=model.k1, k2=model.k2,
    )
    n_matched = 0
    rms = 999.0

    # Filter catalog by altitude
    alt_min_rad = np.radians(alt_min_deg) if alt_min_deg is not None else 0.0
    alt_max_rad = np.radians(alt_max_deg) if alt_max_deg is not None else np.pi / 2
    alt_mask = (cat_alt >= alt_min_rad) & (cat_alt <= alt_max_rad)
    use_az = cat_az[alt_mask]
    use_alt = cat_alt[alt_mask]

    for iteration in range(n_iterations):
        # Project catalog to pixel coordinates
        px, py = current.sky_to_pixel(use_az, use_alt)
        cat_xy = np.column_stack([px, py])

        # Shrinking match distance
        dist = max(8.0, max_match_dist * (0.85 ** iteration))
        pairs, dists = match_sources(det_xy, cat_xy, max_dist=dist)

        if len(pairs) < 6:
            break

        det_x_m = np.array([det_x[di] for di, ci in pairs])
        det_y_m = np.array([det_y[di] for di, ci in pairs])
        cat_az_m = np.array([use_az[ci] for di, ci in pairs])
        cat_alt_m = np.array([use_alt[ci] for di, ci in pairs])

        try:
            current = _fit_model_to_pairs(
                det_x_m, det_y_m, cat_az_m, cat_alt_m, current,
                fix_az0=fix_az0, fit_distortion=fit_distortion,
                horizon_r=horizon_r, horizon_weight=horizon_weight,
            )
        except Exception:
            break

        n_matched = len(pairs)
        px_fit, py_fit = current.sky_to_pixel(cat_az_m, cat_alt_m)
        rms = float(np.sqrt(np.mean(
            (px_fit - det_x_m) ** 2 + (py_fit - det_y_m) ** 2
        )))

    return current, n_matched, rms


def detect_horizon_circle(image, cx_est=None, cy_est=None,
                          threshold=None, r_min=None, r_max=None):
    """Detect the sky/dome boundary circle in an all-sky image.

    Uses two methods and picks the larger radius:

    1. **Threshold method**: traces radially outward to find where
       brightness drops below a threshold (works for high-contrast
       dome/sky boundaries).
    2. **Gradient method**: finds the steepest brightness drop along
       each radial ray (works for low-contrast edges like buildings
       against sky, or cameras without a dome).

    Parameters
    ----------
    image : ndarray
        Raw image.
    cx_est, cy_est : float, optional
        Estimated image center.  Defaults to geometric center.
    threshold : float, optional
        Brightness threshold for sky/dome boundary.  Defaults to
        a value between median sky brightness and dark level.
    r_min : float
        Minimum radius to start searching (skip center).
    r_max : float, optional
        Maximum search radius.  Defaults to image half-diagonal.

    Returns
    -------
    cx, cy : float
        Fitted circle center.
    radius : float
        Fitted circle radius (approximately the horizon distance).
    n_points : int
        Number of boundary points used in fit.
    """
    ny, nx = image.shape
    img = image.astype(np.float64)

    if cx_est is None:
        cx_est = nx / 2.0
    if cy_est is None:
        cy_est = ny / 2.0
    if r_max is None:
        r_max = np.sqrt(cx_est**2 + cy_est**2) + 200
    if r_min is None:
        half_min = min(cx_est, cy_est, nx - cx_est, ny - cy_est)
        r_min = max(100, min(400, half_min * 0.3))

    # Determine threshold between sky and dark levels.
    # Sky = median brightness in the center; dark = median of the image
    # corners (outside any sky circle).
    if threshold is None:
        ys, xs = np.mgrid[:ny, :nx]
        r_img = np.sqrt((xs - cx_est)**2 + (ys - cy_est)**2)
        sky_mask = r_img < r_min
        sky_level = float(np.median(img[sky_mask]))
        # Dark level: corners of the image (beyond possible sky circle)
        corner_r = min(cx_est, cy_est, nx - cx_est, ny - cy_est)
        dark_mask = r_img > corner_r * 1.3
        if np.sum(dark_mask) > 100:
            dark_level = float(np.median(img[dark_mask]))
        else:
            dark_level = float(np.percentile(img, 5))
        # Threshold halfway between dark and sky
        threshold = dark_level + (sky_level - dark_level) * 0.5

    n_angles = 180
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    step = 3

    # --- Method 1: Threshold (original) ---
    thresh_points = []
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        for r in np.arange(r_min, r_max, step):
            xi = int(round(cx_est + r * dx))
            yi = int(round(cy_est + r * dy))
            if xi < 3 or xi >= nx - 3 or yi < 3 or yi >= ny - 3:
                break
            val = float(np.median(img[yi - 2:yi + 3, xi - 2:xi + 3]))
            if val < threshold:
                thresh_points.append((float(xi), float(yi)))
                break

    # --- Method 2: Steepest gradient per ray ---
    grad_points = []
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        radii_ray = []
        vals_ray = []
        for r in np.arange(r_min, r_max, step):
            xi = int(round(cx_est + r * dx))
            yi = int(round(cy_est + r * dy))
            if xi < 3 or xi >= nx - 3 or yi < 3 or yi >= ny - 3:
                break
            val = float(np.median(img[yi - 2:yi + 3, xi - 2:xi + 3]))
            radii_ray.append(r)
            vals_ray.append(val)

        if len(vals_ray) < 10:
            continue
        vals_arr = np.array(vals_ray)
        # Smooth to avoid noise-driven gradients
        from scipy.ndimage import uniform_filter1d
        smooth_vals = uniform_filter1d(vals_arr, size=7)
        grad = np.gradient(smooth_vals)
        # Find steepest drop in outer half of ray
        outer_start = len(grad) // 2
        if outer_start < 1:
            continue
        min_idx = np.argmin(grad[outer_start:]) + outer_start
        # Only accept if the drop is significant
        if grad[min_idx] < -0.5:
            r_edge = radii_ray[min_idx]
            xi_e = cx_est + r_edge * dx
            yi_e = cy_est + r_edge * dy
            grad_points.append((float(xi_e), float(yi_e)))

    # Fit circles from both methods with iterative outlier rejection.
    # Buildings/structures intrude into the sky circle, pulling boundary
    # points inward.  We iteratively fit, reject points far INSIDE the
    # circle (these hit obstructions), and refit.
    results = []
    for label, points in [("threshold", thresh_points),
                          ("gradient", grad_points)]:
        if len(points) < 10:
            continue
        bp = np.array(points)

        for _clip_iter in range(3):
            bx, by = bp[:, 0], bp[:, 1]
            A = np.column_stack([-2 * bx, -2 * by, np.ones(len(bx))])
            b_vec = -(bx**2 + by**2)
            lstsq = np.linalg.lstsq(A, b_vec, rcond=None)
            cx_f = lstsq[0][0]
            cy_f = lstsq[0][1]
            c_val = lstsq[0][2]
            R_f = np.sqrt(max(0, cx_f**2 + cy_f**2 - c_val))

            # Reject points that are far inside the fitted circle
            # (buildings/obstructions).  Keep points near or outside.
            dists = np.sqrt((bx - cx_f)**2 + (by - cy_f)**2)
            residuals = dists - R_f  # negative = inside circle
            # Keep points within 1 sigma of the circle or outside it
            sigma = float(np.std(residuals))
            keep = residuals > -1.5 * sigma
            if np.sum(keep) < 10:
                break
            bp = bp[keep]

        # Reject fits where center is far from estimate
        if abs(cx_f - cx_est) > nx * 0.3 or abs(cy_f - cy_est) > ny * 0.3:
            continue
        results.append((cx_f, cy_f, R_f, len(bp), label))

    if not results:
        return cx_est, cy_est, min(cx_est, cy_est) * 0.8, 0

    # Pick the result with the largest radius — the threshold method
    # tends to fire early on internal features, while the gradient
    # method finds the true sky edge.
    best = max(results, key=lambda x: x[2])
    log.info("  Horizon methods: %s",
             ", ".join(f"{r[4]}=(R={r[2]:.0f}, n={r[3]})" for r in results))
    return best[0], best[1], best[2], best[3]


def _make_detection_score_map(det_table, image_shape, sigma=3.0):
    """Build a score map from detected point sources.

    Places detected source flux at each detection position and smooths
    with a Gaussian to handle sub-pixel offsets.  The result is a map
    where real stars are bright peaks and everything else is zero.
    Unlike a contrast image, dome emission does not appear here because
    only DAOStarFinder-verified point sources contribute.
    """
    from scipy.ndimage import gaussian_filter
    ny, nx = image_shape
    score_map = np.zeros((ny, nx), dtype=np.float64)

    det_x = np.asarray(det_table["x"], dtype=np.float64)
    det_y = np.asarray(det_table["y"], dtype=np.float64)
    det_flux = np.asarray(det_table["flux"], dtype=np.float64)

    for j in range(len(det_x)):
        xi = int(round(det_x[j]))
        yi = int(round(det_y[j]))
        if 0 <= xi < nx and 0 <= yi < ny:
            score_map[yi, xi] += det_flux[j]

    return gaussian_filter(score_map, sigma=sigma)


def score_model(score_map, cat_az, cat_alt, cat_vmag, model,
                alt_min_deg=5.0, alt_max_deg=85.0):
    """Score a camera model by how well projected catalog stars land on
    detected point sources.

    This is a matching-free scoring function.  Projects catalog stars
    through the trial model and reads off the detection score map.
    Correct parameters project stars onto real detections (high score).
    Wrong parameters project onto empty background (zero score).

    Parameters
    ----------
    score_map : ndarray
        Pre-computed detection score map from _make_detection_score_map.
    cat_az, cat_alt : ndarray
        Catalog star positions (radians).
    cat_vmag : ndarray
        Catalog magnitudes (for brightness weighting).
    model : CameraModel
        Trial camera model.
    alt_min_deg, alt_max_deg : float
        Altitude range to include.

    Returns
    -------
    score : float
        Sum of detection map values at projected positions, weighted by
        expected star brightness.
    """
    ny, nx = score_map.shape

    # Filter by altitude
    alt_min = np.radians(alt_min_deg)
    alt_max = np.radians(alt_max_deg)
    mask = (cat_alt >= alt_min) & (cat_alt <= alt_max)
    az = cat_az[mask]
    alt = cat_alt[mask]
    vmag = cat_vmag[mask]

    if len(az) == 0:
        return 0.0

    px, py = model.sky_to_pixel(az, alt)
    pxi = np.round(px).astype(int)
    pyi = np.round(py).astype(int)

    in_frame = ((pxi >= 0) & (pxi < nx) & (pyi >= 0) & (pyi < ny))
    if np.sum(in_frame) == 0:
        return 0.0

    vals = score_map[pyi[in_frame], pxi[in_frame]]

    # Weight by expected brightness: flux ~ 10^(-0.4*vmag)
    weights = 10.0 ** (-0.4 * vmag[in_frame])

    return float(np.sum(vals * weights))


def brightness_parameter_sweep(det_table, image_shape,
                               cat_az, cat_alt, cat_vmag,
                               cx, cy, f, alt0=None,
                               rho_steps=720, f_steps=1,
                               f_range=0.0, k1=0.0,
                               alt0_range=0.0, alt0_steps=1):
    """Sweep camera parameters and score each model by detection alignment.

    For each trial parameter combination, projects catalog stars to pixel
    positions and checks if they land on detected point sources.  The
    correct parameters produce the highest score.

    No star matching needed — this is a direct cross-correlation between
    the projected catalog and the detection map.

    Parameters
    ----------
    det_table : Table
        Detection table with x, y, flux columns.
    image_shape : tuple
        (ny, nx) of the image.
    cat_az, cat_alt : ndarray
        Catalog positions (radians).
    cat_vmag : ndarray
        Catalog magnitudes for weighting.
    cx, cy : float
        Optical center.
    f : float
        Central focal length for sweep.
    alt0 : float or None
        Boresight altitude (radians). Defaults to π/2 (zenith).
    rho_steps : int
        Number of roll angle steps (over full 360°).
    f_steps : int
        Number of focal length steps.
    f_range : float
        Fractional range for f sweep (e.g., 0.1 = ±10%).
    k1 : float
        Distortion coefficient.
    alt0_range : float
        Range for alt0 sweep (radians, e.g., 0.1 = ±0.1 rad).
    alt0_steps : int
        Number of alt0 steps.

    Returns
    -------
    best_rho, best_f, best_alt0 : float
        Best-scoring parameters.
    scores : ndarray
        Full score array for rho dimension.
    """
    if alt0 is None:
        alt0 = np.pi / 2

    score_map = _make_detection_score_map(det_table, image_shape)

    rhos = np.linspace(0, 2 * np.pi, rho_steps, endpoint=False)
    fs = np.linspace(f * (1 - f_range), f * (1 + f_range), f_steps) \
        if f_steps > 1 else np.array([f])
    alt0s = np.linspace(alt0 - alt0_range, alt0 + alt0_range, alt0_steps) \
        if alt0_steps > 1 else np.array([alt0])

    best_score = -1.0
    best_rho = 0.0
    best_f = f
    best_alt0 = alt0
    scores_1d = np.zeros(rho_steps)

    for fi in fs:
        for a0 in alt0s:
            for ri, rho in enumerate(rhos):
                model = CameraModel(
                    cx=cx, cy=cy, az0=0.0, alt0=a0,
                    rho=rho, f=fi,
                    proj_type=ProjectionType.EQUIDISTANT,
                    k1=k1,
                )
                s = score_model(score_map, cat_az, cat_alt, cat_vmag, model)
                scores_1d[ri] = s
                if s > best_score:
                    best_score = s
                    best_rho = rho
                    best_f = fi
                    best_alt0 = a0

    return best_rho, best_f, best_alt0, scores_1d


def _guided_match_count(image, model, cat_az, cat_alt, search_radius=15,
                        min_peak_offset=None, require_compact=True):
    """Count guided matches, optionally requiring point-source morphology.

    For each catalog star, project to pixel position and check if there's
    a compact bright peak nearby.  The compactness criterion rejects dome
    emission and extended features that fool simple brightness checks.

    Parameters
    ----------
    image : ndarray
        Raw image.
    model : CameraModel
        Camera model.
    cat_az, cat_alt : ndarray
        Catalog positions (radians).
    search_radius : int
        Pixel search radius around projected position.
    min_peak_offset : float or None
        Minimum peak above background.  If None, computed from image.
    require_compact : bool
        If True, only count matches where the peak is point-source-like
        (inner box significantly brighter than surrounding annulus).

    Returns
    -------
    n_matches : int
        Number of stars with a compact bright peak nearby.
    median_offset : float
        Median pixel offset between predicted and actual peak positions.
        Lower is better (more accurate model).
    """
    ny, nx = image.shape
    background = float(np.median(image))
    if min_peak_offset is None:
        p999 = float(np.percentile(image, 99.9))
        min_peak_offset = _adaptive_min_peak_offset(background, p999)
    min_peak = background + min_peak_offset

    px, py = model.sky_to_pixel(cat_az, cat_alt)
    r = search_radius
    n_matches = 0
    offsets = []

    for i in range(len(cat_az)):
        xi, yi = int(round(px[i])), int(round(py[i]))
        if xi - r < 0 or xi + r >= nx or yi - r < 0 or yi + r >= ny:
            continue
        box = image[yi - r:yi + r + 1, xi - r:xi + r + 1]
        peak = float(np.max(box))
        if peak < min_peak:
            continue

        # Find the peak pixel position within the search box
        max_pos = np.unravel_index(np.argmax(box), box.shape)
        py_peak = yi - r + max_pos[0]
        px_peak = xi - r + max_pos[1]

        if require_compact:
            # Check compactness: inner 7x7 mean vs outer 15x15 mean
            ir = 3  # inner radius
            outr = 7  # outer radius
            if (px_peak - outr < 0 or px_peak + outr >= nx or
                    py_peak - outr < 0 or py_peak + outr >= ny):
                continue
            inner = image[py_peak - ir:py_peak + ir + 1,
                          px_peak - ir:px_peak + ir + 1]
            outer = image[py_peak - outr:py_peak + outr + 1,
                          px_peak - outr:px_peak + outr + 1]
            inner_mean = float(np.mean(inner))
            outer_mean = float(np.mean(outer))
            if outer_mean <= 0:
                continue
            compact = inner_mean / outer_mean
            # Point sources have compact > 1.5; dome emission is ~1.0-1.3
            if compact < 1.5:
                continue

        n_matches += 1
        offset = np.sqrt((px_peak - px[i]) ** 2 + (py_peak - py[i]) ** 2)
        offsets.append(offset)

    median_offset = float(np.median(offsets)) if offsets else float(search_radius)
    return n_matches, median_offset


def guided_match_grid_search(image, cat_az, cat_alt, cx, cy,
                             f_center, f_range=0.3, rho_hint=None,
                             refine_cat_az=None, refine_cat_alt=None,
                             horizon_r=None):
    """Grid search over (f, rho) using compact point-source match count.

    Uses compactness-filtered guided matching (inner/outer brightness
    ratio > 1.5) to count only genuine point sources, rejecting dome
    emission and extended features.

    After the grid search, the top candidates (within 1 match of best)
    are each tested with a quick guided_refine to pick the one that
    produces the most total matches.  This is necessary because the
    compact count is noisy with few bright stars.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az, cat_alt : ndarray
        Bright catalog star positions (radians) for grid scoring.
    cx, cy : float
        Optical center estimate.
    f_center : float
        Central focal length estimate (e.g., from horizon circle).
    f_range : float
        Fractional range to search around f_center (default 0.3 = ±30%).
    rho_hint : float or None
        If given (radians), restricts rho search to ±30° around hint.
    refine_cat_az, refine_cat_alt : ndarray or None
        Catalog stars (radians) for the refinement validation step.
        Should include more stars (e.g., vmag < 4.0). Falls back to
        cat_az/cat_alt if None.
    horizon_r : float or None
        Observed horizon radius for constraining f during refinement.

    Returns
    -------
    CameraModel
        Best-scoring model.
    """
    if refine_cat_az is None:
        refine_cat_az = cat_az
    if refine_cat_alt is None:
        refine_cat_alt = cat_alt

    # Coarse scan: fix f near horizon-derived center, search all rho.
    # Use only a few f values near f_center (±10%) to avoid false minima
    # at wrong f values where coincidental matches occur.
    f_values = np.arange(max(200, f_center * 0.9),
                         f_center * 1.1 + 1, 15)
    if rho_hint is not None:
        rho_center = np.degrees(rho_hint)
        rho_values = np.radians(np.arange(rho_center - 30,
                                          rho_center + 31, 2) % 360)
    else:
        rho_values = np.radians(np.arange(0, 360, 2))

    # Collect all (n_matches, med_offset, f, rho) results
    all_results = []
    for f in f_values:
        for rho in rho_values:
            model = CameraModel(
                cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
                rho=rho, f=f, proj_type=ProjectionType.EQUIDISTANT,
            )
            n, med_off = _guided_match_count(
                image, model, cat_az, cat_alt,
                search_radius=15, require_compact=True,
            )
            all_results.append((n, med_off, float(f), float(rho)))

    # Find coarse best
    all_results.sort(key=lambda x: (-x[0], x[1]))
    coarse_best_n = all_results[0][0]
    coarse_best_f = all_results[0][2]
    coarse_best_rho = all_results[0][3]

    # Fine scan around coarse optimum
    best_rho_deg = np.degrees(coarse_best_rho)
    f_fine = np.arange(max(200, coarse_best_f - 50), coarse_best_f + 51, 5)
    rho_fine = np.radians(np.arange(best_rho_deg - 10,
                                    best_rho_deg + 11, 1) % 360)

    for f in f_fine:
        for rho in rho_fine:
            model = CameraModel(
                cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
                rho=rho, f=f, proj_type=ProjectionType.EQUIDISTANT,
            )
            n, med_off = _guided_match_count(
                image, model, cat_az, cat_alt,
                search_radius=15, require_compact=True,
            )
            all_results.append((n, med_off, float(f), float(rho)))

    # Sort: most matches first, then lowest offset
    all_results.sort(key=lambda x: (-x[0], x[1]))
    best_grid_n = all_results[0][0]

    # Collect top candidates: within 1 match of best, distinct rho (5° bins)
    seen_rho_bins = set()
    candidates = []
    for n, med_off, f, rho in all_results:
        if n < best_grid_n - 1:
            break
        rho_bin = int(np.degrees(rho) / 5)
        if rho_bin in seen_rho_bins:
            continue
        seen_rho_bins.add(rho_bin)
        candidates.append((n, med_off, f, rho))
        if len(candidates) >= 6:
            break

    log.info(f"  Grid top: {best_grid_n} compact matches, "
             f"{len(candidates)} distinct candidates to refine")

    # Quick guided_refine on each candidate to find the true best
    best_model = None
    best_refine_n = 0
    best_refine_rms = 999.0

    for n, med_off, f, rho in candidates:
        seed = CameraModel(
            cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
            rho=rho, f=f, proj_type=ProjectionType.EQUIDISTANT,
        )
        try:
            m, nm, rms = guided_refine(
                image, refine_cat_az, refine_cat_alt, seed,
                n_iterations=8,
                alt_min_deg=15, alt_max_deg=75,
                fix_az0=True,
                horizon_r=horizon_r, horizon_weight=1.0,
            )
        except Exception:
            continue
        log.info(f"    Candidate rho={np.degrees(rho):.0f}° f={f:.0f}: "
                 f"-> {nm} matches, RMS={rms:.1f}, "
                 f"f={m.f:.0f}, rho={np.degrees(m.rho):.1f}")
        if _is_better(nm, rms, best_refine_n, best_refine_rms):
            best_model = m
            best_refine_n = nm
            best_refine_rms = rms

    if best_model is None:
        # Fallback: use grid best without refinement
        best_model = CameraModel(
            cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
            rho=all_results[0][3], f=all_results[0][2],
            proj_type=ProjectionType.EQUIDISTANT,
        )
        best_refine_n = all_results[0][0]

    log.info(f"  Guided-match grid: f={best_model.f:.0f}, "
             f"rho={np.degrees(best_model.rho):.1f}, "
             f"{best_refine_n} refined matches")

    return best_model


def pixel_brightness_grid_search(image, cat_az, cat_alt, cx, cy,
                                 initial_f=750.0, rho_hint=None):
    """Brute-force grid search using point-source scoring.

    Uses point-source contrast scoring (peak vs annulus) to avoid being
    fooled by extended emission from the dome or lens reflections.

    Parameters
    ----------
    image : ndarray
        Raw image.
    cat_az, cat_alt : ndarray
        Bright catalog star positions (radians).
    cx, cy : float
        Image center.
    initial_f : float
        Focal length estimate.
    rho_hint : float or None
        If given (radians), restricts rho search to ±30° around hint.

    Returns
    -------
    CameraModel
        Best-scoring model.
    """
    ny, nx = image.shape
    background = float(np.median(image))
    image_sub = image.astype(np.float64) - background

    f_lo = max(200, initial_f * 0.5)
    f_hi = initial_f * 1.5

    best_score = -1.0
    best_f = initial_f
    best_rho = 0.0

    # Coarse scan
    f_values = np.arange(f_lo, f_hi + 1, 20)

    if rho_hint is not None:
        rho_center = np.degrees(rho_hint)
        rho_values_deg = np.arange(rho_center - 30, rho_center + 31, 3)
        rho_values = np.radians(rho_values_deg % 360)
    else:
        rho_values = np.radians(np.arange(0, 360, 3))

    for f in f_values:
        for rho in rho_values:
            model = CameraModel(
                cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
                rho=rho, f=f, proj_type=ProjectionType.EQUIDISTANT,
            )
            px, py = model.sky_to_pixel(cat_az, cat_alt)
            score = _point_source_score(image_sub, px, py, nx, ny)
            if score > best_score:
                best_score = score
                best_f = float(f)
                best_rho = float(rho)

    # Fine scan around coarse optimum
    best_rho_deg = np.degrees(best_rho)
    f_fine = np.arange(max(200, best_f - 40), best_f + 41, 3)
    rho_fine_deg = np.arange(best_rho_deg - 8, best_rho_deg + 9, 1)
    rho_fine = np.radians(rho_fine_deg % 360)

    for f in f_fine:
        for rho in rho_fine:
            model = CameraModel(
                cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
                rho=rho, f=f, proj_type=ProjectionType.EQUIDISTANT,
            )
            px, py = model.sky_to_pixel(cat_az, cat_alt)
            score = _point_source_score(image_sub, px, py, nx, ny)
            if score > best_score:
                best_score = score
                best_f = float(f)
                best_rho = float(rho)

    return CameraModel(
        cx=cx, cy=cy, az0=0.0, alt0=np.pi / 2,
        rho=best_rho, f=best_f, proj_type=ProjectionType.EQUIDISTANT,
    )


def instrument_fit_pipeline(image, det_table, cat_table, initial_f=750.0,
                            verbose=False, meta=None, progress=None):
    """Full instrument characterization pipeline.

    Pipeline strategy:
    1. Detect horizon circle → optical center (cx, cy) and horizon radius
    2. Compute initial f from horizon radius (equidistant: f = R / (π/2))
    3. Guided-match grid search over (f, rho) using match count scoring
    4. Guided iterative refinement (geometry, mid-altitude, az0 fixed)
    5. Broaden altitude range
    6. Joint geometry + distortion refinement
    7. Full-field final refinement

    Parameters
    ----------
    image : ndarray
        Raw image data (ny, nx).
    det_table : Table
        Detected sources (columns: x, y, flux).
    cat_table : Table
        Visible catalog stars (columns: az_deg, alt_deg, vmag_expected).
    initial_f : float
        Initial focal length guess in pixels (used as fallback).
    verbose : bool
        Print progress messages.

    Returns
    -------
    best_model : CameraModel
        Best camera model.
    n_matched : int
        Number of guided matches.
    rms : float
        RMS residual (pixels).
    diagnostics : dict
        Pipeline diagnostics for logging/display.
    """
    ny, nx = image.shape
    cx0 = nx / 2.0
    cy0 = ny / 2.0
    diag = {}

    # Adaptive min_peak_offset: noise-aware threshold.
    background = float(np.median(image))
    p999 = float(np.percentile(image, 99.9))
    min_peak_offset = _adaptive_min_peak_offset(background, p999)
    if verbose:
        log.info(f"  Adaptive min_peak_offset: {min_peak_offset:.0f} "
                 f"(bg={background:.0f}, p99.9={p999:.0f})")

    cat_az_deg = np.asarray(cat_table["az_deg"], dtype=np.float64)
    cat_alt_deg = np.asarray(cat_table["alt_deg"], dtype=np.float64)
    vmag_ext = np.asarray(cat_table["vmag_expected"], dtype=np.float64)

    cat_az = np.radians(cat_az_deg)
    cat_alt = np.radians(cat_alt_deg)

    # Bright subset for grid search scoring — exclude near-zenith and
    # near-horizon stars that are unreliable for fitting
    bright_mask = vmag_ext < 3.0
    if np.sum(bright_mask) < 10:
        bright_mask = vmag_ext < 4.5
    alt_filt = (cat_alt_deg >= 10) & (cat_alt_deg <= 85)
    bright_fit_mask = bright_mask & alt_filt
    if np.sum(bright_fit_mask) < 5:
        bright_fit_mask = bright_mask
    bright_az = cat_az[bright_fit_mask]
    bright_alt = cat_alt[bright_fit_mask]

    # Medium-brightness subset for guided matching refinement
    med_mask = vmag_ext < 4.0
    if np.sum(med_mask) < 20:
        med_mask = vmag_ext < 5.0
    med_az = cat_az[med_mask]
    med_alt = cat_alt[med_mask]

    # Wide subset for distortion fitting (need low-altitude stars)
    wide_mask = vmag_ext < 5.0
    wide_az = cat_az[wide_mask]
    wide_alt = cat_alt[wide_mask]

    # Full catalog for final matching (maximize match count)
    full_az = cat_az
    full_alt = cat_alt

    # --- Step 1: Detect horizon circle ---
    if verbose:
        log.info("Step 1: Detect horizon circle (sky/dome boundary)")
    hc_cx, hc_cy, hc_R, hc_n = detect_horizon_circle(image, cx0, cy0)
    diag["horizon_cx"] = hc_cx
    diag["horizon_cy"] = hc_cy
    diag["horizon_R"] = hc_R
    diag["horizon_n_points"] = hc_n

    # Use horizon center as optical center estimate
    if hc_n >= 20:
        cx_est = hc_cx
        cy_est = hc_cy
        f_from_horizon = hc_R / (np.pi / 2)
    else:
        cx_est = cx0
        cy_est = cy0
        f_from_horizon = initial_f
    # Horizon radius for constraining f (and breaking f/k1 degeneracy)
    hr = hc_R if hc_n >= 20 else None

    # Alt0 range: always use (87, 93) which covers cameras tilted
    # up to 3° from zenith.  This is wide enough for Liverpool
    # (alt0=88.5°) without introducing false hypotheses at extreme
    # tilt angles that confuse Haleakala/APICAM.
    alt0_range_deg = (87, 93)

    if verbose:
        log.info(f"  Horizon circle: center=({hc_cx:.0f}, {hc_cy:.0f}), "
                 f"R={hc_R:.0f}, n={hc_n}")
        log.info(f"  Implied f (equidistant): {f_from_horizon:.0f}")
    if progress:
        progress("horizon", cx=hc_cx, cy=hc_cy, radius=hc_R,
                 n_points=hc_n, f_implied=f_from_horizon)

    # --- Step 1b: Density-based rotation estimate ---
    # The Milky Way creates a strong azimuthal density signal in both
    # the image detections and the star catalog.  Cross-correlating these
    # profiles gives a robust initial rho estimate that prevents the
    # pattern matcher from locking onto the wrong rotation.
    rho_density = None
    if hc_n >= 20:
        det_x_arr = np.asarray(det_table["x"], dtype=np.float64)
        det_y_arr = np.asarray(det_table["y"], dtype=np.float64)
        det_r_arr = np.sqrt((det_x_arr - cx_est) ** 2 +
                            (det_y_arr - cy_est) ** 2)
        # Mid-radius detections only (avoid center artifacts and horizon)
        mid = (det_r_arr > hc_R * 0.15) & (det_r_arr < hc_R * 0.85)
        if np.sum(mid) >= 50:
            det_pa = np.arctan2(det_x_arr[mid] - cx_est,
                                det_y_arr[mid] - cy_est)
            n_bins = 72  # 5° bins
            det_hist, _ = np.histogram(det_pa, bins=n_bins,
                                        range=(-np.pi, np.pi))
            # Catalog azimuthal profile (mid-altitude, bright)
            cat_mid = (cat_alt_deg >= 15) & (cat_alt_deg <= 75) & \
                      (vmag_ext < 6.5)
            if np.sum(cat_mid) >= 50:
                cat_pa0 = cat_az[cat_mid]  # PA at rho=0
                cat_hist0, _ = np.histogram(cat_pa0, bins=n_bins,
                                             range=(-np.pi, np.pi))
                # Cross-correlate
                dn = det_hist.astype(float) - np.mean(det_hist)
                cn = cat_hist0.astype(float) - np.mean(cat_hist0)
                corrs = np.array([np.dot(dn, np.roll(cn, s))
                                  for s in range(n_bins)])
                best_shift = int(np.argmax(corrs))
                rho_density = best_shift * (2 * np.pi / n_bins)
                if rho_density > np.pi:
                    rho_density -= 2 * np.pi
                if verbose:
                    log.info(f"Step 1b: Density rotation estimate: "
                             f"rho={np.degrees(rho_density):.0f}°")
                if progress:
                    progress("rotation",
                             rho_deg=float(np.degrees(rho_density)))

    # --- Step 2: Blind pattern-matching solve ---
    # This replaces the old compact-arc + guided-refine approach.
    # Searches over (f, rho, az0, alt0) simultaneously without assuming
    # az0=0, which was the root cause of previous pipeline failures.
    if verbose:
        log.info("Step 2: Blind pattern-matching solve")

    # Build detection arrays, filtered to point sources whose peak is
    # significantly above local background.  This removes noise
    # detections and dome/cloud artifacts that would mislead the
    # pattern match with false correspondences.
    pm_det_x_raw = np.asarray(det_table["x"], dtype=np.float64)
    pm_det_y_raw = np.asarray(det_table["y"], dtype=np.float64)
    pm_peak_raw = np.asarray(det_table["peak"], dtype=np.float64)

    bright_mask = np.zeros(len(pm_det_x_raw), dtype=bool)
    edge_r = 7
    for j in range(len(pm_det_x_raw)):
        xi = int(round(pm_det_x_raw[j]))
        yi = int(round(pm_det_y_raw[j]))
        if (xi - edge_r < 0 or xi + edge_r >= nx or
                yi - edge_r < 0 or yi + edge_r >= ny):
            continue
        peak_val = float(image[yi, xi])
        edge_box = image[yi - edge_r:yi + edge_r + 1,
                         xi - edge_r:xi + edge_r + 1]
        edge_mask_arr = np.ones(edge_box.shape, dtype=bool)
        edge_mask_arr[2:-2, 2:-2] = False
        local_bg_val = float(np.median(edge_box[edge_mask_arr]))
        if peak_val - local_bg_val >= min_peak_offset:
            bright_mask[j] = True

    bright_idx = np.where(bright_mask)[0]
    bright_order = np.argsort(-pm_peak_raw[bright_idx])
    bright_idx = bright_idx[bright_order]

    pm_det_x = pm_det_x_raw[bright_idx]
    pm_det_y = pm_det_y_raw[bright_idx]
    if verbose:
        log.info(f"  Point-source detections: {len(pm_det_x)} / "
                 f"{len(pm_det_x_raw)}")
    if progress:
        progress("pattern_match_start", n_detections=len(pm_det_x))

    # Moon exclusion: if moon is above horizon, find the brightest
    # extended region (heavy Gaussian blur) and exclude detections near it.
    if meta is not None and len(pm_det_x) >= 5:
        try:
            from astropy.coordinates import get_body, EarthLocation, AltAz
            import astropy.units as u
            _loc = EarthLocation(
                lat=meta["lat_deg"] * u.deg,
                lon=meta["lon_deg"] * u.deg)
            _frame = AltAz(obstime=meta["obs_time"], location=_loc)
            _moon = get_body("moon", meta["obs_time"],
                             _loc).transform_to(_frame)
            if float(_moon.alt.deg) > 5:
                from scipy.ndimage import gaussian_filter as _gf_moon
                _smoothed = _gf_moon(image.astype(np.float64), sigma=20)
                myi, mxi = np.unravel_index(
                    np.argmax(_smoothed), _smoothed.shape)
                moon_x, moon_y = float(mxi), float(myi)
                excl_r = max(100, 0.15 * np.sqrt(nx**2 + ny**2))
                dist_sq = ((pm_det_x - moon_x)**2 +
                           (pm_det_y - moon_y)**2)
                keep = dist_sq > excl_r**2
                n_excl = int(np.sum(~keep))
                if n_excl > 0:
                    pm_det_x = pm_det_x[keep]
                    pm_det_y = pm_det_y[keep]
                    if verbose:
                        log.info(f"  Moon (alt={float(_moon.alt.deg):.0f}°) "
                                 f"— excluded {n_excl} detections near "
                                 f"({moon_x:.0f}, {moon_y:.0f})")
                    if progress:
                        progress("moon_excluded",
                                 alt_deg=float(_moon.alt.deg),
                                 n_excluded=n_excl,
                                 x=moon_x, y=moon_y)
        except Exception:
            pass

    # Try multiple f_range windows centered on the horizon-implied f.
    # For an all-sky camera, the horizon constrains f tightly:
    # f ≈ R_horizon / (π/2).  We allow ±50% to accommodate distortion
    # and projection-type uncertainty, but don't search wildly different
    # focal lengths that produce physically impossible solutions.
    f_lo = max(250, f_from_horizon * 0.5)
    f_hi = max(f_from_horizon * 1.5, 1500)
    f_mid = f_from_horizon
    f_ranges = [
        (f_lo, f_mid),
        (f_mid * 0.75, f_hi),
    ]
    # Ensure at least one range covers the horizon estimate
    f_ranges.append((max(250, f_from_horizon * 0.6),
                     f_from_horizon * 1.5))

    pm_model = None
    pm_n = 0
    pm_rms = 999.0
    pm_diag = {}
    pm_candidates = []

    # Scale anchor count with image size — bigger sensors benefit from
    # more hypotheses, and in dense fields the top-25 detections are
    # more likely to be real stars (not dome/obstruction artifacts).
    n_anchor = max(25, min(50, len(pm_det_x) // 20))

    # Test both normal and mirrored (E-W flipped) orientations.
    # Some cameras produce mirrored images depending on their optical
    # train.  We run pattern_match_solve for both and keep the best.
    is_mirrored = False
    pm_det_x_mirror = (nx - 1) - pm_det_x  # flip x about image center

    for label, det_x_use in [("normal", pm_det_x),
                              ("mirror", pm_det_x_mirror)]:
        for fr in f_ranges:
            m, n, r, d = pattern_match_solve(
                det_x_use, pm_det_y, cat_az, cat_alt, vmag_ext,
                cx=cx_est, cy=cy_est, nx=nx, ny=ny,
                f_range=fr, n_bright_det=n_anchor, verbose=False,
                alt0_range=alt0_range_deg,
            )
            if m is not None and n >= 10:
                # Tag mirrored candidates so we know to flip later
                d["_mirrored"] = (label == "mirror")
                pm_candidates.append((m, n, r, d, fr))
                if verbose:
                    log.info(f"  {label} f_range {fr[0]:.0f}-{fr[1]:.0f}: "
                             f"{n} matches, RMS={r:.1f}, f={m.f:.0f}")
                if progress:
                    progress("pattern_match_candidate",
                             orientation=label, n_matches=n,
                             rms=r, f=m.f)

    # Also run with density-derived rho hint (both orientations).
    if rho_density is not None:
        for label, det_x_use in [("normal", pm_det_x),
                                  ("mirror", pm_det_x_mirror)]:
            for fr in f_ranges:
                m, n, r, dd = pattern_match_solve(
                    det_x_use, pm_det_y, cat_az, cat_alt, vmag_ext,
                    cx=cx_est, cy=cy_est, nx=nx, ny=ny,
                    f_range=fr, n_bright_det=n_anchor, verbose=False,
                    rho_hint=rho_density,
                    alt0_range=alt0_range_deg,
                )
                if m is not None and n >= 10:
                    dd["_mirrored"] = (label == "mirror")
                    pm_candidates.append((m, n, r, dd, fr))

    if pm_candidates:
        # Filter by horizon consistency: for an all-sky camera, f must
        # produce a horizon radius close to what we observed.  Reject
        # solutions where f is far from the horizon-implied value.
        n_pre = len(pm_candidates)
        if hr is not None and f_from_horizon > 100:
            consistent = [c for c in pm_candidates
                          if abs(c[0].f - f_from_horizon) / f_from_horizon < 0.35]
            if consistent:
                pm_candidates = consistent
                if verbose:
                    log.info("  Horizon filter: kept %d/%d candidates "
                             "(f within 35%% of %.0f)",
                             len(consistent), n_pre, f_from_horizon)

        # Pick candidate with best score.  Weight match count heavily —
        # for an unrefined blind-solve model, RMS is always high; the
        # match count is the reliable signal that the solution is correct.
        pm_candidates.sort(key=lambda c: -c[1] * c[1] / (1.0 + c[2] * c[2]))
        pm_model, pm_n, pm_rms, pm_diag, _ = pm_candidates[0]

        # If best candidate came from mirrored pass, flip the image
        # and detection table so all subsequent processing uses the
        # correct orientation.
        if pm_diag.get("_mirrored", False) and pm_n >= 30:
            is_mirrored = True
            image = image[:, ::-1]  # flip x-axis
            # Update detection table x-coordinates
            det_x_arr = np.asarray(det_table["x"], dtype=np.float64)
            det_table["x"] = (nx - 1) - det_x_arr
            if verbose:
                log.info("  Image is MIRRORED (E-W flipped) — "
                         "flipping for processing")

    diag["pattern_match"] = pm_diag
    diag["mirrored"] = is_mirrored

    if verbose:
        log.info(f"  Best after Step 2: {pm_n} matches, RMS={pm_rms:.1f}"
                 f"{' (mirrored)' if is_mirrored else ''}")
    if progress:
        progress("pattern_match_done", n_matches=pm_n, rms=pm_rms,
                 f=pm_model.f if pm_model else 0, mirrored=is_mirrored)

    if pm_model is not None and pm_n >= 10:
        best_model = pm_model
        best_n = pm_n
        best_rms = pm_rms
    else:
        # Fallback to old compact-arc approach
        if verbose:
            log.info("  Pattern match failed, falling back to arc profiling")

        arc_rho, arc_corr = compact_arc_rho_search(
            image, wide_az, wide_alt, cx_est, cy_est, f_from_horizon,
            r_max=hc_R * 0.87 if hc_n >= 20 else 1100,
        )
        diag["arc_rho_deg"] = float(np.degrees(arc_rho))
        diag["arc_corr_peak"] = float(np.max(arc_corr))

        grid_model = CameraModel(
            cx=cx_est, cy=cy_est, az0=0.0, alt0=np.pi / 2,
            rho=arc_rho, f=f_from_horizon,
            proj_type=ProjectionType.EQUIDISTANT,
        )

        m3, n3, rms3 = guided_refine(
            image, med_az, med_alt, grid_model,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            alt_min_deg=15, alt_max_deg=75,
            fix_az0=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        m4, n4, rms4 = guided_refine(
            image, med_az, med_alt, m3,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            alt_min_deg=5, alt_max_deg=88,
            fix_az0=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        if _is_better(n4, rms4, n3, rms3):
            best_model = m4
            best_n = n4
            best_rms = rms4
        else:
            best_model = m3
            best_n = n3
            best_rms = rms3

    # If pattern match found a non-trivial az0, don't fix it in later steps.
    # If we fell back to the old arc approach, keep az0 fixed as before.
    if pm_model is not None and pm_n >= 10:
        near_zenith = False  # az0 is meaningful, don't fix it
    else:
        near_zenith = True   # fallback: fix az0=0

    if verbose:
        log.info(f"  Best after Step 2: {best_n} matches, RMS={best_rms:.1f}")

    # --- Step 5: Estimate f and k1 jointly from horizon + star matches ---
    # The horizon circle constrains the effective projection at θ=π/2:
    #   f*π/2 * (1 + k1*(f*π/2)²) = R_horizon
    # The guided matches from step 3/4 provide actual star positions.
    # From matched star pairs at different altitudes, solve for f and k1
    # analytically using two constraints.
    if verbose:
        log.info("Step 5: Joint f and k1 estimation")
    from .projection import _theta_to_r, _apply_distortion

    if hc_n >= 20:
        # Get matched pairs from the geometry model to find mid-altitude scale
        step5_matches = _guided_match(
            image, best_model, med_az, med_alt,
            search_radius=12,
            min_peak=background + min_peak_offset,
            background=background,
            alt_min_rad=np.radians(30),
            alt_max_rad=np.radians(70),
        )
        if verbose:
            log.info(f"  Mid-altitude matches for scale: {len(step5_matches)}")

        if len(step5_matches) >= 5:
            # Compute median pixel radius of matched stars at known altitudes
            px_m, py_m = best_model.sky_to_pixel(med_az, med_alt)
            r_matched = []
            theta_matched = []
            for cat_idx, det_x, det_y, _ in step5_matches:
                r_det = np.sqrt((det_x - best_model.cx) ** 2 +
                                (det_y - best_model.cy) ** 2)
                theta_star = np.pi / 2 - med_alt[cat_idx]  # zenith angle
                r_matched.append(r_det)
                theta_matched.append(theta_star)

            r_matched = np.array(r_matched)
            theta_matched = np.array(theta_matched)

            # Use median mid-altitude point as constraint
            med_r = float(np.median(r_matched))
            med_theta = float(np.median(theta_matched))
            R_horiz = hc_R
            theta_horiz = np.pi / 2

            # Solve two equations for f and k1:
            # f*θ1*(1 + k1*(f*θ1)²) = r1  [mid-altitude]
            # f*θ2*(1 + k1*(f*θ2)²) = r2  [horizon]
            # Let a = f*θ1, b = f*θ2
            # a + k1*a³ = r1  →  k1 = (r1 - a)/a³
            # b + k1*b³ = r2  →  k1 = (r2 - b)/b³
            # (r1 - f*θ1)/(f*θ1)³ = (r2 - f*θ2)/(f*θ2)³
            # Solve numerically for f
            from scipy.optimize import brentq
            try:
                def f_eq(f_try):
                    a = f_try * med_theta
                    b = f_try * theta_horiz
                    if a <= 0 or b <= 0:
                        return 1e10
                    lhs = (med_r - a) / (a ** 3)
                    rhs = (R_horiz - b) / (b ** 3)
                    return lhs - rhs

                f_lo = max(100, best_model.f * 0.5)
                f_hi = max(3000, best_model.f * 3)
                f_solved = brentq(f_eq, f_lo, f_hi)
                a = f_solved * med_theta
                k1_solved = (med_r - a) / (a ** 3)

                # Verify: check that k1 is physically reasonable
                r_h_check = f_solved * theta_horiz * (
                    1 + k1_solved * (f_solved * theta_horiz) ** 2)
                if verbose:
                    log.info(f"  Two-constraint solution: "
                             f"f={f_solved:.0f}, k1={k1_solved:.4e}")
                    log.info(f"  Mid-alt: θ={np.degrees(med_theta):.1f}° "
                             f"r_meas={med_r:.0f}px r_model="
                             f"{f_solved*med_theta*(1+k1_solved*(f_solved*med_theta)**2):.0f}px")
                    log.info(f"  Horizon: r_model={r_h_check:.0f}px "
                             f"(observed {R_horiz:.0f}px)")

                # Clamp k1: allow slight pincushion (up to 10% of
                # k1_max) to accommodate non-equidistant lenses.
                k1_limit = 0.1 * 0.3 / (f_solved * f_solved)
                if k1_solved > k1_limit:
                    k1_solved = 0.0
                    f_solved = best_model.f

                dist_seed = CameraModel(
                    cx=best_model.cx, cy=best_model.cy,
                    az0=best_model.az0, alt0=best_model.alt0,
                    rho=best_model.rho, f=f_solved,
                    proj_type=best_model.proj_type,
                    k1=k1_solved, k2=0.0,
                )
            except (ValueError, ZeroDivisionError):
                # Fallback: simple horizon-based seed
                r_horiz_model = _theta_to_r(np.pi / 2, best_model.f,
                                            best_model.proj_type)
                k1_seed = ((hc_R / r_horiz_model - 1.0) /
                           (r_horiz_model ** 2))
                # Clamp k1: allow slight pincushion
                k1_limit = 0.1 * 0.3 / (best_model.f ** 2)
                if k1_seed > k1_limit:
                    k1_seed = k1_limit
                dist_seed = CameraModel(
                    cx=best_model.cx, cy=best_model.cy,
                    az0=best_model.az0, alt0=best_model.alt0,
                    rho=best_model.rho, f=best_model.f,
                    proj_type=best_model.proj_type,
                    k1=k1_seed, k2=0.0,
                )
                if verbose:
                    log.info(f"  Fallback: k1={k1_seed:.4e}")
        else:
            dist_seed = best_model
            if verbose:
                log.info("  Too few mid-altitude matches, skipping k1 seed")
    else:
        dist_seed = best_model
        if verbose:
            log.info("  No horizon circle, skipping k1 seed")

    # --- Step 5b: Projection type search ---
    # If the two-constraint solver needed positive k1 (clamped),
    # equidistant may be the wrong base projection.  Try stereographic
    # and equisolid, which naturally map the horizon farther out.
    # The best alternative becomes the dist_seed for Step 6, where full
    # distortion fitting will determine the final quality.
    alt_proj_seed = None
    if (hc_n >= 20 and dist_seed.k1 == 0.0 and
            dist_seed.proj_type == ProjectionType.EQUIDISTANT):
        r_h_equi = _theta_to_r(np.pi / 2, dist_seed.f,
                                ProjectionType.EQUIDISTANT)
        if abs(r_h_equi - hc_R) / hc_R > 0.25:
            if verbose:
                log.info("Step 5b: Projection type search "
                         f"(equidistant horizon mismatch: "
                         f"{r_h_equi:.0f} vs {hc_R:.0f}px)")
            best_alt_n = 0
            best_alt_rms = 999.0
            for alt_proj in [ProjectionType.STEREOGRAPHIC,
                             ProjectionType.EQUISOLID]:
                f_alt = hc_R / _theta_to_r(np.pi / 2, 1.0, alt_proj)
                alt_model = CameraModel(
                    cx=best_model.cx, cy=best_model.cy,
                    az0=best_model.az0, alt0=best_model.alt0,
                    rho=best_model.rho, f=f_alt,
                    proj_type=alt_proj,
                )
                # Wide initial match to bridge the projection difference
                # (positions can shift 40-80px from equidistant → alt).
                wide_matches = _guided_match(
                    image, alt_model, med_az, med_alt,
                    search_radius=80,
                    min_peak=background + min_peak_offset,
                    background=background,
                    alt_min_rad=np.radians(20),
                    alt_max_rad=np.radians(80),
                )
                if len(wide_matches) >= 10:
                    det_x_w = np.array([m[1] for m in wide_matches])
                    det_y_w = np.array([m[2] for m in wide_matches])
                    cat_az_w = np.array([med_az[m[0]]
                                         for m in wide_matches])
                    cat_alt_w = np.array([med_alt[m[0]]
                                          for m in wide_matches])
                    try:
                        alt_model = _fit_model_to_pairs(
                            det_x_w, det_y_w, cat_az_w, cat_alt_w,
                            alt_model, fix_az0=True,
                            horizon_r=hr, horizon_weight=1.0,
                        )
                    except Exception:
                        pass
                m_alt, n_alt, rms_alt = guided_refine(
                    image, med_az, med_alt, alt_model,
                    n_iterations=15,
                    min_peak_offset=min_peak_offset,
                    alt_min_deg=10, alt_max_deg=80,
                    fix_az0=True,
                    horizon_r=hr, horizon_weight=1.0,
                )
                if verbose:
                    log.info(f"  {alt_proj.value}: f={f_alt:.0f} → "
                             f"{n_alt} matches, RMS={rms_alt:.1f}")
                if n_alt >= 50 and _is_better(n_alt, rms_alt,
                                               best_alt_n, best_alt_rms):
                    best_alt_n = n_alt
                    best_alt_rms = rms_alt
                    alt_proj_seed = CameraModel(
                        cx=m_alt.cx, cy=m_alt.cy,
                        az0=m_alt.az0, alt0=m_alt.alt0,
                        rho=m_alt.rho, f=m_alt.f,
                        proj_type=alt_proj,
                        k1=m_alt.k1, k2=m_alt.k2,
                    )
            if alt_proj_seed is not None:
                if verbose:
                    log.info(f"  Alt projection seed: "
                             f"{alt_proj_seed.proj_type.value} "
                             f"({best_alt_n} matches, "
                             f"RMS={best_alt_rms:.1f})")

    # --- Step 6: Joint geometry + distortion refinement ---
    # Use guided matching (finds many matches) with f_scale=50 in the
    # optimizer so the horizon constraint actually works (was being
    # downweighted by soft_l1 loss with f_scale=5).
    if verbose:
        log.info("Step 6: Joint geometry + distortion refinement")
    if progress:
        progress("refine_start")

    # Phase A: Joint fit from analytical seed, mid-altitude
    m6a, n6a, rms6a = guided_refine(
        image, wide_az, wide_alt, dist_seed,
        n_iterations=15,
        min_peak_offset=min_peak_offset,
        alt_min_deg=15, alt_max_deg=75,
        fix_az0=near_zenith,
        fit_distortion=True,
        horizon_r=hr, horizon_weight=2.0,
    )
    if verbose:
        r_h = _apply_distortion(
            _theta_to_r(np.pi / 2, m6a.f, m6a.proj_type), m6a.k1, m6a.k2)
        log.info(f"  Phase A (mid-alt): {n6a} matches, RMS={rms6a:.1f}, "
                 f"f={m6a.f:.0f}, k1={m6a.k1:.2e}, "
                 f"horizon={r_h:.0f}px")
    if progress:
        progress("refine_phase", phase="A", n_matches=n6a,
                 rms=rms6a, f=m6a.f)

    # Phase B: Broaden to include low-altitude stars
    m6b, n6b, rms6b = guided_refine(
        image, wide_az, wide_alt, m6a,
        n_iterations=15,
        min_peak_offset=min_peak_offset,
        alt_min_deg=5, alt_max_deg=88,
        fix_az0=near_zenith,
        fit_distortion=True,
        horizon_r=hr, horizon_weight=2.0,
    )
    if verbose:
        r_h = _apply_distortion(
            _theta_to_r(np.pi / 2, m6b.f, m6b.proj_type), m6b.k1, m6b.k2)
        log.info(f"  Phase B (wide): {n6b} matches, RMS={rms6b:.1f}, "
                 f"f={m6b.f:.0f}, k1={m6b.k1:.2e}, "
                 f"horizon={r_h:.0f}px")
    if progress:
        progress("refine_phase", phase="B", n_matches=n6b,
                 rms=rms6b, f=m6b.f)

    # Phase C: Full-field final refinement
    m6c, n6c, rms6c = guided_refine(
        image, wide_az, wide_alt, m6b,
        n_iterations=15,
        min_peak_offset=min_peak_offset,
        alt_min_deg=3, alt_max_deg=88,
        fix_az0=near_zenith,
        fit_distortion=True,
        horizon_r=hr, horizon_weight=1.0,
    )
    if verbose:
        r_h = _apply_distortion(
            _theta_to_r(np.pi / 2, m6c.f, m6c.proj_type), m6c.k1, m6c.k2)
        log.info(f"  Phase C (full-field): {n6c} matches, RMS={rms6c:.1f}, "
                 f"f={m6c.f:.0f}, k1={m6c.k1:.2e}, "
                 f"horizon={r_h:.0f}px")
    if progress:
        progress("refine_phase", phase="C", n_matches=n6c,
                 rms=rms6c, f=m6c.f)

    # Pick the best of Phases A/B/C
    for m_cand, n_cand, rms_cand in [(m6a, n6a, rms6a), (m6b, n6b, rms6b),
                                      (m6c, n6c, rms6c)]:
        if _is_better(n_cand, rms_cand, best_n, best_rms):
            best_model = m_cand
            best_n = n_cand
            best_rms = rms_cand

    # Phase D: Detection-based matching with full catalog.
    # Uses model-independent DAOStarFinder positions to avoid the
    # self-reinforcing error problem with guided matching on huge catalogs.
    m6d, n6d, rms6d = detection_refine(
        det_table, full_az, full_alt, best_model,
        n_iterations=20,
        alt_min_deg=3, alt_max_deg=88,
        fix_az0=near_zenith,
        fit_distortion=True,
        horizon_r=hr, horizon_weight=1.0,
    )
    if verbose:
        r_h = _apply_distortion(
            _theta_to_r(np.pi / 2, m6d.f, m6d.proj_type), m6d.k1, m6d.k2)
        log.info(f"  Phase D (full catalog det): {n6d} matches, RMS={rms6d:.1f}, "
                 f"f={m6d.f:.0f}, k1={m6d.k1:.2e}, "
                 f"horizon={r_h:.0f}px")
    if progress:
        progress("refine_phase", phase="D", n_matches=n6d,
                 rms=rms6d, f=m6d.f)

    if _is_better(n6d, rms6d, best_n, best_rms):
        best_model = m6d
        best_n = n6d
        best_rms = rms6d

    # Phase E: Deep guided refinement.
    # Extend slightly beyond the vmag<5 used in A-C to pick up a few more
    # real stars, but not so deep that noise matches degrade the fit.
    deep_mask = vmag_ext < 5.5
    deep_az = cat_az[deep_mask]
    deep_alt = cat_alt[deep_mask]
    # Use a tighter search radius for Phase E since the model is already
    # well-refined.  In dense star fields, a large radius grabs wrong stars.
    phase_e_sr = max(12, min(30, int(best_rms * 2.5)))
    m6e, n6e, rms6e = guided_refine(
        image, deep_az, deep_alt, best_model,
        n_iterations=15,
        min_peak_offset=min_peak_offset,
        alt_min_deg=5, alt_max_deg=88,
        fix_az0=False,
        fit_distortion=True,
        horizon_r=hr, horizon_weight=1.0,
        initial_search_radius=phase_e_sr,
    )
    if verbose:
        r_h = _apply_distortion(
            _theta_to_r(np.pi / 2, m6e.f, m6e.proj_type), m6e.k1, m6e.k2)
        log.info(f"  Phase E (full guided): {n6e} matches, RMS={rms6e:.1f}, "
                 f"f={m6e.f:.0f}, k1={m6e.k1:.2e}, "
                 f"horizon={r_h:.0f}px")
    if progress:
        progress("refine_phase", phase="E", n_matches=n6e,
                 rms=rms6e, f=m6e.f)

    # Only adopt Phase E if RMS didn't degrade significantly
    if n6e > best_n and rms6e <= best_rms * 1.15:
        best_model = m6e
        best_n = n6e
        best_rms = rms6e

    if progress:
        progress("refine_done", n_matches=best_n, rms=best_rms,
                 f=best_model.f)

    # Phase F: Alternative projection (if Step 5b found one).
    # First stabilize geometry (no distortion), then add distortion.
    if alt_proj_seed is not None:
        if verbose:
            log.info(f"  Phase F (alt projection: "
                     f"{alt_proj_seed.proj_type.value})")
        # F.1: Geometry-only refinement to stabilize the model
        mf_g, nf_g, rmsf_g = guided_refine(
            image, med_az, med_alt, alt_proj_seed,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            alt_min_deg=15, alt_max_deg=75,
            fix_az0=True,
            fit_distortion=False,
            horizon_r=hr, horizon_weight=1.0,
        )
        # F.2: Add distortion fitting from the stabilized model
        mf_a, nf_a, rmsf_a = guided_refine(
            image, wide_az, wide_alt, mf_g,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            alt_min_deg=10, alt_max_deg=80,
            fix_az0=near_zenith,
            fit_distortion=True,
            horizon_r=hr, horizon_weight=2.0,
        )
        # F.3: Widen altitude range
        mf_b, nf_b, rmsf_b = guided_refine(
            image, wide_az, wide_alt, mf_a,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            alt_min_deg=5, alt_max_deg=88,
            fix_az0=near_zenith,
            fit_distortion=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        # F.4: Detection-based refinement
        mf_d, nf_d, rmsf_d = detection_refine(
            det_table, full_az, full_alt, mf_b,
            n_iterations=20,
            alt_min_deg=3, alt_max_deg=88,
            fix_az0=near_zenith,
            fit_distortion=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        # Pick best from alt-projection phases
        for m_cand, n_cand, rms_cand in [(mf_a, nf_a, rmsf_a),
                                          (mf_b, nf_b, rmsf_b),
                                          (mf_d, nf_d, rmsf_d)]:
            if _is_better(n_cand, rms_cand, best_n, best_rms):
                best_model = m_cand
                best_n = n_cand
                best_rms = rms_cand
        if verbose:
            log.info(f"    geom: {nf_g} matches, RMS={rmsf_g:.1f}")
            log.info(f"    +dist: {nf_a} matches, RMS={rmsf_a:.1f}, "
                     f"f={mf_a.f:.0f}")
            log.info(f"    wide: {nf_b} matches, RMS={rmsf_b:.1f}, "
                     f"f={mf_b.f:.0f}")
            log.info(f"    det:  {nf_d} matches, RMS={rmsf_d:.1f}, "
                     f"f={mf_d.f:.0f}")

    # --- Step 6c: Score-map sweep backup ---
    # When the pattern-match pipeline couldn't solve the camera (weak
    # result), try an independent path: score-map parameter sweep →
    # altitude-descent f/k1 → full refinement.  This catches cameras
    # like Liverpool where DAOStarFinder detections are too sparse for
    # the pattern match but the raw image has detectable stars.
    if hc_n >= 20:
        if verbose:
            log.info("Step 6c: Score-map sweep backup")

        from scipy.ndimage import gaussian_filter as _gf
        from scipy.ndimage import maximum_filter as _xf
        _img = image.astype(np.float64)
        _bg_map = _gf(_img, sigma=25.0)
        _contrast = _img - _bg_map
        _star_mask = (_contrast > min_peak_offset * 0.5).astype(np.float64)
        _star_dilated = _xf(_star_mask, size=11)
        wide_score_map = _gf(_star_dilated, sigma=6.0)

        # Moon exclusion: if moon is up, mask the smoothed image
        # around the brightest extended region before sweeping.
        moon_excl_x, moon_excl_y = None, None
        if meta is not None:
            try:
                from astropy.coordinates import get_body, EarthLocation, AltAz
                import astropy.units as u
                _loc = EarthLocation(
                    lat=meta["lat_deg"] * u.deg,
                    lon=meta["lon_deg"] * u.deg)
                _frame = AltAz(obstime=meta["obs_time"], location=_loc)
                _moon = get_body("moon", meta["obs_time"],
                                 _loc).transform_to(_frame)
                if float(_moon.alt.deg) > 5:
                    _smoothed = _gf(_img, sigma=20)
                    myi, mxi = np.unravel_index(
                        np.argmax(_smoothed), _smoothed.shape)
                    moon_excl_x, moon_excl_y = float(mxi), float(myi)
                    excl_r = max(100, 0.15 * np.sqrt(nx**2 + ny**2))
                    # Zero out score map near moon
                    _ys, _xs = np.mgrid[:ny, :nx]
                    moon_mask = (((_xs - moon_excl_x)**2 +
                                  (_ys - moon_excl_y)**2) < excl_r**2)
                    wide_score_map[moon_mask] = 0
                    if verbose:
                        log.info(f"  Moon exclusion at "
                                 f"({moon_excl_x:.0f}, {moon_excl_y:.0f})")
            except Exception:
                pass

        f_vals = f_from_horizon * np.array([0.85, 0.92, 1.0, 1.08, 1.15])
        alt0_vals = np.radians(np.array([87, 88.5, 90, 91.5, 93]))
        rho_vals = np.radians(np.arange(0, 360, 2))

        sweep_best_score = -1.0
        sweep_best_rho = 0.0
        sweep_best_f = f_from_horizon
        sweep_best_alt0 = np.pi / 2

        n_total_sweep = len(f_vals) * len(alt0_vals) * len(rho_vals)
        if progress:
            progress("sweep_start", n_models=n_total_sweep)
        sweep_count = 0
        for fi in f_vals:
            for a0 in alt0_vals:
                for rho in rho_vals:
                    m = CameraModel(
                        cx=cx_est, cy=cy_est, az0=0.0, alt0=a0,
                        rho=rho, f=fi,
                        proj_type=ProjectionType.EQUIDISTANT,
                    )
                    s = score_model(wide_score_map, cat_az, cat_alt,
                                    vmag_ext, m)
                    if s > sweep_best_score:
                        sweep_best_score = s
                        sweep_best_rho = rho
                        sweep_best_f = fi
                        sweep_best_alt0 = a0
                    sweep_count += 1
                    if progress and sweep_count % 200 == 0:
                        progress("sweep_progress",
                                 fraction=sweep_count / n_total_sweep)

        if verbose:
            log.info(f"  Sweep: rho={np.degrees(sweep_best_rho):.0f}°, "
                     f"f={sweep_best_f:.0f}, "
                     f"alt0={np.degrees(sweep_best_alt0):.1f}°")

        # Altitude-descent refinement from sweep
        sweep_seed = CameraModel(
            cx=cx_est, cy=cy_est, az0=0.0, alt0=sweep_best_alt0,
            rho=sweep_best_rho, f=sweep_best_f,
            proj_type=ProjectionType.EQUIDISTANT,
        )

        # Phase 1: high-altitude geometry (distortion negligible).
        # Wide search radius (50px) because the sweep's rho may be
        # 10-20° off, causing ~30-40px positional errors even at high
        # altitude.  The first iteration needs to catch these.
        sw1, n_sw1, rms_sw1 = guided_refine(
            image, wide_az, wide_alt, sweep_seed,
            n_iterations=12, min_peak_offset=min_peak_offset,
            alt_min_deg=60, alt_max_deg=85,
            fix_az0=True, fit_distortion=False,
            horizon_r=None, horizon_weight=0.0,
            initial_search_radius=50,
        )
        if progress:
            progress("sweep_refine", phase=1, n_matches=n_sw1,
                     rms=rms_sw1, f=sw1.f)
        if n_sw1 >= 5:
            sweep_seed = sw1

        # Phase 2: mid-altitude with distortion
        sw2, n_sw2, rms_sw2 = guided_refine(
            image, wide_az, wide_alt, sweep_seed,
            n_iterations=15, min_peak_offset=min_peak_offset,
            alt_min_deg=15, alt_max_deg=75,
            fix_az0=True, fit_distortion=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        if progress:
            progress("sweep_refine", phase=2, n_matches=n_sw2,
                     rms=rms_sw2, f=sw2.f)
        if n_sw2 >= 10:
            sweep_seed = sw2

        # Phase 3: wide altitude
        sw3, n_sw3, rms_sw3 = guided_refine(
            image, wide_az, wide_alt, sweep_seed,
            n_iterations=15, min_peak_offset=min_peak_offset,
            alt_min_deg=5, alt_max_deg=88,
            fix_az0=True, fit_distortion=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        if progress:
            progress("sweep_refine", phase=3, n_matches=n_sw3,
                     rms=rms_sw3, f=sw3.f)
        if _is_better(n_sw3, rms_sw3, n_sw2, rms_sw2):
            sweep_seed = sw3

        # Phase 4: deep guided
        deep_mask = vmag_ext < 5.5
        sw4, n_sw4, rms_sw4 = guided_refine(
            image, cat_az[deep_mask], cat_alt[deep_mask], sweep_seed,
            n_iterations=15, min_peak_offset=min_peak_offset,
            alt_min_deg=5, alt_max_deg=88,
            fix_az0=False, fit_distortion=True,
            horizon_r=hr, horizon_weight=1.0,
        )
        n_sweep_final = n_sw4 if n_sw4 > max(n_sw2, n_sw3) else max(n_sw2, n_sw3)
        rms_sweep_final = rms_sw4 if n_sw4 > max(n_sw2, n_sw3) else min(rms_sw2, rms_sw3)
        model_sweep_final = sw4 if n_sw4 > max(n_sw2, n_sw3) else sweep_seed

        if progress:
            progress("sweep_refine", phase=4, n_matches=n_sw4,
                     rms=rms_sw4, f=sw4.f)
            progress("sweep_result", n_matches=n_sweep_final,
                     rms=rms_sweep_final, f=model_sweep_final.f)
        if verbose:
            log.info(f"  Sweep backup: {n_sweep_final} matches, "
                     f"RMS={rms_sweep_final:.2f}, "
                     f"f={model_sweep_final.f:.0f}")

        # Compare main pipeline and sweep using bright-star validation,
        # NOT match count.  The main pipeline can produce false-positive
        # matches (dome/cloud artifacts) that inflate counts.  Validation
        # checks whether bright catalog stars have real point sources at
        # the modeled positions — this catches wrong solutions that
        # "look good" by match count.
        def _validate(model_v, label_v):
            """Fraction of bright in-frame catalog stars with point sources."""
            bright_v = (vmag_ext < 4.0) & \
                (cat_alt > np.radians(15)) & (cat_alt < np.radians(80))
            v_az = cat_az[bright_v]
            v_alt = cat_alt[bright_v]
            v_matches = _guided_match(
                image, model_v, v_az, v_alt,
                search_radius=8,
                min_peak=background + min_peak_offset,
                background=background,
            )
            vx, vy = model_v.sky_to_pixel(v_az, v_alt)
            in_frame = ((vx > 20) & (vx < nx - 20) &
                        (vy > 20) & (vy < ny - 20))
            n_if = int(np.sum(in_frame))
            frac = len(v_matches) / n_if if n_if > 0 else 0.0
            if verbose:
                log.info(f"  Validation {label_v}: "
                         f"{len(v_matches)}/{n_if} = {frac:.0%}")
            return frac

        main_val = _validate(best_model, "main")
        main_n_if = int(np.sum(
            ((lambda vx, vy: (vx > 20) & (vx < nx-20) & (vy > 20) & (vy < ny-20))
             (*best_model.sky_to_pixel(
                 cat_az[(vmag_ext < 4.0) & (cat_alt > np.radians(15)) & (cat_alt < np.radians(80))],
                 cat_alt[(vmag_ext < 4.0) & (cat_alt > np.radians(15)) & (cat_alt < np.radians(80))])))))
        sweep_val = _validate(model_sweep_final, "sweep")
        sweep_n_if = int(np.sum(
            ((lambda vx, vy: (vx > 20) & (vx < nx-20) & (vy > 20) & (vy < ny-20))
             (*model_sweep_final.sky_to_pixel(
                 cat_az[(vmag_ext < 4.0) & (cat_alt > np.radians(15)) & (cat_alt < np.radians(80))],
                 cat_alt[(vmag_ext < 4.0) & (cat_alt > np.radians(15)) & (cat_alt < np.radians(80))])))))

        winner = ""
        if sweep_val > main_val and n_sweep_final >= 30:
            winner = "sweep"
            if verbose:
                log.info(f"  Sweep backup wins by validation "
                         f"({sweep_val:.0%} vs {main_val:.0%})")
            best_model = model_sweep_final
            best_n = n_sweep_final
            best_rms = rms_sweep_final
        else:
            winner = "main"

        if progress:
            progress("validation",
                     main_frac=main_val,
                     main_n=int(main_val * main_n_if),
                     main_total=main_n_if,
                     sweep_frac=sweep_val,
                     sweep_n=int(sweep_val * sweep_n_if),
                     sweep_total=sweep_n_if,
                     winner=winner)

    # --- Step 7: Residual diagnostics ---
    if verbose:
        log.info("Step 7: Residual diagnostics")
    if best_n >= 6:
        matches = _guided_match(
            image, best_model, full_az, full_alt,
            search_radius=10,
            min_peak=background + min_peak_offset,
            background=background,
        )
        if len(matches) >= 6:
            dx = np.array([m[1] for m in matches])
            dy = np.array([m[2] for m in matches])
            az_m = np.array([full_az[m[0]] for m in matches])
            alt_m = np.array([full_alt[m[0]] for m in matches])
            px, py = best_model.sky_to_pixel(az_m, alt_m)
            rd = diagnose_residuals(dx, dy, px, py,
                                    best_model.cx, best_model.cy)
            diag["residual_diagnostics"] = rd
            if verbose:
                log.info(f"  Pattern: {rd['pattern']}")
                log.info(f"  Mean offset: ({rd['mean_dx']:.2f}, "
                         f"{rd['mean_dy']:.2f})")

    diag["final_n_matched"] = best_n
    diag["final_rms"] = best_rms
    diag["final_f"] = best_model.f
    diag["final_rho_deg"] = float(np.degrees(best_model.rho))
    diag["final_projection"] = best_model.proj_type.value
    diag["final_k1"] = best_model.k1
    diag["final_k2"] = best_model.k2

    if verbose:
        log.info(f"Final: {best_n} matches, RMS={best_rms:.2f}, "
                 f"f={best_model.f:.1f}, "
                 f"rho={np.degrees(best_model.rho):.1f}°, "
                 f"k1={best_model.k1:.2e}, "
                 f"proj={best_model.proj_type.value}")

    return best_model, best_n, best_rms, diag
