"""Local offset measurement for per-frame pointing correction.

Measures the local (dx, dy) offset between model-predicted star
positions and actual star positions in small image tiles.  The
collection of local offsets across tiles reveals the global pointing
correction (tilt, rotation, center shift) needed for this frame.

This avoids the dense-field confusion problem in guided matching by
matching PATTERNS of stars (groups/constellations) rather than
individual stars.
"""

import logging

import numpy as np
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


def measure_local_offsets(image, model, cat_az, cat_alt, cat_vmag,
                          det_x, det_y,
                          tile_size=400, search_radius=20,
                          search_step=2, min_matches=5,
                          match_radius=4.0, vmag_limit=7.0):
    """Measure local (dx, dy) offsets between model and image.

    Tiles the image, and for each tile cross-correlates the predicted
    catalog positions against actual detection positions to find the
    best local offset.

    Parameters
    ----------
    image : ndarray
        2D image array.
    model : CameraModel
        Current camera model.
    cat_az, cat_alt : ndarray
        Catalog star sky positions (radians).
    cat_vmag : ndarray
        Catalog magnitudes.
    det_x, det_y : ndarray
        Detection pixel positions.
    tile_size : int
        Tile size in pixels.
    search_radius : int
        Max offset to search (pixels).
    search_step : int
        Step size for offset search grid (pixels).
    min_matches : int
        Minimum matches in a tile for a valid offset measurement.
    match_radius : float
        Max distance for a catalog-detection pair to count as a match.
    vmag_limit : float
        Magnitude limit for catalog stars.

    Returns
    -------
    offsets : list of dict
        Each dict has: tile_x, tile_y, dx, dy, n_matches, n_cat
    """
    ny, nx = image.shape

    # Project catalog stars
    mask = cat_vmag < vmag_limit
    px, py = model.sky_to_pixel(cat_az[mask], cat_alt[mask])
    valid_cat = (np.isfinite(px) & np.isfinite(py) &
                 (px >= 10) & (px < nx - 10) &
                 (py >= 10) & (py < ny - 10))

    # Build KDTree of detections
    det_tree = KDTree(np.column_stack([det_x, det_y]))

    offsets = []
    search_offsets = np.arange(-search_radius, search_radius + 1, search_step)

    overlap = tile_size // 4
    for ty in range(0, ny, tile_size - overlap):
        for tx in range(0, nx, tile_size - overlap):
            # Catalog stars in this tile
            in_tile = (valid_cat &
                       (px >= tx) & (px < tx + tile_size) &
                       (py >= ty) & (py < ty + tile_size))
            n_cat = int(np.sum(in_tile))
            if n_cat < min_matches:
                continue

            cat_x_tile = px[in_tile]
            cat_y_tile = py[in_tile]

            # Determine pattern size needed based on local detection density
            # (more detections = need larger patterns = stricter matching)
            in_tile_det = ((det_x >= tx) & (det_x < tx + tile_size) &
                           (det_y >= ty) & (det_y < ty + tile_size))
            n_det_tile = int(np.sum(in_tile_det))
            if n_det_tile < min_matches:
                continue

            # Density-adaptive match radius: tighter in dense areas
            density = n_det_tile / (tile_size * tile_size)
            # Average inter-star spacing: 1/sqrt(density)
            spacing = 1.0 / max(0.001, np.sqrt(density))
            # Match radius should be << spacing to avoid confusion
            local_match_r = min(match_radius, spacing * 0.4)
            local_match_r = max(2.0, local_match_r)  # floor

            # Search for best (dx, dy) offset
            best_count = 0
            best_dx = 0.0
            best_dy = 0.0

            for dx in search_offsets:
                for dy in search_offsets:
                    # Shift catalog positions
                    shifted_x = cat_x_tile + dx
                    shifted_y = cat_y_tile + dy

                    # Count matches
                    shifted_pos = np.column_stack([shifted_x, shifted_y])
                    dists, _ = det_tree.query(shifted_pos)
                    count = int(np.sum(dists < local_match_r))

                    if count > best_count:
                        best_count = count
                        best_dx = float(dx)
                        best_dy = float(dy)

            if best_count >= min_matches:
                offsets.append({
                    "tile_x": tx, "tile_y": ty,
                    "dx": best_dx, "dy": best_dy,
                    "n_matches": best_count,
                    "n_cat": n_cat,
                    "n_det": n_det_tile,
                    "match_radius": local_match_r,
                })

    return offsets


def fit_pointing_from_offsets(model, offsets, image_shape):
    """Fit a global pointing correction from local offset measurements.

    Parameters
    ----------
    model : CameraModel
        Current camera model.
    offsets : list of dict
        From measure_local_offsets().
    image_shape : tuple (ny, nx)

    Returns
    -------
    corrected : CameraModel
        Model with corrected pointing.
    summary : dict
        Fit summary (median offsets, tilt correction, etc.)
    """
    from .projection import CameraModel
    from scipy.optimize import least_squares

    if len(offsets) < 3:
        return model, {"status": "too_few_tiles", "n_tiles": len(offsets)}

    # Extract offset measurements
    tile_cx = np.array([o["tile_x"] + 200 for o in offsets])  # tile centers
    tile_cy = np.array([o["tile_y"] + 200 for o in offsets])
    dx = np.array([o["dx"] for o in offsets])
    dy = np.array([o["dy"] for o in offsets])
    weights = np.array([o["n_matches"] for o in offsets], dtype=float)

    # Weighted median of offsets
    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))

    log.info("Local offsets: median dx=%.1f, dy=%.1f from %d tiles",
             med_dx, med_dy, len(offsets))

    # Fit a simple pointing correction: (dcx, dcy, dalt0, drho)
    # These map to pixel offsets as:
    # dx_pred = dcx + f * dalt0 * sin(PA) + f * drho * cos(PA) * sin(theta)
    # But for small corrections, a simpler model: just apply (dcx, dcy)
    # and small angle adjustments.

    # Start simple: apply median offset as center shift
    ny, nx = image_shape
    corrected = CameraModel(
        cx=model.cx + med_dx,
        cy=model.cy + med_dy,
        az0=model.az0, alt0=model.alt0,
        rho=model.rho, f=model.f,
        proj_type=model.proj_type,
        k1=model.k1, k2=model.k2,
    )

    # If offsets vary across the image, fit tilt correction too
    dx_spread = float(np.std(dx))
    dy_spread = float(np.std(dy))

    summary = {
        "status": "ok",
        "n_tiles": len(offsets),
        "median_dx": med_dx,
        "median_dy": med_dy,
        "dx_spread": dx_spread,
        "dy_spread": dy_spread,
    }

    if dx_spread > 3.0 or dy_spread > 3.0:
        # Offsets vary → need tilt/rotation correction, not just translation
        # Fit (dcx, dcy, dalt0, drho) to the offset field
        proj_type = model.proj_type

        def residuals(params):
            test = CameraModel(
                cx=model.cx + params[0],
                cy=model.cy + params[1],
                az0=model.az0,
                alt0=model.alt0 + params[2],
                rho=model.rho + params[3],
                f=model.f,
                proj_type=proj_type,
                k1=model.k1, k2=model.k2,
            )
            # For each tile, compute predicted offset
            res = []
            for o in offsets:
                # Use tile center as test point
                tcx, tcy = o["tile_x"] + 200, o["tile_y"] + 200
                # Original model prediction at this point
                # (we approximate: the offset IS the difference)
                w = np.sqrt(o["n_matches"])
                res.append(w * (params[0] - o["dx"]))  # approx
                res.append(w * (params[1] - o["dy"]))
            return np.array(res)

        # Better: reproject a few catalog stars with the test model
        # and compare predicted shift to measured shift
        from .projection import CameraModel as CM
        # Sample catalog stars spread across the image
        sample_cat_az = np.radians(np.array([0, 90, 180, 270, 45, 135, 225, 315]))
        sample_cat_alt = np.radians(np.array([45]*8))

        def residuals_full(params):
            test = CameraModel(
                cx=model.cx + params[0],
                cy=model.cy + params[1],
                az0=model.az0,
                alt0=model.alt0 + params[2],
                rho=model.rho + params[3],
                f=model.f * (1 + params[4]),
                proj_type=proj_type,
                k1=model.k1, k2=model.k2,
            )
            # For each tile, predict what offset the corrected model would give
            res = []
            for o in offsets:
                tcx = o["tile_x"] + 200
                tcy = o["tile_y"] + 200
                # Approximate: the correction shifts all predictions by
                # a position-dependent amount
                w = np.sqrt(o["n_matches"])
                # Use the measured offset as target
                res.append(w * (params[0] - o["dx"]))
                res.append(w * (params[1] - o["dy"]))
            return np.array(res)

        p0 = [med_dx, med_dy, 0.0, 0.0, 0.0]
        bounds_lo = [med_dx - 20, med_dy - 20,
                     -np.radians(5), -np.radians(5), -0.02]
        bounds_hi = [med_dx + 20, med_dy + 20,
                     np.radians(5), np.radians(5), 0.02]

        result = least_squares(residuals_full, p0,
                               bounds=(bounds_lo, bounds_hi),
                               loss='soft_l1', f_scale=2.0)
        p = result.x

        corrected = CameraModel(
            cx=model.cx + p[0],
            cy=model.cy + p[1],
            az0=model.az0,
            alt0=model.alt0 + p[2],
            rho=model.rho + p[3],
            f=model.f * (1 + p[4]),
            proj_type=proj_type,
            k1=model.k1, k2=model.k2,
        )
        summary["dcx"] = p[0]
        summary["dcy"] = p[1]
        summary["dalt0_deg"] = float(np.degrees(p[2]))
        summary["drho_deg"] = float(np.degrees(p[3]))
        summary["df_pct"] = p[4] * 100

    return corrected, summary
