#!/usr/bin/env python3
"""Generate publication-quality benchmark figures for the AllClear paper.

Produces a 2x2 panel figure for each camera:
  (a) Raw frame (zscale stretch)
  (b) Solved overlay (alt-az grid + matched stars)
  (c) Transmission map overlay
  (d) Residual quiver plot on image

All panels use fixed physical font sizes regardless of sensor resolution,
so figures for 4096x4096 APICAM look consistent alongside 1392x1040
Liverpool/Cloudynight.

Usage:
    python generate_paper_figures.py --output benchmark/results/paper_figures

    # Single camera:
    python generate_paper_figures.py --cameras apicam --output /tmp/figs
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera configurations — model path, representative frame, site coords
# ---------------------------------------------------------------------------
CAMERAS = {
    "apicam": {
        "label": "ESO APICAM",
        "subtitle": "KAF-16803 4096x4096, Canon 12mm, tracked 120s",
        "site": "Paranal, Chile",
        "model": "benchmark/solutions/apicam.json",
        "frame": "benchmark/data/eso_apicam/APICAM.2019-06-15T01:05:02.000.fits",
        "lat": -24.6276, "lon": -70.4051,
    },
    "haleakala": {
        "label": "Haleakala",
        "subtitle": "IMX178 3096x2080, 1.8mm fisheye, fixed",
        "site": "Haleakala, HI",
        "model": "benchmark/solutions/haleakala.json",
        "frame": "benchmark/data/haleakala/2023_11_19__00_01_19.fits",
        "lat": 20.7458, "lon": -156.4317,
    },
    "liverpool": {
        "label": "Liverpool SkyCam",
        "subtitle": "ICX267AL 1392x1040, 1.55mm f/2.0, fixed 30s",
        "site": "La Palma, Spain",
        "model": "benchmark/solutions/liverpool.json",
        "frame": "benchmark/data/liverpool_skycam/a_e_20240519_246_1_1_1.fits",
        "lat": 28.762, "lon": -17.879,
    },
    "cloudynight": {
        "label": "Cloudynight",
        "subtitle": "SX Oculus 1392x1040, 1.55mm f/1.2, fixed 60s",
        "site": "Flagstaff, AZ",
        "model": "benchmark/solutions/cloudynight.json",
        "frame": "benchmark/data/cloudynight/005.fits",
        "lat": 35.0974, "lon": -111.535,
    },
}

# Fixed font sizes (points) — independent of image resolution
FONT = {
    "panel_label": 12,   # (a), (b), (c), (d)
    "title": 10,         # Camera name
    "subtitle": 8,       # sensor specs
    "grid_label": 7,     # N, E, S, W + altitude numbers
    "stats": 8,          # n=XXX, RMS=X.XX
    "legend": 7,
    "colorbar": 7,
}

# Panel width in inches (double-column PASP = ~7")
PANEL_W = 3.3
DPI = 300


def stretch_image(data, method="asinh"):
    """Stretch image for visualization.

    Parameters
    ----------
    method : str
        'zscale' — traditional astronomical stretch (linear between zscale limits)
        'asinh'  — arcsinh stretch (shows faint + bright simultaneously)
    """
    from astropy.visualization import ZScaleInterval
    d = data.astype(float)
    d[~np.isfinite(d)] = np.nanmedian(d)

    if method == "zscale":
        vmin, vmax = ZScaleInterval(contrast=0.2).get_limits(d)
        stretched = (d - vmin) / max(vmax - vmin, 1)
    elif method == "asinh":
        # Asinh stretch: good dynamic range for all-sky images
        vmin, vmax = ZScaleInterval(contrast=0.15).get_limits(d)
        normed = (d - vmin) / max(vmax - vmin, 1)
        # Soften parameter controls the transition — 0.05 shows faint MW well
        a = 0.05
        stretched = np.arcsinh(normed / a) / np.arcsinh(1.0 / a)
    else:
        raise ValueError(f"Unknown stretch method: {method}")
    return np.clip(stretched, 0, 1)


def crop_to_sky(image, model, padding_frac=0.03):
    """Crop image to a square bounding the sky circle.

    Returns cropped image and (x_offset, y_offset) for coordinate transform.
    """
    from allclear.projection import _theta_to_r, _apply_distortion
    ny, nx = image.shape

    # Compute horizon radius from model
    r_horizon = _theta_to_r(np.pi / 2, model.f, model.proj_type)
    r_horizon = _apply_distortion(r_horizon, model.k1, model.k2)

    # Add padding
    pad = int(r_horizon * padding_frac)
    half = int(r_horizon) + pad

    # Center on optical center
    cx, cy = int(round(model.cx)), int(round(model.cy))

    # Compute crop bounds, clamped to image
    x0 = max(0, cx - half)
    x1 = min(nx, cx + half)
    y0 = max(0, cy - half)
    y1 = min(ny, cy + half)

    # Make it square (use the smaller extent)
    crop_w = x1 - x0
    crop_h = y1 - y0
    side = min(crop_w, crop_h)

    # Re-center the crop
    x_mid = (x0 + x1) / 2
    y_mid = (y0 + y1) / 2
    x0 = int(max(0, x_mid - side / 2))
    y0 = int(max(0, y_mid - side / 2))
    x1 = x0 + side
    y1 = y0 + side

    # Clamp again
    if x1 > nx:
        x1 = nx
        x0 = max(0, x1 - side)
    if y1 > ny:
        y1 = ny
        y0 = max(0, y1 - side)

    cropped = image[y0:y1, x0:x1]
    return cropped, x0, y0


def load_and_solve(cam_cfg, root):
    """Load frame, model, run fast_solve, compute transmission."""
    from allclear.instrument import InstrumentModel
    from allclear.cli import _load_frame
    from allclear.solver import fast_solve
    from allclear.transmission import compute_transmission

    model_path = root / cam_cfg["model"]
    frame_path = root / cam_cfg["frame"]

    if not frame_path.exists():
        log.warning(f"Frame not found: {frame_path}")
        return None

    inst = InstrumentModel.load(model_path)
    camera_model = inst.to_camera_model()

    data, meta, cat, det, _ = _load_frame(
        frame_path, cam_cfg["lat"], cam_cfg["lon"],
    )

    # Mirror image if the instrument model was fit to a mirrored image
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

    result = fast_solve(data, det, cat, camera_model)

    # Use guided det table if available (centroided positions)
    use_det = result.guided_det_table if result.guided_det_table is not None else det

    # Transmission
    ref_zp = inst.photometric_zeropoint if inst.photometric_zeropoint != 0.0 else None
    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )

    return {
        "image": data,
        "model": result.camera_model,
        "det": use_det,
        "cat": cat,
        "pairs": result.matched_pairs,
        "n_matched": result.n_matched,
        "rms": result.rms_residual,
        "trans_az": az,
        "trans_alt": alt,
        "trans_vals": trans,
        "meta": meta,
        "inst": inst,
        "obs_time": meta.get("obs_time"),
    }


def draw_grid(ax, model, nx, ny):
    """Draw alt-az grid with fixed font size."""
    alt_circles = [10, 20, 30, 45, 60, 80]
    az_lines = [0, 90, 180, 270]
    az_labels = {0: "N", 90: "E", 180: "S", 270: "W"}

    for alt_deg in alt_circles:
        alt_rad = np.radians(alt_deg)
        az_sweep = np.linspace(0, 2 * np.pi, 361)
        alt_sweep = np.full_like(az_sweep, alt_rad)
        px, py = model.sky_to_pixel(az_sweep, alt_sweep)
        valid = np.isfinite(px) & np.isfinite(py) & (px >= -50) & (px < nx + 50) & (py >= -50) & (py < ny + 50)
        if np.sum(valid) > 10:
            ax.plot(px[valid], py[valid], "w--", alpha=0.5, linewidth=0.6)

    for az_deg in az_lines:
        az_rad = np.radians(az_deg)
        alt_sweep = np.linspace(np.radians(5), np.radians(85), 100)
        az_sweep = np.full_like(alt_sweep, az_rad)
        px, py = model.sky_to_pixel(az_sweep, alt_sweep)
        valid = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        if np.sum(valid) > 5:
            ax.plot(px[valid], py[valid], "w--", alpha=0.5, linewidth=0.6)

        # Cardinal label near horizon (clipped to axes)
        lbl_alt = np.radians(12)
        lbl_x, lbl_y = model.sky_to_pixel(np.array([az_rad]), np.array([lbl_alt]))
        margin = nx * 0.03
        if (np.isfinite(lbl_x[0]) and margin <= lbl_x[0] < nx - margin
                and margin <= lbl_y[0] < ny - margin):
            label = az_labels.get(az_deg, "")
            if label:
                ax.text(lbl_x[0], lbl_y[0], label, color="white",
                        fontsize=FONT["grid_label"], ha="center", va="center",
                        fontweight="bold", clip_on=True,
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def draw_matched_stars(ax, det, cat, pairs, model, nx, ny,
                       vmag_limit=5.5):
    """Draw matched stars as green circles, unmatched as red.

    Only shows stars brighter than vmag_limit for visual clarity.
    """
    cat_az = np.radians(np.asarray(cat["az_deg"], dtype=float))
    cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=float))
    vmag = np.asarray(cat["vmag_expected"], dtype=float)
    proj_x, proj_y = model.sky_to_pixel(cat_az, cat_alt)

    # Marker size scales with magnitude but is in points (physical)
    def msize(v):
        return max(1.5, 5.0 - 0.5 * v)

    # Matched stars (only bright ones for visual clarity)
    matched_cat_idx = set(ci for _, ci in pairs)
    n_shown = 0
    for di, ci in pairs:
        if vmag[ci] > vmag_limit:
            continue
        px, py = float(proj_x[ci]), float(proj_y[ci])
        if 0 <= px < nx and 0 <= py < ny:
            ms = msize(vmag[ci])
            ax.plot(px, py, "o", markeredgecolor="#44ff44",
                    markerfacecolor="none", markersize=ms,
                    markeredgewidth=0.6, alpha=0.9)
            n_shown += 1

    # Unmatched catalog stars (bright only)
    in_frame = (np.isfinite(proj_x) & np.isfinite(proj_y)
                & (proj_x >= 0) & (proj_x < nx)
                & (proj_y >= 0) & (proj_y < ny)
                & (vmag < vmag_limit))
    for idx in np.where(in_frame)[0]:
        if idx not in matched_cat_idx:
            ms = msize(vmag[idx])
            ax.plot(proj_x[idx], proj_y[idx], "o",
                    markeredgecolor="#ff4444", markerfacecolor="none",
                    markersize=ms, markeredgewidth=0.5, alpha=0.6)


def overlay_transmission(ax, model, trans_az, trans_alt, trans_vals, nx, ny):
    """Overlay transmission colormap on frame."""
    from scipy.interpolate import RBFInterpolator

    if len(trans_az) < 10:
        return

    # Convert to Cartesian on unit sphere for RBF
    az_r = np.radians(trans_az)
    alt_r = np.radians(trans_alt)
    x_s = np.cos(alt_r) * np.sin(az_r)
    y_s = np.cos(alt_r) * np.cos(az_r)
    z_s = np.sin(alt_r)
    pts = np.column_stack([x_s, y_s, z_s])
    vals = np.clip(trans_vals, 0, 1.2)

    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=5.0)

    # Coarse pixel grid
    step = max(4, max(nx, ny) // 200)
    gx, gy = np.meshgrid(np.arange(0, nx, step), np.arange(0, ny, step))
    flat_x, flat_y = gx.ravel().astype(float), gy.ravel().astype(float)

    az_g, alt_g = model.pixel_to_sky(flat_x, flat_y)
    valid = np.isfinite(az_g) & np.isfinite(alt_g) & (alt_g > np.radians(3))

    trans_grid = np.full(flat_x.shape, np.nan)
    if np.sum(valid) > 0:
        x_g = np.cos(alt_g[valid]) * np.sin(az_g[valid])
        y_g = np.cos(alt_g[valid]) * np.cos(az_g[valid])
        z_g = np.sin(alt_g[valid])
        grid_pts = np.column_stack([x_g, y_g, z_g])
        trans_grid[valid] = np.clip(rbf(grid_pts), 0, 1.2)

    trans_2d = trans_grid.reshape(gx.shape)
    ax.imshow(trans_2d, origin="lower", cmap="RdYlGn",
              vmin=0, vmax=1.2, alpha=0.55,
              extent=[0, nx, 0, ny], aspect="auto",
              interpolation="bilinear")


def draw_planets(ax, model, obs_time, lat_deg, lon_deg, nx, ny):
    """Draw planet/Moon diamond markers with labels."""
    import warnings
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
    import astropy.units as u

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    frame = AltAz(obstime=obs_time, location=location)

    bodies = [
        ("Moon", "#ffffaa"),
        ("Jupiter", "#ffcc44"),
        ("Saturn", "#ddaa44"),
        ("Mars", "#ff6644"),
        ("Venus", "#ffffff"),
        ("Mercury", "#cccccc"),
    ]

    # DSOs
    dso_objects = [
        ("LMC", 80.894, -69.756, "#ffcc44"),
        ("SMC", 13.187, -72.828, "#ffcc44"),
    ]

    r = 10  # diamond radius in pixels (fixed for paper)

    for name, color in bodies:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                body = get_body(name.lower(), obs_time, location)
                altaz = body.transform_to(frame)
        except Exception:
            continue

        if altaz.alt.deg < 5:
            continue
        px, py = model.sky_to_pixel(np.radians(altaz.az.deg),
                                     np.radians(altaz.alt.deg))
        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue
        px, py = float(px), float(py)
        diamond_x = [px, px + r, px, px - r, px]
        diamond_y = [py + r, py, py - r, py, py + r]
        ax.plot(diamond_x, diamond_y, color=color, linewidth=1.2, alpha=0.9,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                clip_on=True)
        ax.text(px + r + 3, py, name, color=color,
                fontsize=FONT["grid_label"], va="center", ha="left",
                fontweight="bold", clip_on=True,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    for name, ra_deg, dec_deg, color in dso_objects:
        try:
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                altaz = coord.transform_to(frame)
        except Exception:
            continue
        if altaz.alt.deg < 0:
            continue
        px, py = model.sky_to_pixel(np.radians(altaz.az.deg),
                                     np.radians(altaz.alt.deg))
        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue
        px, py = float(px), float(py)
        diamond_x = [px, px + r, px, px - r, px]
        diamond_y = [py + r, py, py - r, py, py + r]
        ax.plot(diamond_x, diamond_y, color=color, linewidth=1.2, alpha=0.9,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                clip_on=True)
        ax.text(px + r + 3, py, name, color=color,
                fontsize=FONT["grid_label"], va="center", ha="left",
                fontweight="bold", clip_on=True,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])


def get_zoom_regions(model, cnx, cny, det, cat, pairs):
    """Pick two good zoom regions: one near zenith, one at mid-altitude.

    Returns list of (x0, y0, width, height, label) tuples in cropped coords.
    Zoom boxes are ~2:1 aspect (wider than tall).
    """
    # Zoom box size: ~15% of image width, 2:1 aspect
    zoom_w = int(cnx * 0.30)
    zoom_h = int(zoom_w * 0.5)

    # Region 1: near zenith — project zenith and center the box
    zx, zy = model.sky_to_pixel(np.array([0.0]), np.array([np.pi / 2]))
    if np.isfinite(zx[0]) and np.isfinite(zy[0]):
        # Offset slightly so it's not exactly centered (more interesting)
        z1_cx = int(zx[0]) + zoom_w // 6
        z1_cy = int(zy[0]) + zoom_h // 4
    else:
        z1_cx, z1_cy = int(model.cx), int(model.cy)

    z1_x0 = max(0, z1_cx - zoom_w // 2)
    z1_y0 = max(0, z1_cy - zoom_h // 2)
    z1_x0 = min(z1_x0, cnx - zoom_w)
    z1_y0 = min(z1_y0, cny - zoom_h)

    # Region 2: mid-altitude — find a region with good star density
    # Pick azimuth with most matched stars at alt 30-60°
    cat_az = np.radians(np.asarray(cat["az_deg"], dtype=float))
    cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=float))
    matched_ci = set(ci for _, ci in pairs)

    # Find densest cluster of matched stars at mid-altitude
    best_count = 0
    best_pos = (cnx // 4, cny // 4)
    for az_test in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        alt_test = np.radians(40)
        tx, ty = model.sky_to_pixel(np.array([az_test]), np.array([alt_test]))
        if not (np.isfinite(tx[0]) and np.isfinite(ty[0])):
            continue
        cx_t, cy_t = int(tx[0]), int(ty[0])
        if cx_t < zoom_w // 2 or cx_t > cnx - zoom_w // 2:
            continue
        if cy_t < zoom_h // 2 or cy_t > cny - zoom_h // 2:
            continue
        # Count matched stars in this box
        proj_x, proj_y = model.sky_to_pixel(cat_az, cat_alt)
        in_box = ((proj_x > cx_t - zoom_w // 2) & (proj_x < cx_t + zoom_w // 2)
                  & (proj_y > cy_t - zoom_h // 2) & (proj_y < cy_t + zoom_h // 2))
        in_box_matched = sum(1 for i in np.where(in_box)[0] if i in matched_ci)
        if in_box_matched > best_count:
            best_count = in_box_matched
            best_pos = (cx_t, cy_t)

    z2_cx, z2_cy = best_pos
    z2_x0 = max(0, z2_cx - zoom_w // 2)
    z2_y0 = max(0, z2_cy - zoom_h // 2)
    z2_x0 = min(z2_x0, cnx - zoom_w)
    z2_y0 = min(z2_y0, cny - zoom_h)

    return [
        (z1_x0, z1_y0, zoom_w, zoom_h, "zenith"),
        (z2_x0, z2_y0, zoom_w, zoom_h, "mid-alt"),
    ]


def draw_zoom_panel(ax, image_stretched, crop_model, det, cat, pairs,
                     zoom_regions, cnx, cny):
    """Draw two stacked zoom cutouts in a single panel.

    Each zoom shows the solved overlay with grid + star markers at a
    scale where individual stars are visible.
    """
    n_zooms = len(zoom_regions)
    # Stack vertically with a small gap
    gap_frac = 0.03

    for i, (zx0, zy0, zw, zh, label) in enumerate(zoom_regions):
        # Vertical position in axes coords
        y_bot = (n_zooms - 1 - i) * (1.0 / n_zooms + gap_frac / 2)
        y_height = 1.0 / n_zooms - gap_frac

        # Create inset axes
        inset = ax.inset_axes([0.0, y_bot, 1.0, y_height])

        # Extract zoom region from stretched image
        zoom_img = image_stretched[zy0:zy0+zh, zx0:zx0+zw]
        inset.imshow(zoom_img, cmap="gray", origin="lower", aspect="auto",
                     extent=[zx0, zx0+zw, zy0, zy0+zh])

        # Stars only — no grid in zooms (grid labels cause whitespace)
        draw_matched_stars(inset, det, cat, pairs, crop_model, cnx, cny,
                           vmag_limit=6.5)

        inset.set_xlim(zx0, zx0 + zw)
        inset.set_ylim(zy0, zy0 + zh)
        inset.set_xticks([])
        inset.set_yticks([])

        # Border color to match the box drawn on panel (a)
        color = "#00ccff" if i == 0 else "#ffcc00"
        for spine in inset.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

        # Label
        inset.text(0.02, 0.95, label, transform=inset.transAxes,
                   fontsize=FONT["stats"], color=color, va="top",
                   fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


def draw_residual_quiver(ax, det, cat, pairs, model, nx, ny,
                         max_arrows=500):
    """Residual quiver plot on dark background.

    Subsamples to max_arrows for visual clarity; sigma-clips outliers
    so a few bad matches don't dominate the color scale.
    """
    if not pairs:
        return

    det_x = np.array([float(det["x"][di]) for di, ci in pairs])
    det_y = np.array([float(det["y"][di]) for di, ci in pairs])
    cat_az = np.array([float(cat["az_deg"][ci]) for di, ci in pairs])
    cat_alt = np.array([float(cat["alt_deg"][ci]) for di, ci in pairs])

    proj_x, proj_y = model.sky_to_pixel(
        np.radians(cat_az), np.radians(cat_alt))
    dx = proj_x - det_x
    dy = proj_y - det_y
    rmag = np.sqrt(dx**2 + dy**2)

    # Sigma-clip outliers for cleaner visualization
    med = np.median(rmag)
    mad = np.median(np.abs(rmag - med))
    clip = med + 4 * mad * 1.4826
    keep = rmag < clip
    n_total = len(pairs)
    rms_all = np.sqrt(np.mean(rmag**2))

    det_x, det_y = det_x[keep], det_y[keep]
    dx, dy, rmag = dx[keep], dy[keep], rmag[keep]

    # Subsample if too many arrows
    if len(det_x) > max_arrows:
        idx = np.random.RandomState(42).choice(len(det_x), max_arrows,
                                                replace=False)
        det_x, det_y = det_x[idx], det_y[idx]
        dx, dy, rmag = dx[idx], dy[idx], rmag[idx]

    # Scale: make median residual visible as ~4% of image width
    median_r = max(np.median(rmag), 0.5)
    arrow_scale = median_r / (max(nx, ny) * 0.04)

    ax.set_facecolor("#111111")
    q = ax.quiver(det_x, det_y, dx, dy, rmag,
                  cmap="coolwarm", scale=arrow_scale, scale_units="xy",
                  angles="xy", width=0.003,
                  clim=[0, np.percentile(rmag, 95)])
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")

    cb = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.02, shrink=0.8)
    cb.set_label("Residual (px)", fontsize=FONT["colorbar"])
    cb.ax.tick_params(labelsize=FONT["colorbar"])

    ax.text(0.97, 0.97,
            f"n={n_total}\nRMS={rms_all:.2f} px\nmed={med:.2f} px",
            transform=ax.transAxes, fontsize=FONT["stats"],
            color="white", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))


def generate_camera_figure(cam_name, cam_cfg, root, output_dir):
    """Generate a 2x2 figure for one camera."""
    from allclear.projection import CameraModel

    print(f"\n{'='*60}")
    print(f"  {cam_cfg['label']} ({cam_cfg['site']})")
    print(f"{'='*60}")

    solved = load_and_solve(cam_cfg, root)
    if solved is None:
        print(f"  SKIPPED — frame not found")
        return None

    image = solved["image"]
    model = solved["model"]
    orig_ny, orig_nx = image.shape

    # Crop to square around sky circle
    cropped, x0, y0 = crop_to_sky(image, model, padding_frac=0.03)
    cny, cnx = cropped.shape
    print(f"  Cropped {orig_nx}x{orig_ny} -> {cnx}x{cny} (offset {x0},{y0})")

    # Create a shifted camera model for the cropped coordinate system
    crop_model = CameraModel(
        cx=model.cx - x0, cy=model.cy - y0,
        az0=model.az0, alt0=model.alt0, rho=model.rho, f=model.f,
        proj_type=model.proj_type, k1=model.k1, k2=model.k2,
    )

    # Shift detection coordinates
    from astropy.table import Table
    crop_det = Table(solved["det"])
    crop_det["x"] = np.asarray(crop_det["x"], dtype=float) - x0
    crop_det["y"] = np.asarray(crop_det["y"], dtype=float) - y0

    # Figure size: all panels square, 2 columns wide
    fig_w = PANEL_W * 2 + 0.6
    fig_h = PANEL_W * 2 + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    ax_raw, ax_solved, ax_trans, ax_resid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    stretched = stretch_image(cropped, method="asinh")

    # Compute zoom regions for panel (b)
    zoom_regions = get_zoom_regions(crop_model, cnx, cny,
                                    crop_det, solved["cat"], solved["pairs"])
    zoom_colors = ["#00ccff", "#ffcc00"]

    # --- Panel (a): Raw frame with zoom boxes ---
    ax_raw.imshow(stretched, cmap="gray", origin="lower", aspect="equal")
    # Draw alt-az grid (faint, for context)
    draw_grid(ax_raw, crop_model, cnx, cny)
    # Draw matched stars on the full frame too
    draw_matched_stars(ax_raw, crop_det, solved["cat"],
                       solved["pairs"], crop_model, cnx, cny)
    # Planets/Moon
    if solved["obs_time"] is not None:
        draw_planets(ax_raw, crop_model, solved["obs_time"],
                     cam_cfg["lat"], cam_cfg["lon"], cnx, cny)
    # Draw zoom region boxes
    from matplotlib.patches import Rectangle
    for (zx0, zy0, zw, zh, _), color in zip(zoom_regions, zoom_colors):
        rect = Rectangle((zx0, zy0), zw, zh, linewidth=1.5,
                          edgecolor=color, facecolor="none", linestyle="-")
        ax_raw.add_patch(rect)
    ax_raw.set_xticks([])
    ax_raw.set_yticks([])
    ax_raw.set_xlim(0, cnx)
    ax_raw.set_ylim(0, cny)
    ax_raw.text(0.03, 0.97, "(a)", transform=ax_raw.transAxes,
                fontsize=FONT["panel_label"], fontweight="bold",
                color="white", va="top",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    ax_raw.text(0.97, 0.03,
                f"n={solved['n_matched']}  RMS={solved['rms']:.2f} px",
                transform=ax_raw.transAxes, fontsize=FONT["stats"],
                color="#44ff44", ha="right", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])
    # Legend
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#44ff44",
               markerfacecolor="none", markersize=4, markeredgewidth=0.8,
               label="Matched"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#ff4444",
               markerfacecolor="none", markersize=4, markeredgewidth=0.8,
               label="Unmatched"),
    ]
    leg = ax_raw.legend(handles=legend_items, loc="lower left",
                        fontsize=FONT["legend"], framealpha=0.6,
                        facecolor="black", edgecolor="gray",
                        labelcolor="white", handletextpad=0.3)

    # --- Panel (b): Zoom cutouts ---
    draw_zoom_panel(ax_solved, stretched, crop_model, crop_det,
                    solved["cat"], solved["pairs"],
                    zoom_regions, cnx, cny)
    # Place (b) label after zoom panel is drawn, anchored to figure
    ax_solved.text(0.03, 0.97, "(b)", transform=ax_solved.transAxes,
                   fontsize=FONT["panel_label"], fontweight="bold",
                   color="white", va="top", zorder=20,
                   path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # --- Panel (c): Transmission overlay ---
    ax_trans.imshow(stretched, cmap="gray", origin="lower", aspect="equal")
    overlay_transmission(ax_trans, crop_model,
                         solved["trans_az"], solved["trans_alt"],
                         solved["trans_vals"], cnx, cny)
    draw_grid(ax_trans, crop_model, cnx, cny)
    if solved["obs_time"] is not None:
        draw_planets(ax_trans, crop_model, solved["obs_time"],
                     cam_cfg["lat"], cam_cfg["lon"], cnx, cny)
    ax_trans.set_xticks([])
    ax_trans.set_yticks([])
    ax_trans.text(0.03, 0.97, "(c)", transform=ax_trans.transAxes,
                  fontsize=FONT["panel_label"], fontweight="bold",
                  color="white", va="top",
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Transmission colorbar (small, in-axis)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdYlGn", norm=Normalize(0, 1.2))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_trans, fraction=0.046, pad=0.02, shrink=0.6,
                      location="right")
    cb.set_label("Transmission", fontsize=FONT["colorbar"])
    cb.ax.tick_params(labelsize=FONT["colorbar"])

    # --- Panel (d): Residual quiver ---
    draw_residual_quiver(ax_resid, crop_det, solved["cat"],
                         solved["pairs"], crop_model, cnx, cny)
    ax_resid.set_xticks([])
    ax_resid.set_yticks([])
    ax_resid.text(0.03, 0.97, "(d)", transform=ax_resid.transAxes,
                  fontsize=FONT["panel_label"], fontweight="bold",
                  color="white", va="top",
                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Super title
    fig.suptitle(f"{cam_cfg['label']}  \u2014  {cam_cfg['subtitle']}",
                 fontsize=FONT["title"], fontweight="bold", y=0.99)

    out_path = output_dir / f"fig_benchmark_{cam_name}.pdf"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    # Also save PNG for quick preview
    png_path = output_dir / f"fig_benchmark_{cam_name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"  -> {out_path}")
    print(f"  -> {png_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper benchmark figures for AllClear")
    parser.add_argument("--cameras", nargs="*", default=list(CAMERAS.keys()),
                        choices=list(CAMERAS.keys()),
                        help="Which cameras to process (default: all)")
    parser.add_argument("--output", default="benchmark/results/paper_figures",
                        help="Output directory for figures")
    parser.add_argument("--root", default=".",
                        help="Project root directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"AllClear paper figure generation")
    print(f"Output: {output_dir}/")

    results = {}
    for cam_name in args.cameras:
        cam_cfg = CAMERAS[cam_name]
        path = generate_camera_figure(cam_name, cam_cfg, root, output_dir)
        if path:
            results[cam_name] = path

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} figures:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
