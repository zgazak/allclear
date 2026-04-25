"""Annotated all-sky image plotting with grid, star, and transmission overlays."""

import warnings

import matplotlib

if matplotlib.get_backend().lower() in ('', 'agg'):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke


def zscale(data, contrast=0.2):
    """Apply astronomical zscale stretch for visualization.

    Handles NaN/Inf values that commonly appear in FITS data.

    Parameters
    ----------
    data : ndarray
        Input image data array.
    contrast : float
        Contrast parameter for ZScaleInterval (default 0.2).

    Returns
    -------
    ndarray
        Normalized array with values scaled to [0, 1] range.
    """
    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)

    fill = float(np.median(arr[finite]))
    if not np.all(finite):
        arr = arr.copy()
        arr[~finite] = fill

    norm = ZScaleInterval(contrast=contrast)
    out = np.asarray(norm(arr), dtype=np.float32)

    out_finite = np.isfinite(out)
    if not np.any(out_finite):
        return np.zeros_like(out)
    if not np.all(out_finite):
        out = out.copy()
        out[~out_finite] = float(np.median(out[out_finite]))

    return out


def plot_frame(image, camera_model, det_table=None, cat_table=None,
               matched_pairs=None, show_grid=True, transmission_data=None,
               obs_time=None, lat_deg=None, lon_deg=None,
               output_path=None, dpi=None, horizon_r=None,
               horizon_center=None, obscuration=None):
    """Render an annotated all-sky camera frame.

    Parameters
    ----------
    image : ndarray
        2D image array.
    camera_model : CameraModel
        Solved camera projection model.
    det_table : Table, optional
        Detected sources (columns: x, y, flux).
    cat_table : Table, optional
        Catalog stars (columns: az_deg, alt_deg, vmag_expected).
    matched_pairs : list of (det_idx, cat_idx), optional
        Matched detection-catalog pairs.  Shown as thin green circles
        at catalog projected positions, sized by magnitude.
    show_grid : bool
        Draw az/alt coordinate grid lines (default True).
    transmission_data : tuple of (az, alt, trans), optional
        Per-star transmission measurements (degrees) for colored overlay.
    obs_time : astropy.time.Time, optional
        Observation time — used to compute and annotate planet positions.
    lat_deg, lon_deg : float, optional
        Observer location — needed together with obs_time for planets.
    output_path : str or Path, optional
        If set, saves PNG and closes figure. Otherwise returns (fig, ax).
    dpi : int, optional
        Output DPI. Auto-scaled from image size if not given.

    Returns
    -------
    None or (fig, ax)
    """
    ny, nx = image.shape

    if dpi is None:
        max_dim = max(ny, nx)
        dpi = 75 if max_dim > 4000 else 150

    fig = plt.figure(
        figsize=(nx / dpi, ny / dpi),
        dpi=dpi,
        frameon=False,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)

    # Base image
    ax.imshow(zscale(image), cmap="gray", origin="lower")

    # Transmission overlay (draw under grid/stars)
    if transmission_data is not None:
        trans_az, trans_alt, trans_vals = transmission_data
        _overlay_transmission(ax, camera_model, trans_az, trans_alt,
                              trans_vals, nx, ny)

    # Obscuration overlay — mark persistently-blocked pixels with a
    # distinct hatch pattern so occluded sky is visibly different from
    # cloudy sky.
    if obscuration is not None:
        _overlay_obscuration(ax, camera_model, obscuration, nx, ny)

    # Az/Alt grid
    if show_grid:
        _draw_altaz_grid(ax, camera_model, nx, ny)

    # Horizon circle (measured sky/ground boundary)
    if horizon_r is not None:
        hcx = horizon_center[0] if horizon_center else camera_model.cx
        hcy = horizon_center[1] if horizon_center else camera_model.cy
        horizon_circle = Circle(
            (hcx, hcy), horizon_r,
            fill=False, edgecolor='#4488ff', linewidth=2.0,
            linestyle='--', alpha=0.9,
        )
        ax.add_patch(horizon_circle)

    # Stars
    if matched_pairs is not None:
        _draw_stars(ax, det_table, cat_table, matched_pairs,
                    camera_model, nx, ny)

    # Planets
    if obs_time is not None and lat_deg is not None and lon_deg is not None:
        _draw_planets(ax, camera_model, obs_time, lat_deg, lon_deg, nx, ny)

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    if matched_pairs is not None and cat_table is not None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch
        fs = _font_size(image) * 0.85
        legend_items = [
            Line2D([0], [0], marker='s', color='none', markeredgecolor='#44ff44',
                   markerfacecolor='none', markersize=fs*0.9, markeredgewidth=1.2,
                   label='Matched (detected)'),
            Line2D([0], [0], marker='o', color='none', markeredgecolor='#44ff44',
                   markerfacecolor='none', markersize=fs*0.9, markeredgewidth=1.2,
                   label='Matched (catalog)'),
            Line2D([0], [0], marker='o', color='none', markeredgecolor='#ff4444',
                   markerfacecolor='none', markersize=fs*0.9, markeredgewidth=1.2,
                   label='Catalog (predicted)'),
        ]
        leg = ax.legend(handles=legend_items, loc='lower left',
                        fontsize=fs, framealpha=0.6,
                        facecolor='black', edgecolor='gray',
                        labelcolor='white')

    # Version watermark
    shift = 0.005 * np.array(image.shape)
    try:
        from importlib.metadata import version
        ver = version("allclear")
    except Exception:
        from allclear import __version__ as ver
    ax.text(
        nx - shift[1],
        ny - shift[0],
        f"allclear v{ver}",
        color="white",
        ha="right",
        va="top",
        size=_font_size(image) * 1.2,
        alpha=0.8,
    )

    if output_path:
        from pathlib import Path
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return None
    else:
        return fig, ax


def _font_size(img):
    """Calculate appropriate font size based on image dimensions."""
    return max(6, min(img.shape[1], img.shape[0]) * 0.01)


def _draw_altaz_grid(ax, camera_model, nx, ny):
    """Draw az/alt coordinate grid lines projected through the camera model.

    Parameters
    ----------
    ax : matplotlib Axes
    camera_model : CameraModel
    nx, ny : int
        Image dimensions.
    """
    n_samples = 200
    fs = _font_size(np.empty((ny, nx)))

    alt_circles_deg = [10, 20, 30, 45, 60, 80]
    az_lines_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    cardinal_names = {0: "N", 45: "NE", 90: "E", 135: "SE",
                      180: "S", 225: "SW", 270: "W", 315: "NW"}

    text_kwargs = dict(
        color="white",
        size=fs,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                  alpha=0.7, edgecolor="none"),
    )

    # --- Altitude circles ---
    for alt_deg in alt_circles_deg:
        az_samples = np.linspace(0, 2 * np.pi, n_samples)
        alt_rad = np.full(n_samples, np.radians(alt_deg))

        x, y = camera_model.sky_to_pixel(az_samples, alt_rad)

        valid = np.isfinite(x) & np.isfinite(y)
        valid &= (x >= -nx * 0.1) & (x < nx * 1.1)
        valid &= (y >= -ny * 0.1) & (y < ny * 1.1)
        if np.sum(valid) < 2:
            continue

        # Break line at large jumps (wrapping artifacts)
        xv, yv = x[valid], y[valid]
        _plot_segments(ax, xv, yv, nx, color="white", linestyle="--",
                       alpha=0.6, linewidth=1.0)

        # Label at one in-frame point
        in_frame = (xv >= 0) & (xv < nx) & (yv >= 0) & (yv < ny)
        if np.any(in_frame):
            idx = np.where(in_frame)[0][len(in_frame[in_frame]) // 2]
            ax.text(xv[idx], yv[idx], f"{alt_deg}\u00b0",
                    ha="center", va="center", **text_kwargs)

    # --- Azimuth lines ---
    for az_deg in az_lines_deg:
        alt_samples = np.linspace(np.radians(5), np.radians(89), n_samples)
        az_rad = np.full(n_samples, np.radians(az_deg))

        x, y = camera_model.sky_to_pixel(az_rad, alt_samples)

        valid = np.isfinite(x) & np.isfinite(y)
        valid &= (x >= -nx * 0.1) & (x < nx * 1.1)
        valid &= (y >= -ny * 0.1) & (y < ny * 1.1)
        if np.sum(valid) < 2:
            continue

        xv, yv = x[valid], y[valid]
        _plot_segments(ax, xv, yv, nx, color="white", linestyle="--",
                       alpha=0.6, linewidth=1.0)

        # Cardinal direction label at alt=5° end
        lx, ly = camera_model.sky_to_pixel(np.radians(az_deg), np.radians(5))
        if (np.isfinite(lx) and np.isfinite(ly)
                and 0 <= lx < nx and 0 <= ly < ny):
            label = cardinal_names.get(az_deg, f"{az_deg}\u00b0")
            ax.text(float(lx), float(ly), label,
                    ha="center", va="center",
                    fontweight="bold", **text_kwargs)


def _plot_segments(ax, x, y, nx, **kwargs):
    """Plot a line, breaking at large pixel jumps to avoid wrap artifacts."""
    if len(x) < 2:
        return
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    breaks = np.where(dist > nx * 0.3)[0]

    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(x)]])

    for s, e in zip(starts, ends):
        if e - s >= 2:
            ax.plot(x[s:e], y[s:e], **kwargs)


def _draw_stars(ax, det_table, cat_table, matched_pairs, camera_model,
                nx, ny):
    """Draw matched star markers.

    For each matched pair:
    - Green square at the DETECTED centroid position (actual star)
    - Blue circle at the CATALOG projected position (where model predicts)
    The offset between them shows the model residual.

    Only draws markers where the detection flux indicates a real star
    (not a noise peak from a faint undetectable catalog entry).

    Parameters
    ----------
    ax : matplotlib Axes
    det_table : Table or None
        Detected sources with x, y, flux columns.
    cat_table : Table or None
        Catalog stars with az_deg, alt_deg, vmag_expected columns.
    matched_pairs : list of (det_idx, cat_idx) or None
    camera_model : CameraModel
    nx, ny : int
        Image dimensions.
    """
    from matplotlib.patches import Rectangle

    matched_cat = set()

    if (matched_pairs and det_table is not None
            and cat_table is not None and camera_model is not None):
        if "flux" in det_table.colnames and len(det_table) > 0:
            fluxes = np.array(det_table["flux"], dtype=np.float64)
            flux_thresh = float(np.median(fluxes)) * 2.0
        else:
            flux_thresh = 0.0

        for di, ci in matched_pairs:
            if "flux" in det_table.colnames and len(det_table) > di:
                flux = float(det_table["flux"][di])
                if flux < flux_thresh:
                    continue

            matched_cat.add(ci)
            vmag = float(cat_table["vmag_expected"][ci])
            radius = max(3, int(12 - 1.5 * vmag))
            lw = max(0.6, 1.5 - 0.15 * vmag)

            # Green square at detected centroid
            if len(det_table) > di:
                dx = float(det_table["x"][di])
                dy = float(det_table["y"][di])
                if 0 <= dx < nx and 0 <= dy < ny:
                    side = radius * 2
                    rect = Rectangle(
                        (dx - side / 2, dy - side / 2), side, side,
                        facecolor="none", edgecolor="#44ff44",
                        linewidth=lw, alpha=0.7,
                    )
                    ax.add_patch(rect)

            # Green circle at catalog projected position
            az_rad = np.radians(float(cat_table["az_deg"][ci]))
            alt_rad = np.radians(float(cat_table["alt_deg"][ci]))
            cx, cy = camera_model.sky_to_pixel(az_rad, alt_rad)
            if (np.isfinite(cx) and np.isfinite(cy)
                    and 0 <= cx < nx and 0 <= cy < ny):
                circle = Circle(
                    (float(cx), float(cy)), radius=radius,
                    facecolor="none", edgecolor="#44ff44",
                    linewidth=lw, alpha=0.7,
                )
                ax.add_patch(circle)

    # Unmatched catalog stars — red circles for all bright in-frame
    # catalog stars not already shown as matched
    if cat_table is not None and camera_model is not None:
        vmag_all = np.array(cat_table["vmag_expected"], dtype=np.float64)
        order = np.argsort(vmag_all)
        n_shown = 0
        for i in order:
            if n_shown >= 2000:
                break
            if i in matched_cat:
                continue
            vmag = float(vmag_all[i])
            az_rad = np.radians(float(cat_table["az_deg"][i]))
            alt_rad = np.radians(float(cat_table["alt_deg"][i]))
            cx, cy = camera_model.sky_to_pixel(az_rad, alt_rad)
            if not (np.isfinite(cx) and np.isfinite(cy)
                    and 0 <= cx < nx and 0 <= cy < ny):
                continue
            radius = max(3, int(10 - 1.2 * vmag))
            circle = Circle(
                (float(cx), float(cy)), radius=radius,
                facecolor="none", edgecolor="#ff4444",
                linewidth=1.0, alpha=0.7,
            )
            ax.add_patch(circle)
            n_shown += 1


def _overlay_transmission(ax, camera_model, trans_az, trans_alt, trans_vals,
                          nx, ny):
    """Overlay a semi-transparent transmission colormap on the image.

    Builds a pixel-space transmission image by projecting through the camera
    model and interpolating from per-star measurements via RBF.

    Parameters
    ----------
    ax : matplotlib Axes
    camera_model : CameraModel
    trans_az, trans_alt : ndarray
        Star positions (degrees).
    trans_vals : ndarray
        Measured transmission at each star.
    nx, ny : int
        Image dimensions.
    """
    from scipy.interpolate import RBFInterpolator

    valid = (np.isfinite(trans_az) & np.isfinite(trans_alt)
             & np.isfinite(trans_vals))
    if np.sum(valid) < 3:
        return

    az_v = np.radians(trans_az[valid])
    alt_v = np.radians(trans_alt[valid])
    tv = trans_vals[valid]

    # Build RBF in Cartesian on unit sphere
    pts = np.column_stack([
        np.cos(alt_v) * np.sin(az_v),
        np.cos(alt_v) * np.cos(az_v),
        np.sin(alt_v),
    ])
    rbf = RBFInterpolator(pts, tv, smoothing=5.0,
                          kernel="thin_plate_spline")

    # Coarse pixel grid (every 8 pixels)
    step = 8
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    grid_az, grid_alt = camera_model.pixel_to_sky(
        xx.astype(np.float64).ravel(),
        yy.astype(np.float64).ravel(),
    )

    below_horizon = grid_alt < 0
    invalid_sky = ~np.isfinite(grid_az) | ~np.isfinite(grid_alt)
    mask = below_horizon | invalid_sky

    # Interpolate where valid
    gx = np.cos(grid_alt) * np.sin(grid_az)
    gy = np.cos(grid_alt) * np.cos(grid_az)
    gz = np.sin(grid_alt)
    grid_pts = np.column_stack([gx, gy, gz])

    trans_img = np.full(grid_pts.shape[0], np.nan)
    ok = ~mask
    if np.sum(ok) > 0:
        trans_img[ok] = rbf(grid_pts[ok])

    trans_img = trans_img.reshape(xx.shape)
    trans_img = np.clip(trans_img, 0, 1.2)

    ax.imshow(
        trans_img, cmap="RdYlGn", alpha=0.55, origin="lower",
        extent=[0, nx, 0, ny], vmin=0, vmax=1.2,
        interpolation="bilinear",
    )


def _overlay_obscuration(ax, camera_model, obscuration, nx, ny,
                         threshold=0.3, step=8, color="#5555aa", alpha=0.55):
    """Hatch-shade pixels whose sky direction is persistently obscured.

    Projects the sky-space mask onto a coarsened pixel grid and draws
    a solid semi-transparent overlay so "occluded" is visually distinct
    from "cloudy" (green-to-red) in the transmission rendering.
    """
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    w = obscuration.project_to_pixel(
        camera_model, image_shape=(ny, nx))
    # Downsample to the coarse grid
    w_small = w[::step, ::step]
    mask = w_small < threshold

    if not mask.any():
        return

    # Build RGBA image: transparent where visible, semi-opaque color
    # where obscured
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    r, g, b = [int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    rgba[mask] = (r, g, b, alpha)

    ax.imshow(
        rgba, origin="lower",
        extent=[0, nx, 0, ny],
        interpolation="nearest",
    )


def _draw_planets(ax, camera_model, obs_time, lat_deg, lon_deg, nx, ny):
    """Annotate solar system planets visible in the frame.

    Computes alt/az for major planets and the Moon, projects through the
    camera model, and draws labeled diamond markers for those in-frame.

    Parameters
    ----------
    ax : matplotlib Axes
    camera_model : CameraModel
    obs_time : astropy.time.Time
        Observation time.
    lat_deg, lon_deg : float
        Observer latitude and longitude in degrees.
    nx, ny : int
        Image dimensions.
    """
    from astropy.coordinates import (
        EarthLocation, AltAz, SkyCoord, get_body,
    )
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

    fs = _font_size(np.empty((ny, nx)))

    for name, color in bodies:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                body = get_body(name.lower(), obs_time, location)
                altaz = body.transform_to(frame)
        except Exception:
            continue

        alt_deg_val = altaz.alt.deg
        az_deg_val = altaz.az.deg

        if alt_deg_val < 5:
            continue

        az_rad = np.radians(az_deg_val)
        alt_rad = np.radians(alt_deg_val)
        px, py = camera_model.sky_to_pixel(az_rad, alt_rad)

        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue

        px, py = float(px), float(py)

        # Diamond marker
        r = max(8, int(fs * 1.5))
        diamond_x = [px, px + r, px, px - r, px]
        diamond_y = [py + r, py, py - r, py, py + r]
        ax.plot(diamond_x, diamond_y, color=color, linewidth=1.5, alpha=0.9,
                path_effects=[withStroke(linewidth=2.5, foreground="black")])

        # Label
        ax.text(
            px + r + 2, py,
            name,
            color=color,
            size=fs * 0.9,
            va="center",
            ha="left",
            fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    # --- Magellanic Clouds (fixed equatorial coordinates) ---
    dso_objects = [
        ("LMC", 80.894, -69.756, "#ffcc44"),
        ("SMC", 13.187, -72.828, "#ffcc44"),
    ]

    for name, ra_deg, dec_deg, color in dso_objects:
        try:
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                altaz = coord.transform_to(frame)
        except Exception:
            continue

        alt_deg_val = altaz.alt.deg
        az_deg_val = altaz.az.deg

        if alt_deg_val < 0:
            continue

        az_rad = np.radians(az_deg_val)
        alt_rad = np.radians(alt_deg_val)
        px, py = camera_model.sky_to_pixel(az_rad, alt_rad)

        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue

        px, py = float(px), float(py)

        # Diamond marker (same style as planets)
        r = max(8, int(fs * 1.5))
        diamond_x = [px, px + r, px, px - r, px]
        diamond_y = [py + r, py, py - r, py, py + r]
        ax.plot(diamond_x, diamond_y, color=color, linewidth=1.5, alpha=0.9,
                path_effects=[withStroke(linewidth=2.5, foreground="black")])

        # Label
        ax.text(
            px + r + 2, py,
            name,
            color=color,
            size=fs * 0.9,
            va="center",
            ha="left",
            fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )


def plot_residuals(det_table, cat_table, matched_pairs, camera_model,
                   output_path=None):
    """Plot matched-pair residual vectors in pixel and sky coordinates.

    Produces a 2-panel figure:
    - Left: residual quiver plot over the image plane (det position,
      arrow = catalog_proj - det)
    - Right: residual magnitude vs altitude, colored by azimuth

    Parameters
    ----------
    det_table : Table
        Detected sources with columns: x, y.
    cat_table : Table
        Catalog stars with columns: az_deg, alt_deg.
    matched_pairs : list of (det_idx, cat_idx)
        Matched detection-catalog pairs.
    camera_model : CameraModel
        Solved camera model.
    output_path : str or Path, optional
        If set, saves and closes figure. Otherwise returns (fig, axes).

    Returns
    -------
    None or (fig, (ax_quiver, ax_scatter))
    """
    if not matched_pairs:
        return None

    det_x = np.array([float(det_table["x"][di]) for di, ci in matched_pairs])
    det_y = np.array([float(det_table["y"][di]) for di, ci in matched_pairs])
    cat_az = np.array([float(cat_table["az_deg"][ci]) for di, ci in matched_pairs])
    cat_alt = np.array([float(cat_table["alt_deg"][ci]) for di, ci in matched_pairs])

    # Project catalog positions through camera model
    proj_x, proj_y = camera_model.sky_to_pixel(
        np.radians(cat_az), np.radians(cat_alt)
    )

    dx = proj_x - det_x
    dy = proj_y - det_y
    residual_mag = np.sqrt(dx**2 + dy**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: quiver plot in pixel space
    # Scale arrows so median residual is ~5% of image width
    median_resid = np.median(residual_mag) if len(residual_mag) > 0 else 1.0
    img_scale = max(det_x.max() - det_x.min(), det_y.max() - det_y.min())
    if img_scale < 1:
        img_scale = 1000
    arrow_scale = median_resid / (img_scale * 0.05)
    q = ax1.quiver(det_x, det_y, dx, dy, residual_mag,
                   cmap="coolwarm", scale=arrow_scale, scale_units="xy",
                   angles="xy", width=0.004,
                   clim=[0, np.percentile(residual_mag, 95)])
    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.set_title(f"Residuals in pixel space (n={len(matched_pairs)}, "
                  f"RMS={np.sqrt(np.mean(residual_mag**2)):.2f} px)")
    plt.colorbar(q, ax=ax1, label="Residual (px)")

    # Right panel: residual magnitude vs altitude, colored by azimuth
    sc = ax2.scatter(cat_alt, residual_mag, c=cat_az, cmap="hsv",
                     vmin=0, vmax=360, s=30, alpha=0.8, edgecolors="black",
                     linewidths=0.5)
    ax2.set_xlabel("Altitude (deg)")
    ax2.set_ylabel("Residual (px)")
    ax2.set_title("Residual vs altitude")
    ax2.axhline(np.sqrt(np.mean(residual_mag**2)), color="red",
                linestyle="--", alpha=0.6, label="RMS")
    ax2.legend()
    plt.colorbar(sc, ax=ax2, label="Azimuth (deg)")

    fig.tight_layout()

    if output_path:
        from pathlib import Path
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig, (ax1, ax2)

