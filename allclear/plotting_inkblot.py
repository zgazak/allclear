"""Annotated all-sky image plotting using inkblot (no matplotlib).

Drop-in replacement for plotting.plot_frame that uses the inkblot Rust
rendering backend for faster image generation.
"""

import warnings

import numpy as np

import inkblot
from inkblot.scale import zscale


def _fy(y, ny):
    """Flip y from FITS convention (0=bottom) to screen convention (0=top)."""
    return (ny - 1) - y


def plot_frame(image, camera_model, det_table=None, cat_table=None,
               matched_pairs=None, show_grid=True, transmission_data=None,
               obs_time=None, lat_deg=None, lon_deg=None,
               output_path=None, dpi=None, horizon_r=None,
               horizon_center=None):
    """Render an annotated all-sky camera frame using inkblot.

    Same signature as plotting.plot_frame for drop-in replacement.
    """
    ny, nx = image.shape

    # Base image with zscale
    scaled = zscale(image, contrast=0.2)
    fig = inkblot.Figure(width=nx, height=ny, frameless=True, cmap="gray")
    fig.imshow(scaled, vmin=0.0, vmax=1.0, cmap="gray", origin="lower")

    # Transmission overlay (under grid/stars)
    if transmission_data is not None:
        trans_az, trans_alt, trans_vals = transmission_data
        _overlay_transmission(fig, camera_model, trans_az, trans_alt,
                              trans_vals, nx, ny)

    # Az/Alt grid
    if show_grid:
        _draw_altaz_grid(fig, camera_model, nx, ny)

    # Horizon circle
    if horizon_r is not None:
        hcx = horizon_center[0] if horizon_center else camera_model.cx
        hcy = _fy(horizon_center[1] if horizon_center else camera_model.cy, ny)
        fig.circle(hcx, hcy, radius=horizon_r, color=[68, 136, 255, 230],
                   linewidth=2.0)

    # Stars
    if matched_pairs is not None:
        _draw_stars(fig, det_table, cat_table, matched_pairs,
                    camera_model, nx, ny)

    # Planets
    if obs_time is not None and lat_deg is not None and lon_deg is not None:
        _draw_planets(fig, camera_model, obs_time, lat_deg, lon_deg, nx, ny)

    # Legend
    if matched_pairs is not None and cat_table is not None:
        _draw_legend(fig, nx, ny)

    # Version watermark
    fs = _font_size(ny, nx)
    try:
        from importlib.metadata import version
        ver = version("allclear")
    except Exception:
        ver = "dev"
    fig.label(nx - int(0.005 * nx) - 10, int(0.005 * ny) + int(fs * 1.5),
              f"allclear v{ver}",
              font_size=fs * 1.2, color=[255, 255, 255, 200])

    if output_path:
        fig.save(str(output_path))
        return None
    return fig


def _font_size(ny, nx):
    return max(6, min(nx, ny) * 0.01)


def _draw_altaz_grid(fig, camera_model, nx, ny):
    """Draw az/alt coordinate grid projected through the camera model."""
    n_samples = 200
    fs = _font_size(ny, nx)
    max_jump = max(nx, ny) * 0.3

    alt_circles_deg = [10, 20, 30, 45, 60, 80]
    az_lines_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    cardinal_names = {0: "N", 45: "NE", 90: "E", 135: "SE",
                      180: "S", 225: "SW", 270: "W", 315: "NW"}

    grid_color = [255, 255, 255, 153]  # white, alpha=0.6

    # Altitude circles
    for alt_deg in alt_circles_deg:
        az_samples = np.linspace(0, 2 * np.pi, n_samples)
        alt_rad = np.full(n_samples, np.radians(alt_deg))
        x, y = camera_model.sky_to_pixel(az_samples, alt_rad)
        y = _fy(y, ny)  # flip to screen coords

        valid = (np.isfinite(x) & np.isfinite(y)
                 & (x >= -nx * 0.1) & (x < nx * 1.1)
                 & (y >= -ny * 0.1) & (y < ny * 1.1))
        if np.sum(valid) < 2:
            continue

        xv, yv = x[valid].tolist(), y[valid].tolist()
        _add_segments(fig, xv, yv, max_jump, grid_color, dashed=True)

        # Label at midpoint
        in_frame = [(0 <= xv[i] < nx and 0 <= yv[i] < ny)
                    for i in range(len(xv))]
        in_idx = [i for i, m in enumerate(in_frame) if m]
        if in_idx:
            idx = in_idx[len(in_idx) // 2]
            fig.label(xv[idx], yv[idx], f"{alt_deg}\u00b0",
                      font_size=fs, color="white",
                      bg_color=[0, 0, 0, 180])

    # Azimuth lines
    for az_deg in az_lines_deg:
        alt_samples = np.linspace(np.radians(5), np.radians(89), n_samples)
        az_rad = np.full(n_samples, np.radians(az_deg))
        x, y = camera_model.sky_to_pixel(az_rad, alt_samples)
        y = _fy(y, ny)

        valid = (np.isfinite(x) & np.isfinite(y)
                 & (x >= -nx * 0.1) & (x < nx * 1.1)
                 & (y >= -ny * 0.1) & (y < ny * 1.1))
        if np.sum(valid) < 2:
            continue

        xv, yv = x[valid].tolist(), y[valid].tolist()
        _add_segments(fig, xv, yv, max_jump, grid_color, dashed=True)

        # Cardinal direction label at alt=5°
        lx, ly = camera_model.sky_to_pixel(np.radians(az_deg), np.radians(5))
        ly = _fy(ly, ny)
        if (np.isfinite(lx) and np.isfinite(ly)
                and 0 <= lx < nx and 0 <= ly < ny):
            label = cardinal_names.get(az_deg, f"{az_deg}\u00b0")
            fig.label(float(lx), float(ly), label,
                      font_size=fs, color="white",
                      bg_color=[0, 0, 0, 180])


def _add_segments(fig, x, y, max_jump, color, dashed=False, linewidth=1.0):
    """Add polyline segments, breaking at large jumps."""
    if len(x) < 2:
        return
    segments = []
    cur_xs, cur_ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        dist = ((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2) ** 0.5
        if dist > max_jump:
            if len(cur_xs) >= 2:
                segments.append((cur_xs, cur_ys))
            cur_xs, cur_ys = [x[i]], [y[i]]
        else:
            cur_xs.append(x[i])
            cur_ys.append(y[i])
    if len(cur_xs) >= 2:
        segments.append((cur_xs, cur_ys))

    for xs, ys in segments:
        fig.polyline(xs, ys, color=color, linewidth=linewidth, dashed=dashed)


def _draw_stars(fig, det_table, cat_table, matched_pairs, camera_model,
                nx, ny):
    """Draw matched star markers."""
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
            green = [68, 255, 68, 180]

            # Green square at detected centroid
            if len(det_table) > di:
                dx = float(det_table["x"][di])
                dy = _fy(float(det_table["y"][di]), ny)
                if 0 <= dx < nx and 0 <= dy < ny:
                    side = radius * 2
                    fig.rect(dx - side / 2, dy - side / 2, side, side,
                             color=green, linewidth=lw)

            # Green circle at catalog projected position
            az_rad = np.radians(float(cat_table["az_deg"][ci]))
            alt_rad = np.radians(float(cat_table["alt_deg"][ci]))
            cx, cy = camera_model.sky_to_pixel(az_rad, alt_rad)
            cy = _fy(cy, ny)
            if (np.isfinite(cx) and np.isfinite(cy)
                    and 0 <= cx < nx and 0 <= cy < ny):
                fig.circle(float(cx), float(cy), radius=radius,
                           color=green, linewidth=lw)

    # Unmatched catalog stars — red circles
    if cat_table is not None and camera_model is not None:
        vmag_all = np.array(cat_table["vmag_expected"], dtype=np.float64)
        order = np.argsort(vmag_all)
        n_shown = 0
        red = [255, 68, 68, 180]
        for i in order:
            if n_shown >= 2000:
                break
            if i in matched_cat:
                continue
            az_rad = np.radians(float(cat_table["az_deg"][i]))
            alt_rad = np.radians(float(cat_table["alt_deg"][i]))
            cx, cy = camera_model.sky_to_pixel(az_rad, alt_rad)
            cy = _fy(cy, ny)
            if not (np.isfinite(cx) and np.isfinite(cy)
                    and 0 <= cx < nx and 0 <= cy < ny):
                continue
            radius = max(3, int(10 - 1.2 * float(vmag_all[i])))
            fig.circle(float(cx), float(cy), radius=radius,
                       color=red, linewidth=1.0)
            n_shown += 1


def _overlay_transmission(fig, camera_model, trans_az, trans_alt, trans_vals,
                          nx, ny):
    """Overlay semi-transparent transmission colormap via RBF interpolation."""
    from scipy.interpolate import RBFInterpolator

    valid = (np.isfinite(trans_az) & np.isfinite(trans_alt)
             & np.isfinite(trans_vals))
    if np.sum(valid) < 3:
        return

    az_v = np.radians(trans_az[valid])
    alt_v = np.radians(trans_alt[valid])
    tv = trans_vals[valid]

    pts = np.column_stack([
        np.cos(alt_v) * np.sin(az_v),
        np.cos(alt_v) * np.cos(az_v),
        np.sin(alt_v),
    ])
    rbf = RBFInterpolator(pts, tv, smoothing=5.0,
                          kernel="thin_plate_spline")

    step = 8
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    grid_az, grid_alt = camera_model.pixel_to_sky(
        xx.astype(np.float64).ravel(),
        yy.astype(np.float64).ravel(),
    )

    below_horizon = grid_alt < 0
    invalid_sky = ~np.isfinite(grid_az) | ~np.isfinite(grid_alt)
    mask = below_horizon | invalid_sky

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

    # Use extent to let Rust bilinear-resample the coarse grid to full resolution
    fig.imshow(trans_img, vmin=0, vmax=1.2, cmap="RdYlGn", alpha=0.55,
               origin="lower", extent=[0, nx, 0, ny])


def _draw_planets(fig, camera_model, obs_time, lat_deg, lon_deg, nx, ny):
    """Annotate planets and DSOs visible in the frame."""
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

    fs = _font_size(ny, nx)

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

        px, py = camera_model.sky_to_pixel(
            np.radians(altaz.az.deg), np.radians(altaz.alt.deg))
        py = _fy(py, ny)
        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue

        px, py = float(px), float(py)
        r = max(8, int(fs * 1.5))
        fig.diamond(px, py, size=r, color=color, linewidth=1.5)
        fig.label(px + r + 2, py, name, font_size=fs * 0.9,
                  color=color, stroke=True)

    # Magellanic Clouds
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

        if altaz.alt.deg < 0:
            continue

        px, py = camera_model.sky_to_pixel(
            np.radians(altaz.az.deg), np.radians(altaz.alt.deg))
        py = _fy(py, ny)
        if not (np.isfinite(px) and np.isfinite(py)
                and 0 <= px < nx and 0 <= py < ny):
            continue

        px, py = float(px), float(py)
        r = max(8, int(fs * 1.5))
        fig.diamond(px, py, size=r, color=color, linewidth=1.5)
        fig.label(px + r + 2, py, name, font_size=fs * 0.9,
                  color=color, stroke=True)


def _draw_legend(fig, nx, ny):
    """Draw a simple legend box at bottom-left."""
    fs = _font_size(ny, nx) * 0.85
    x0 = 10
    y0 = ny - 10
    line_h = fs + 6
    pad = 6

    items = [
        ([68, 255, 68, 255], "\u25a1", "Matched (detected)"),
        ([68, 255, 68, 255], "\u25cb", "Matched (catalog)"),
        ([255, 68, 68, 255], "\u25cb", "Catalog (predicted)"),
    ]

    # Background box
    box_w = fs * 16
    box_h = len(items) * line_h + pad * 2
    fig.rect(x0, y0 - box_h, box_w, box_h,
             color=[0, 0, 0, 153], filled=True)

    for i, (color, marker, text) in enumerate(items):
        ly = y0 - pad - i * line_h
        fig.label(x0 + pad, ly, f"{marker} {text}",
                  font_size=fs, color=color)
