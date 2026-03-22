"""Interactive manual-fit GUI for bootstrapping camera models.

Diagnostic tool: click known objects in an all-sky image to determine
camera rotation and geometry. Useful when automated blind-solve fails
or as a starting point for instrument-fit.
"""

import warnings

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # fall back to whatever is available

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from .projection import CameraModel, ProjectionType


# -----------------------------------------------------------------------
# Named bright stars visible from both hemispheres
# (name, ra_deg J2000, dec_deg J2000, vmag)
# -----------------------------------------------------------------------
NAMED_STARS = [
    ("Sirius",          101.287, -16.716,  -1.46),
    ("Canopus",          95.988, -52.696,  -0.74),
    ("Alpha Centauri",  219.902, -60.834,  -0.27),
    ("Arcturus",        213.915,  19.182,  -0.05),
    ("Vega",            279.235,  38.784,   0.03),
    ("Capella",          79.172,  45.998,   0.08),
    ("Rigel",            78.634,  -8.202,   0.13),
    ("Procyon",         114.826,   5.225,   0.34),
    ("Betelgeuse",       88.793,   7.407,   0.42),
    ("Achernar",         24.429, -57.237,   0.46),
    ("Altair",          297.696,   8.868,   0.77),
    ("Aldebaran",        68.980,  16.509,   0.86),
    ("Spica",           201.298, -11.161,   0.97),
    ("Antares",         247.352, -26.432,   1.04),
    ("Pollux",          116.329,  28.026,   1.14),
    ("Fomalhaut",       344.413, -29.622,   1.16),
    ("Deneb",           310.358,  45.280,   1.25),
    ("Regulus",         152.093,  11.967,   1.35),
    ("Acrux",           186.650, -63.100,   1.33),
    ("Mimosa",          191.930, -59.689,   1.25),
    ("Gacrux",          187.791, -57.113,   1.63),
    ("Shaula",          263.402, -37.104,   1.62),
    ("Bellatrix",        81.283,   6.350,   1.64),
    ("Alnilam",          84.053,  -1.202,   1.69),
    ("Alnitak",          85.190,  -1.943,   1.77),
]


def get_identifiable_objects(lat_deg, lon_deg, obs_time):
    """Return bright objects visible from the given location and time.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Observer latitude/longitude in degrees.
    obs_time : astropy.time.Time
        Observation time.

    Returns
    -------
    list of dict
        Each dict has: name, az_deg, alt_deg, vmag, category.
        Sorted by brightness (lowest vmag first). Filtered to alt > 5 deg.
    """
    from astropy.coordinates import (
        EarthLocation, AltAz, SkyCoord, get_body,
    )
    import astropy.units as u

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    frame = AltAz(obstime=obs_time, location=location)

    objects = []

    # Planets
    planet_vmags = {
        "Moon": -12.0, "Venus": -4.0, "Jupiter": -2.5,
        "Mars": -1.5, "Saturn": 0.5, "Mercury": 0.0,
    }
    for name, approx_vmag in planet_vmags.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                body = get_body(name.lower(), obs_time, location)
                altaz = body.transform_to(frame)
            alt_d = float(altaz.alt.deg)
            az_d = float(altaz.az.deg)
            if alt_d > 5:
                objects.append({
                    "name": name, "az_deg": az_d, "alt_deg": alt_d,
                    "vmag": approx_vmag, "category": "planet",
                })
        except Exception:
            continue

    # Magellanic Clouds
    dso_list = [
        ("LMC", 80.894, -69.756, 0.9),
        ("SMC", 13.187, -72.828, 2.7),
    ]
    for name, ra_d, dec_d, vmag in dso_list:
        try:
            coord = SkyCoord(ra=ra_d * u.deg, dec=dec_d * u.deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                altaz = coord.transform_to(frame)
            alt_d = float(altaz.alt.deg)
            az_d = float(altaz.az.deg)
            if alt_d > 5:
                objects.append({
                    "name": name, "az_deg": az_d, "alt_deg": alt_d,
                    "vmag": vmag, "category": "dso",
                })
        except Exception:
            continue

    # Named stars
    for name, ra_d, dec_d, vmag in NAMED_STARS:
        try:
            coord = SkyCoord(ra=ra_d * u.deg, dec=dec_d * u.deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                altaz = coord.transform_to(frame)
            alt_d = float(altaz.alt.deg)
            az_d = float(altaz.az.deg)
            if alt_d > 5:
                objects.append({
                    "name": name, "az_deg": az_d, "alt_deg": alt_d,
                    "vmag": vmag, "category": "star",
                })
        except Exception:
            continue

    # Sort by brightness
    objects.sort(key=lambda o: o["vmag"])
    return objects


def solve_from_clicks(click_px, click_py, click_az_rad, click_alt_rad,
                       image_shape):
    """Solve for CameraModel from user-clicked correspondences.

    Parameters
    ----------
    click_px, click_py : array-like
        Pixel positions of clicked objects.
    click_az_rad, click_alt_rad : array-like
        Sky positions (radians) of clicked objects.
    image_shape : tuple
        (ny, nx) image dimensions.

    Returns
    -------
    CameraModel
        Solved camera model.
    """
    px = np.asarray(click_px, dtype=np.float64)
    py = np.asarray(click_py, dtype=np.float64)
    az = np.asarray(click_az_rad, dtype=np.float64)
    alt = np.asarray(click_alt_rad, dtype=np.float64)
    ny, nx = image_shape

    n = len(px)
    if n < 3:
        raise ValueError(f"Need at least 3 clicks, got {n}")

    # --- Phase A: geometric bootstrap ---

    # Estimate center from image center
    cx0 = nx / 2.0
    cy0 = ny / 2.0

    # Estimate focal length from angular/pixel separations of all pairs
    f_estimates = []
    for i in range(n):
        for j in range(i + 1, n):
            dpx = np.sqrt((px[i] - px[j])**2 + (py[i] - py[j])**2)
            if dpx < 10:
                continue
            # Angular separation on sky
            cos_sep = (np.sin(alt[i]) * np.sin(alt[j]) +
                       np.cos(alt[i]) * np.cos(alt[j]) *
                       np.cos(az[i] - az[j]))
            cos_sep = np.clip(cos_sep, -1, 1)
            ang_sep = np.arccos(cos_sep)
            if ang_sep < np.radians(5):
                continue
            # For equidistant projection, angular separation maps to
            # pixel separation through f.  This is approximate since
            # it ignores projection effects, but good for a bootstrap.
            # Zenith angle difference approach: for each star,
            # theta = pi/2 - alt, r = f*theta.
            # Use the per-star zenith angles as a better estimator.
            theta_i = np.pi / 2 - alt[i]
            theta_j = np.pi / 2 - alt[j]
            r_i = np.sqrt((px[i] - cx0)**2 + (py[i] - cy0)**2)
            r_j = np.sqrt((px[j] - cx0)**2 + (py[j] - cy0)**2)
            if theta_i > 0.05:
                f_estimates.append(r_i / theta_i)
            if theta_j > 0.05:
                f_estimates.append(r_j / theta_j)

    if f_estimates:
        f0 = float(np.median(f_estimates))
    else:
        f0 = min(nx, ny) * 0.5

    # Estimate roll (rho) from PA offset
    # For a zenith-pointing camera with az0~0:
    # pixel PA = arctan2(x - cx, y - cy)
    # sky PA = azimuth
    # rho = median(pixel_PA - sky_az)
    rho_estimates = []
    for i in range(n):
        img_pa = np.arctan2(px[i] - cx0, py[i] - cy0)
        rho_estimates.append(img_pa - az[i])
    rho0 = float(np.median(rho_estimates)) % (2 * np.pi)
    # Keep in [0, 2*pi)
    az0_0 = 0.0
    alt0_0 = np.pi / 2  # assume zenith-pointing

    # --- Phase B: least-squares refinement ---

    p0 = np.array([cx0, cy0, az0_0, alt0_0, rho0, f0])

    bounds_lo = [
        cx0 - nx * 0.3, cy0 - ny * 0.3,
        az0_0 - np.pi, alt0_0 - 0.5,
        rho0 - np.pi, f0 * 0.3,
    ]
    bounds_hi = [
        cx0 + nx * 0.3, cy0 + ny * 0.3,
        az0_0 + np.pi, alt0_0 + 0.1,
        rho0 + np.pi, f0 * 3.0,
    ]

    def residuals(params):
        m = CameraModel(
            cx=params[0], cy=params[1],
            az0=params[2], alt0=params[3],
            rho=params[4], f=params[5],
            proj_type=ProjectionType.EQUIDISTANT,
            k1=0.0, k2=0.0,
        )
        mx, my = m.sky_to_pixel(az, alt)
        return np.concatenate([mx - px, my - py])

    result = least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi),
                           method="trf", max_nfev=5000)
    p = result.x

    return CameraModel(
        cx=p[0], cy=p[1], az0=p[2], alt0=p[3],
        rho=p[4], f=p[5],
        proj_type=ProjectionType.EQUIDISTANT,
        k1=0.0, k2=0.0,
    )


class ManualFitGUI:
    """Interactive GUI for correcting camera model fits.

    Workflow: start from an auto-fit model (or scratch), see where the
    model PREDICTS identifiable objects are (yellow diamonds with labels),
    then click on the label to select it, then click where the object
    ACTUALLY is in the image.  3+ corrections re-solve the model.

    Parameters
    ----------
    image : ndarray
        2D image array.
    objects : list of dict
        From get_identifiable_objects(). Each has name, az_deg, alt_deg, vmag.
    cat_table : astropy.table.Table
        Full star catalog (for guided_refine overlay).
    meta : dict
        Frame metadata (obs_time, lat_deg, lon_deg, etc.).
    output_path : str
        Default output path for saving the model JSON.
    initial_model : CameraModel, optional
        If provided, start with this model and draw overlay immediately.
    """

    def __init__(self, image, objects, cat_table, meta, output_path,
                 initial_model=None):
        self.image = image
        self.objects = objects
        self.cat_table = cat_table
        self.meta = meta
        self.output_path = output_path
        self.initial_model = initial_model

        # Correction tracking: list of (obj_index, actual_px, actual_py)
        self.corrections = []
        self.pending_obj_idx = None  # selected, waiting for click
        self._pick_consumed = False  # debounce: suppress click after pick

        # Model state
        self.model = initial_model
        self.n_guided = 0
        self.guided_rms = 0.0

        # Overlay artists
        self._obj_artists = []   # predicted object markers (clickable)
        self._grid_artists = []  # grid + catalog overlay
        self._corr_artists = []  # correction arrows

    def run(self):
        """Launch the interactive GUI."""
        self._setup_figure()
        if self.model is not None:
            self._draw_overlay()
        self._draw_predicted_objects()
        self._print_help()
        self.fig.canvas.draw_idle()
        plt.show()

    def _print_help(self):
        """Print instructions to terminal."""
        print("\n=== Manual Fit Tool ===")
        print("The image shows predicted positions of identifiable objects")
        print("(yellow diamonds). Click a label to select it, then click")
        print("where the object ACTUALLY is in the image.\n")
        print("Keyboard shortcuts:")
        print("  r  = run guided refine (auto-match hundreds of stars)")
        print("  d  = run guided refine WITH distortion fitting")
        print("  m  = mirror (flip East-West) — test if image is flipped")
        print("  u  = undo last correction")
        print("  s  = save model")
        print("  q  = quit\n")
        if self.model is not None:
            print(f"Starting model: f={self.model.f:.1f}, "
                  f"rho={np.degrees(self.model.rho):.1f}°")
        print(f"{len(self.objects)} identifiable objects shown.\n")

    def _setup_figure(self):
        """Create the matplotlib figure and connect events."""
        ny, nx = self.image.shape
        dpi = 100
        # Cap figure size for very large images
        scale = min(1.0, 1600.0 / max(nx, ny))
        fig = plt.figure(figsize=(nx * scale / dpi, ny * scale / dpi),
                         dpi=dpi)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)
        ax = fig.add_subplot(111)

        vmin, vmax = np.percentile(self.image, [1, 99.5])
        ax.imshow(self.image, cmap="gray", origin="lower",
                  vmin=vmin, vmax=vmax)
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)

        self.fig = fig
        self.ax = ax
        self._status_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            color="white", fontsize=10, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                      alpha=0.7, edgecolor="gray"),
        )
        self._prompt_text = ax.text(
            0.5, 0.02,
            "Click a yellow label to select, then click actual position",
            transform=ax.transAxes,
            color="yellow", fontsize=11, va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                      alpha=0.8, edgecolor="yellow"),
        )

        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("pick_event", self._on_pick)
        fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._update_status()

    def _draw_predicted_objects(self):
        """Draw predicted positions of identifiable objects as clickable labels."""
        for a in self._obj_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._obj_artists = []

        if self.model is None:
            return

        ny, nx = self.image.shape
        for i, obj in enumerate(self.objects):
            az_rad = np.radians(obj["az_deg"])
            alt_rad = np.radians(obj["alt_deg"])
            px, py = self.model.sky_to_pixel(
                np.array([az_rad]), np.array([alt_rad]))
            x, y = float(px[0]), float(py[0])

            if not (np.isfinite(x) and np.isfinite(y)
                    and -50 < x < nx + 50 and -50 < y < ny + 50):
                continue

            # Diamond marker
            marker = self.ax.plot(
                x, y, 'D', color='#ffcc44', markersize=8,
                markeredgewidth=1.5, markerfacecolor='none',
                picker=True, pickradius=15)[0]
            marker._obj_idx = i

            # Label text (also pickable)
            txt = self.ax.text(
                x + 12, y + 8, obj["name"],
                color='#ffcc44', fontsize=9, fontweight='bold',
                picker=True,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.7, edgecolor="#ffcc44"))
            txt._obj_idx = i

            self._obj_artists.extend([marker, txt])

    def _on_pick(self, event):
        """Handle clicking on a predicted object label."""
        artist = event.artist
        idx = getattr(artist, '_obj_idx', None)
        if idx is None:
            return

        # If we already have a pending selection and the user picks a
        # different object, just switch selection (don't place).
        obj = self.objects[idx]
        self.pending_obj_idx = idx
        # Suppress the button_press_event that fires from the same click
        self._pick_consumed = True
        self._prompt_text.set_text(
            f"Now click where {obj['name']} actually is in the image")
        self._prompt_text.set_color('#00ffff')
        self.fig.canvas.draw_idle()
        print(f"  Selected: {obj['name']} — click its actual position")

    def _on_key(self, event):
        """Handle key press events."""
        key = event.key
        if key == 'q':
            plt.close(self.fig)
        elif key == 'u':
            self._undo_last()
        elif key == 's':
            self._save_model()
        elif key == 'r':
            self._run_guided_refine(fit_distortion=False)
        elif key == 'd':
            self._run_guided_refine(fit_distortion=True)
        elif key == 'm':
            self._toggle_mirror()

    def _toggle_mirror(self):
        """Flip the model's x-axis (test East-West mirror hypothesis)."""
        if self.model is None:
            print("  No model to mirror.")
            return
        # Mirror by negating the rho and az0 signs, which flips
        # the x-axis mapping in sky_to_pixel.
        # Equivalently: reflect pixel x about cx.
        ny, nx = self.image.shape
        self.model = CameraModel(
            cx=self.model.cx, cy=self.model.cy,
            az0=-self.model.az0, alt0=self.model.alt0,
            rho=-self.model.rho, f=self.model.f,
            proj_type=self.model.proj_type,
            k1=self.model.k1, k2=self.model.k2,
        )
        print(f"  Mirrored! rho={np.degrees(self.model.rho):.1f}°, "
              f"az0={np.degrees(self.model.az0):.1f}°")
        self._clear_grid_overlay()
        self._draw_overlay()
        self._draw_predicted_objects()
        # Clear correction arrows since they're now invalid
        for a in self._corr_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._corr_artists = []
        self.corrections = []
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click — place the actual position of selected object."""
        # Suppress the click that accompanied a pick event (same trackpad tap)
        if self._pick_consumed:
            self._pick_consumed = False
            return
        if event.inaxes != self.ax:
            return
        if self.pending_obj_idx is None:
            return
        if event.button != 1:
            return

        px = float(event.xdata)
        py = float(event.ydata)
        idx = self.pending_obj_idx
        obj = self.objects[idx]

        # Remove any existing correction for this object
        self.corrections = [(i, x, y) for i, x, y in self.corrections
                            if i != idx]
        self.corrections.append((idx, px, py))
        self.pending_obj_idx = None

        # Draw arrow from predicted to actual position
        if self.model is not None:
            az_rad = np.radians(obj["az_deg"])
            alt_rad = np.radians(obj["alt_deg"])
            pred_x, pred_y = self.model.sky_to_pixel(
                np.array([az_rad]), np.array([alt_rad]))
            arrow = self.ax.annotate(
                '', xy=(px, py), xytext=(float(pred_x[0]), float(pred_y[0])),
                arrowprops=dict(arrowstyle='->', color='#00ffff', lw=2))
            self._corr_artists.append(arrow)

        # Cyan marker at actual position
        marker = self.ax.plot(px, py, '+', color='#00ffff',
                              markersize=15, markeredgewidth=2)[0]
        label = self.ax.text(px + 12, py - 12, obj["name"],
                             color='#00ffff', fontsize=8)
        self._corr_artists.extend([marker, label])

        self._prompt_text.set_text(
            "Click a yellow label to select, then click actual position")
        self._prompt_text.set_color('yellow')

        print(f"  {obj['name']}: actual position ({px:.0f}, {py:.0f})")

        # Auto-solve if we have 3+ corrections
        if len(self.corrections) >= 3:
            self._auto_solve()

        self._update_status()
        self.fig.canvas.draw_idle()

    def _undo_last(self):
        """Remove the last correction."""
        if not self.corrections:
            print("  Nothing to undo.")
            return
        removed = self.corrections.pop()
        obj = self.objects[removed[0]]
        print(f"  Undone: {obj['name']}")

        # Remove last 3 artists (arrow + marker + label)
        for _ in range(3):
            if self._corr_artists:
                a = self._corr_artists.pop()
                try:
                    a.remove()
                except Exception:
                    pass

        if len(self.corrections) >= 3:
            self._auto_solve()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _auto_solve(self):
        """Re-solve model from correction clicks."""
        click_px = [c[1] for c in self.corrections]
        click_py = [c[2] for c in self.corrections]
        click_az = [np.radians(self.objects[c[0]]["az_deg"])
                    for c in self.corrections]
        click_alt = [np.radians(self.objects[c[0]]["alt_deg"])
                     for c in self.corrections]

        try:
            self.model = solve_from_clicks(
                click_px, click_py, click_az, click_alt,
                self.image.shape)
            self.n_guided = 0
            self.guided_rms = 0.0

            mx, my = self.model.sky_to_pixel(
                np.array(click_az), np.array(click_alt))
            rms = float(np.sqrt(np.mean(
                (mx - np.array(click_px))**2 +
                (my - np.array(click_py))**2)))
            print(f"  Solved: f={self.model.f:.1f}, "
                  f"rho={np.degrees(self.model.rho):.1f} deg, "
                  f"RMS={rms:.1f} px")

            self._clear_grid_overlay()
            self._draw_overlay()
            self._draw_predicted_objects()
        except Exception as e:
            print(f"  Solve failed: {e}")

    def _clear_grid_overlay(self):
        """Remove grid/catalog overlay artists."""
        # Keep click markers (they are first in the list), remove grid artists
        # Actually, we track grid artists separately
        for artist in self._grid_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._grid_artists = []

    def _draw_overlay(self):
        """Draw alt/az grid and catalog stars using current model."""
        if self.model is None:
            return

        self._clear_grid_overlay()

        ny, nx = self.image.shape
        model = self.model
        n_samples = 200

        # --- Altitude circles ---
        alt_circles_deg = [10, 20, 30, 45, 60, 80]
        for alt_deg in alt_circles_deg:
            az_samples = np.linspace(0, 2 * np.pi, n_samples)
            alt_rad = np.full(n_samples, np.radians(alt_deg))
            x, y = model.sky_to_pixel(az_samples, alt_rad)

            valid = (np.isfinite(x) & np.isfinite(y) &
                     (x >= -nx * 0.1) & (x < nx * 1.1) &
                     (y >= -ny * 0.1) & (y < ny * 1.1))
            if np.sum(valid) < 2:
                continue

            xv, yv = x[valid], y[valid]
            artists = self._plot_segments(xv, yv, nx, color="white",
                                          linestyle="--", alpha=0.5,
                                          linewidth=0.8)
            self._grid_artists.extend(artists)

            # Label
            in_frame = (xv >= 0) & (xv < nx) & (yv >= 0) & (yv < ny)
            if np.any(in_frame):
                idx = np.where(in_frame)[0][np.sum(in_frame) // 2]
                txt = self.ax.text(
                    xv[idx], yv[idx], f"{alt_deg} deg",
                    color="white", fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                              alpha=0.6, edgecolor="none"))
                self._grid_artists.append(txt)

        # --- Azimuth lines ---
        az_lines_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        cardinal_names = {
            0: "N", 45: "NE", 90: "E", 135: "SE",
            180: "S", 225: "SW", 270: "W", 315: "NW",
        }
        for az_deg in az_lines_deg:
            alt_samples = np.linspace(np.radians(5), np.radians(89), n_samples)
            az_rad = np.full(n_samples, np.radians(az_deg))
            x, y = model.sky_to_pixel(az_rad, alt_samples)

            valid = (np.isfinite(x) & np.isfinite(y) &
                     (x >= -nx * 0.1) & (x < nx * 1.1) &
                     (y >= -ny * 0.1) & (y < ny * 1.1))
            if np.sum(valid) < 2:
                continue

            xv, yv = x[valid], y[valid]
            artists = self._plot_segments(xv, yv, nx, color="white",
                                          linestyle="--", alpha=0.5,
                                          linewidth=0.8)
            self._grid_artists.extend(artists)

            # Compass label at horizon
            lx, ly = model.sky_to_pixel(np.radians(az_deg), np.radians(5))
            if (np.isfinite(lx) and np.isfinite(ly)
                    and 0 <= lx < nx and 0 <= ly < ny):
                label = cardinal_names.get(az_deg, f"{az_deg} deg")
                txt = self.ax.text(
                    float(lx), float(ly), label,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                              alpha=0.7, edgecolor="none"))
                self._grid_artists.append(txt)

        # --- Catalog stars (red circles for bright, predicted positions) ---
        if self.cat_table is not None:
            cat_az = np.radians(
                np.asarray(self.cat_table["az_deg"], dtype=np.float64))
            cat_alt = np.radians(
                np.asarray(self.cat_table["alt_deg"], dtype=np.float64))
            cat_x, cat_y = model.sky_to_pixel(cat_az, cat_alt)
            vmag = np.asarray(self.cat_table["vmag_extinct"], dtype=np.float64)

            order = np.argsort(vmag)
            n_shown = 0
            for i in order:
                if n_shown >= 150:
                    break
                cx, cy = float(cat_x[i]), float(cat_y[i])
                if not (np.isfinite(cx) and np.isfinite(cy)
                        and 0 <= cx < nx and 0 <= cy < ny):
                    continue
                radius = max(3, int(10 - 1.2 * float(vmag[i])))
                from matplotlib.patches import Circle
                circle = Circle(
                    (cx, cy), radius=radius,
                    facecolor="none", edgecolor="#ff4444",
                    linewidth=0.8, alpha=0.6)
                self.ax.add_patch(circle)
                self._grid_artists.append(circle)
                n_shown += 1

    def _plot_segments(self, x, y, nx, **kwargs):
        """Plot a line, breaking at large pixel jumps. Returns artist list."""
        artists = []
        if len(x) < 2:
            return artists
        dx = np.diff(x)
        dy = np.diff(y)
        dist = np.sqrt(dx**2 + dy**2)
        breaks = np.where(dist > nx * 0.3)[0]

        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(x)]])

        for s, e in zip(starts, ends):
            if e - s >= 2:
                line, = self.ax.plot(x[s:e], y[s:e], **kwargs)
                artists.append(line)
        return artists

    def _run_guided_refine(self, fit_distortion=False):
        """Run guided_refine against the image using current model."""
        if self.model is None:
            print("  No model to refine. Identify at least 3 objects first.")
            return

        label = "+distortion" if fit_distortion else "geometry"
        print(f"  Running guided refine ({label})...")

        from .strategies import guided_refine, _adaptive_min_peak_offset

        cat_az = np.radians(
            np.asarray(self.cat_table["az_deg"], dtype=np.float64))
        cat_alt = np.radians(
            np.asarray(self.cat_table["alt_deg"], dtype=np.float64))

        background = float(np.median(self.image))
        p999 = float(np.percentile(self.image, 99.9))
        min_peak_offset = _adaptive_min_peak_offset(background, p999)

        refined, n_matched, rms = guided_refine(
            self.image, cat_az, cat_alt, self.model,
            n_iterations=15,
            min_peak_offset=min_peak_offset,
            fit_distortion=fit_distortion,
            initial_search_radius=30,
        )

        if n_matched >= 6:
            self.model = refined
            self.n_guided = n_matched
            self.guided_rms = rms
            print(f"  Guided refine: {n_matched} matches, "
                  f"RMS={rms:.2f} px")
            print(f"    f={refined.f:.1f}, "
                  f"rho={np.degrees(refined.rho):.1f} deg, "
                  f"k1={refined.k1:.2e}")

            self._clear_grid_overlay()
            self._draw_overlay()
            self._draw_predicted_objects()
            self._update_status()
            self.fig.canvas.draw_idle()
        else:
            print(f"  Guided refine failed: only {n_matched} matches")

    def _save_model(self):
        """Save the current model as an InstrumentModel JSON."""
        if self.model is None:
            print("  No model to save.")
            return

        from .instrument import InstrumentModel
        from datetime import datetime, timezone

        ny, nx = self.image.shape
        inst = InstrumentModel.from_camera_model(
            self.model,
            site_lat=self.meta.get("lat_deg", 0.0),
            site_lon=self.meta.get("lon_deg", 0.0),
            image_width=nx,
            image_height=ny,
            n_stars_matched=self.n_guided or len(self.corrections),
            rms_residual_px=self.guided_rms,
            fit_timestamp=datetime.now(timezone.utc).isoformat(),
            frame_used="manual-fit",
        )

        import pathlib
        pathlib.Path(self.output_path).parent.mkdir(parents=True,
                                                     exist_ok=True)
        inst.save(self.output_path)
        print(f"  Model saved to {self.output_path}")

    def _update_status(self):
        """Update the stats text on the figure."""
        n_id = len(self.corrections)
        parts = [f"{n_id} identifications"]
        if self.n_guided > 0:
            parts.append(f"{self.n_guided} guided matches")
            parts.append(f"RMS={self.guided_rms:.1f}px")
        elif self.model is not None and n_id >= 3:
            # Show click-based RMS
            click_px = np.array([c[1] for c in self.corrections])
            click_py = np.array([c[2] for c in self.corrections])
            click_az = np.array([np.radians(self.objects[c[0]]["az_deg"])
                                 for c in self.corrections])
            click_alt = np.array([np.radians(self.objects[c[0]]["alt_deg"])
                                  for c in self.corrections])
            mx, my = self.model.sky_to_pixel(click_az, click_alt)
            rms = float(np.sqrt(np.mean(
                (mx - click_px)**2 + (my - click_py)**2)))
            parts.append(f"RMS={rms:.1f}px")

        self._status_text.set_text(", ".join(parts))
