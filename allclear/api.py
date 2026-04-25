"""Programmatic API for sky transmission queries.

Provides a single high-level function ``get_sky_transmission`` that
processes an all-sky camera frame and returns a structured result
an agent or script can use to decide which parts of the sky are clear.

For testing and simulation (e.g. argusgym), ``get_test_transmission``
generates synthetic scenarios without real images.

Example
-------
>>> from allclear.api import get_sky_transmission
>>> result = get_sky_transmission("frame.fits", "instrument_model.json")
>>> result.clear_fraction
0.82
>>> result.query(az_deg=180, alt_deg=45)
0.95
>>> result.query_radec(ra_deg=83.63, dec_deg=22.01)  # Crab Nebula
{'az_deg': 210.3, 'alt_deg': 62.1, 'transmission': 0.91, 'status': 'CLEAR'}
>>> result.to_dict()  # JSON-serializable summary
{...}

Test scenarios::

>>> from allclear.api import get_test_transmission
>>> clear = get_test_transmission("clear")
>>> clear.clear_fraction
1.0
>>> overcast = get_test_transmission("overcast")
>>> overcast.clear_fraction
0.0
>>> patchy = get_test_transmission("random")
>>> 0.0 < patchy.clear_fraction < 1.0
True
"""

import pathlib
from dataclasses import dataclass, field

import numpy as np

from .transmission import TransmissionMap


@dataclass
class SkyTransmissionResult:
    """Result of a sky transmission analysis.

    Attributes
    ----------
    transmission_map : TransmissionMap
        Gridded (az, alt) transmission map.
    obs_time : astropy.time.Time
        Observation timestamp.
    site_lat : float
        Observer latitude (degrees).
    site_lon : float
        Observer longitude (degrees).
    n_matched : int
        Number of catalog stars matched in the frame.
    n_expected : int
        Number of catalog stars expected in the frame.
    rms_px : float
        Astrometric RMS residual (pixels).
    status : str
        Solve status: "ok", "low_matches", "camera_shifted", "cloudy".
    threshold : float
        Clear-sky threshold used.
    clear_fraction : float
        Fraction of sky above threshold.
    frame_zeropoint : float
        Photometric zeropoint measured from this frame.
    per_star : list
        Per-star transmission data: list of dicts with az_deg, alt_deg,
        transmission for each measured star.
    """

    transmission_map: TransmissionMap
    obs_time: object = None
    site_lat: float = 0.0
    site_lon: float = 0.0
    n_matched: int = 0
    n_expected: int = 0
    rms_px: float = 0.0
    status: str = "ok"
    threshold: float = 0.7
    clear_fraction: float = 0.0
    frame_zeropoint: float = 0.0
    per_star: list = field(default_factory=list)

    def query(self, az_deg, alt_deg):
        """Look up transmission at an (az, alt) sky position.

        Parameters
        ----------
        az_deg : float
            Azimuth in degrees (0=N, 90=E).
        alt_deg : float
            Altitude in degrees (0=horizon, 90=zenith).

        Returns
        -------
        float
            Transmission (0=opaque, 1=clear). NaN if outside coverage.
        """
        return self.transmission_map.query(az_deg, alt_deg)

    def query_radec(self, ra_deg, dec_deg, threshold=None):
        """Look up transmission at a (RA, Dec) position.

        Converts equatorial coordinates to horizontal (az/alt) for the
        observation time and site, then queries the transmission map.

        Parameters
        ----------
        ra_deg : float
            Right ascension in degrees (J2000).
        dec_deg : float
            Declination in degrees (J2000).
        threshold : float, optional
            Override clear/cloudy threshold (default: use self.threshold).

        Returns
        -------
        dict
            Keys: az_deg, alt_deg, transmission, status.
            status is one of: "CLEAR", "CLOUDY", "NO_DATA", "BELOW_HORIZON".
        """
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        import astropy.units as u
        import warnings

        thresh = threshold if threshold is not None else self.threshold

        location = EarthLocation(
            lat=self.site_lat * u.deg,
            lon=self.site_lon * u.deg,
        )
        frame = AltAz(obstime=self.obs_time, location=location)
        target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            altaz = target.transform_to(frame)

        alt = float(altaz.alt.deg)
        az = float(altaz.az.deg)

        if alt < 5:
            return {
                "az_deg": round(az, 2),
                "alt_deg": round(alt, 2),
                "transmission": float("nan"),
                "status": "BELOW_HORIZON",
            }

        trans = self.transmission_map.query(az, alt)

        if np.isnan(trans):
            status = "NO_DATA"
        elif trans >= thresh:
            status = "CLEAR"
        else:
            status = "CLOUDY"

        return {
            "az_deg": round(az, 2),
            "alt_deg": round(alt, 2),
            "transmission": round(trans, 3) if not np.isnan(trans) else None,
            "status": status,
        }

    def to_dict(self):
        """Serialize to a JSON-compatible dict.

        Returns a summary suitable for JSON serialization, including the
        full gridded transmission map, metadata, and per-star data.
        """
        return {
            "obs_time": str(self.obs_time) if self.obs_time else None,
            "site_lat": self.site_lat,
            "site_lon": self.site_lon,
            "n_matched": self.n_matched,
            "n_expected": self.n_expected,
            "rms_px": round(self.rms_px, 2),
            "status": self.status,
            "threshold": self.threshold,
            "clear_fraction": round(self.clear_fraction, 3),
            "frame_zeropoint": round(self.frame_zeropoint, 4),
            "transmission_map": self.transmission_map.to_dict(),
            "per_star": self.per_star,
        }


def get_sky_transmission(frame, model, *, time=None, threshold=0.7):
    """Process a frame and return a sky transmission map.

    This is the main entry point for programmatic access. It loads the
    image, solves for star positions, measures photometry, and returns
    a structured result with the full transmission map and query methods.

    Parameters
    ----------
    frame : str or Path
        Path to an image file (FITS, JPG, PNG, TIFF).
    model : str, Path, or InstrumentModel
        Path to an instrument model JSON file, or a loaded InstrumentModel.
    time : str or astropy.time.Time, optional
        Observation time override (UTC). Required for images without
        FITS headers or EXIF timestamps. Accepts ISO format strings
        like ``"2024-01-15 03:30:00"`` or ``"2024-01-15T03:30:00+00:00"``.
    threshold : float
        Clear-sky transmission threshold (default 0.7). Used for
        ``clear_fraction`` and ``query_radec`` status.

    Returns
    -------
    SkyTransmissionResult
        Structured result with transmission map, query methods, and
        serialization support.

    Raises
    ------
    FileNotFoundError
        If the frame or model file does not exist.
    ValueError
        If observation time cannot be determined.

    Examples
    --------
    Basic usage::

        result = get_sky_transmission("frame.fits", "model.json")
        print(f"Sky is {result.clear_fraction:.0%} clear")

    Query a specific target::

        info = result.query_radec(ra_deg=83.63, dec_deg=22.01)
        if info["status"] == "CLEAR":
            print("Target is observable!")

    Get the full grid for custom analysis::

        tmap = result.transmission_map
        print(tmap.az_grid)    # (180,) array of azimuths in degrees
        print(tmap.alt_grid)   # (45,) array of altitudes in degrees
        print(tmap.transmission)  # (45, 180) grid of transmission values
    """
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission, interpolate_transmission

    # Load model
    if isinstance(model, (str, pathlib.Path)):
        model_path = pathlib.Path(model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model}")
        inst = InstrumentModel.load(model_path)
    else:
        inst = model

    camera = inst.to_camera_model()

    # Parse time
    obs_time = None
    if time is not None:
        from astropy.time import Time as AstroTime
        if isinstance(time, str):
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(time)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)
            obs_time = AstroTime(dt, scale="utc")
        else:
            obs_time = time

    # Load frame
    frame_path = pathlib.Path(frame)
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame}")

    from .utils import load_image, extract_obs_time, parse_fits_header
    from .catalog import BrightStarCatalog
    from .detection import detect_stars

    data, header = load_image(str(frame_path))
    if header is not None:
        meta = parse_fits_header(header)
        if obs_time is not None:
            meta["obs_time"] = obs_time
    else:
        resolved_time = obs_time or extract_obs_time(frame_path)
        if resolved_time is None:
            raise ValueError(
                f"No observation time for {frame}. "
                "Provide time= or use an image with FITS/EXIF timestamps."
            )
        meta = {
            "obs_time": resolved_time,
            "lat_deg": inst.site_lat,
            "lon_deg": inst.site_lon,
            "exposure": 1.0,
            "focal_mm": 1.8,
            "pixel_um": 2.4,
        }

    catalog = BrightStarCatalog()
    cat = catalog.get_visible_stars(inst.site_lat, inst.site_lon, meta["obs_time"])
    det = detect_stars(data, fwhm=5.0, threshold_sigma=5.0, n_brightest=1000)

    # Solve
    result = fast_solve(data, det, cat, camera, guided=True,
                        obscuration=inst.obscuration)

    if result.n_matched < 3:
        # Not enough matches — return empty map with status
        empty_map = interpolate_transmission(
            np.array([]), np.array([]), np.array([])
        )
        return SkyTransmissionResult(
            transmission_map=empty_map,
            obs_time=meta["obs_time"],
            site_lat=inst.site_lat,
            site_lon=inst.site_lon,
            n_matched=result.n_matched,
            n_expected=result.n_expected,
            rms_px=result.rms_residual,
            status=result.status or "low_matches",
            threshold=threshold,
            clear_fraction=0.0,
        )

    # Transmission
    use_det = (result.guided_det_table
               if result.guided_det_table is not None
               and len(result.guided_det_table) > 0
               else det)
    ref_zp = inst.photometric_zeropoint or None

    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
        obscuration=inst.obscuration,
    )
    tmap = interpolate_transmission(az, alt, trans)

    clear_mask = tmap.get_observability_mask(threshold=threshold)
    clear_frac = float(np.nanmean(clear_mask))

    # Build per-star list
    valid = np.isfinite(trans)
    per_star = [
        {
            "az_deg": round(float(az[i]), 2),
            "alt_deg": round(float(alt[i]), 2),
            "transmission": round(float(trans[i]), 3),
        }
        for i in range(len(az))
        if valid[i]
    ]

    return SkyTransmissionResult(
        transmission_map=tmap,
        obs_time=meta["obs_time"],
        site_lat=inst.site_lat,
        site_lon=inst.site_lon,
        n_matched=result.n_matched,
        n_expected=result.n_expected,
        rms_px=result.rms_residual,
        status=result.status,
        threshold=threshold,
        clear_fraction=clear_frac,
        frame_zeropoint=zp,
        per_star=per_star,
    )


# ---------------------------------------------------------------------------
# Synthetic test scenarios (for argusgym / agent training / testing)
# ---------------------------------------------------------------------------

#: Available scenario names for ``get_test_transmission``.
TEST_SCENARIOS = [
    "clear",
    "overcast",
    "random",
    "band",
    "hole",
    "gradient",
]


def _build_test_grid(n_az=180, n_alt=45):
    """Return az/alt grid arrays and meshgrids."""
    az_grid = np.linspace(0, 360, n_az, endpoint=False)
    alt_grid = np.linspace(5, 90, n_alt)
    AZ, ALT = np.meshgrid(az_grid, alt_grid)
    return az_grid, alt_grid, AZ, ALT


def _scenario_clear(n_az, n_alt, rng):
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)
    trans = np.ones((n_alt, n_az)) * rng.uniform(0.92, 1.05)
    trans += rng.normal(0, 0.02, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


def _scenario_overcast(n_az, n_alt, rng):
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)
    trans = np.ones((n_alt, n_az)) * rng.uniform(0.0, 0.15)
    trans += rng.normal(0, 0.03, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


def _scenario_random(n_az, n_alt, rng):
    """Patchy clouds: Perlin-like blobs via superposed cosine bumps."""
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)

    # Start clear, subtract cloud blobs
    trans = np.ones((n_alt, n_az), dtype=np.float64)
    n_clouds = rng.integers(3, 12)
    for _ in range(n_clouds):
        c_az = rng.uniform(0, 360)
        c_alt = rng.uniform(15, 85)
        radius = rng.uniform(15, 60)  # degrees
        depth = rng.uniform(0.4, 1.0)

        # Angular distance on sphere
        az_r, alt_r = np.radians(AZ), np.radians(ALT)
        c_az_r, c_alt_r = np.radians(c_az), np.radians(c_alt)
        cos_sep = (np.sin(alt_r) * np.sin(c_alt_r) +
                   np.cos(alt_r) * np.cos(c_alt_r) *
                   np.cos(az_r - c_az_r))
        sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))

        cloud = depth * np.exp(-0.5 * (sep_deg / (radius * 0.5)) ** 2)
        trans -= cloud

    trans += rng.normal(0, 0.02, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


def _scenario_band(n_az, n_alt, rng):
    """Cloud band across the sky at a random azimuth."""
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)
    trans = np.ones((n_alt, n_az), dtype=np.float64)

    band_az = rng.uniform(0, 360)
    band_width = rng.uniform(20, 60)  # degrees
    depth = rng.uniform(0.6, 1.0)

    # Distance from the band great circle (simplified: azimuth distance)
    az_diff = np.abs(AZ - band_az)
    az_diff = np.minimum(az_diff, 360 - az_diff)
    band = depth * np.exp(-0.5 * (az_diff / (band_width * 0.5)) ** 2)
    trans -= band

    trans += rng.normal(0, 0.02, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


def _scenario_hole(n_az, n_alt, rng):
    """Mostly overcast with a clear hole."""
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)
    trans = np.ones((n_alt, n_az), dtype=np.float64) * rng.uniform(0.0, 0.15)

    hole_az = rng.uniform(0, 360)
    hole_alt = rng.uniform(30, 80)
    hole_radius = rng.uniform(15, 45)

    az_r, alt_r = np.radians(AZ), np.radians(ALT)
    h_az_r, h_alt_r = np.radians(hole_az), np.radians(hole_alt)
    cos_sep = (np.sin(alt_r) * np.sin(h_alt_r) +
               np.cos(alt_r) * np.cos(h_alt_r) *
               np.cos(az_r - h_az_r))
    sep_deg = np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))

    clear = 0.9 * np.exp(-0.5 * (sep_deg / (hole_radius * 0.5)) ** 2)
    trans += clear

    trans += rng.normal(0, 0.02, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


def _scenario_gradient(n_az, n_alt, rng):
    """Clear at high altitude, cloudy near horizon (or vice versa)."""
    az, alt, AZ, ALT = _build_test_grid(n_az, n_alt)

    if rng.random() > 0.5:
        # Clear zenith, cloudy horizon
        trans = (ALT - 5) / 85.0
    else:
        # Cloudy zenith, clear horizon
        trans = 1.0 - (ALT - 5) / 85.0

    trans = trans * rng.uniform(0.8, 1.0) + rng.uniform(0.0, 0.1)
    trans += rng.normal(0, 0.02, trans.shape)
    return az, alt, np.clip(trans, 0, 1.2)


_SCENARIO_FUNCS = {
    "clear": _scenario_clear,
    "overcast": _scenario_overcast,
    "random": _scenario_random,
    "band": _scenario_band,
    "hole": _scenario_hole,
    "gradient": _scenario_gradient,
}


def get_test_transmission(scenario="random", *, threshold=0.7,
                          seed=None, n_az=180, n_alt=45,
                          site_lat=20.7458, site_lon=-156.4317,
                          obs_time=None):
    """Generate a synthetic sky transmission result for testing.

    Produces a ``SkyTransmissionResult`` with a procedurally generated
    transmission map — no real image or instrument model needed. Useful
    for agent training environments (e.g. argusgym) and unit tests.

    Parameters
    ----------
    scenario : str
        One of:

        - ``"clear"`` — full clear sky (transmission ~1.0 everywhere)
        - ``"overcast"`` — total cloud cover (transmission ~0.0)
        - ``"random"`` — patchy clouds (random blobs)
        - ``"band"`` — cloud band across one azimuth range
        - ``"hole"`` — mostly overcast with one clear hole
        - ``"gradient"`` — altitude-dependent (clear zenith / cloudy horizon
          or vice versa)

    threshold : float
        Clear-sky threshold (default 0.7).
    seed : int, optional
        Random seed for reproducibility.
    n_az, n_alt : int
        Grid resolution (default 180x45).
    site_lat, site_lon : float
        Observer coordinates in degrees (default: Haleakala).
    obs_time : str or astropy.time.Time, optional
        Observation time. Defaults to ``"2024-01-15 03:00:00"`` UTC.

    Returns
    -------
    SkyTransmissionResult
        Fully functional result with ``.query()``, ``.query_radec()``,
        ``.to_dict()``, etc.

    Examples
    --------
    ::

        from allclear.api import get_test_transmission

        # Deterministic clear sky
        sky = get_test_transmission("clear", seed=42)
        assert sky.clear_fraction > 0.95

        # Random patchy clouds, reproducible
        sky = get_test_transmission("random", seed=123)
        print(sky.query(az_deg=180, alt_deg=60))

        # Iterate over scenarios for training
        for scenario in ["clear", "overcast", "random", "band", "hole"]:
            sky = get_test_transmission(scenario, seed=i)
            agent.observe(sky.to_dict())
    """
    if scenario not in _SCENARIO_FUNCS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Choose from: {', '.join(TEST_SCENARIOS)}"
        )

    rng = np.random.default_rng(seed)
    az_grid, alt_grid, trans = _SCENARIO_FUNCS[scenario](n_az, n_alt, rng)

    tmap = TransmissionMap(
        az_grid=az_grid,
        alt_grid=alt_grid,
        transmission=trans,
        zeropoint=15.0,
    )

    clear_mask = tmap.get_observability_mask(threshold=threshold)
    clear_frac = float(np.nanmean(clear_mask))

    # Resolve obs_time
    if obs_time is None:
        from astropy.time import Time as AstroTime
        resolved_time = AstroTime("2024-01-15 03:00:00", scale="utc")
    elif isinstance(obs_time, str):
        from astropy.time import Time as AstroTime
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(obs_time)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        resolved_time = AstroTime(dt, scale="utc")
    else:
        resolved_time = obs_time

    # Synthetic status
    if clear_frac > 0.9:
        status = "ok"
    elif clear_frac < 0.1:
        status = "cloudy"
    else:
        status = "ok"

    return SkyTransmissionResult(
        transmission_map=tmap,
        obs_time=resolved_time,
        site_lat=site_lat,
        site_lon=site_lon,
        n_matched=350,      # synthetic — looks like a good solve
        n_expected=400,
        rms_px=2.5,
        status=status,
        threshold=threshold,
        clear_fraction=clear_frac,
        frame_zeropoint=15.0,
        per_star=[],         # no real stars in synthetic mode
    )
