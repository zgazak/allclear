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
from typing import Optional

import numpy as np

from .obscuration import ObscurationMask
from .transmission import TransmissionMap


# Per-direction status values returned by the dict-shaped query methods.
# Order of checks at query time: STALE → BELOW_HORIZON → OBSCURED →
# NO_DATA → CLEAR/CLOUDY.  OBSCURED means a persistent occlusion (dome,
# tree, horizon terrain, dead column), distinct from CLOUDY which is a
# transient transmission loss measured photometrically.  This split is
# what makes the result actionable for link triage: cloud → "wait it
# out", obscured → "you can never shoot there", clear-but-failed →
# "look at the pointing chain".
STATUS_STALE = "STALE"
STATUS_BELOW_HORIZON = "BELOW_HORIZON"
STATUS_OBSCURED = "OBSCURED"
STATUS_NO_DATA = "NO_DATA"
STATUS_CLEAR = "CLEAR"
STATUS_CLOUDY = "CLOUDY"
STATUS_SGP4_ERROR = "SGP4_ERROR"

#: Minimum elevation considered "above horizon" for status queries.
#: Matches the camera's horizon detection cutoff; downstream OGS code
#: should apply its own (typically higher) elevation gate.
HORIZON_ALT_DEG = 5.0


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
    obscuration : ObscurationMask, optional
        Persistent obscuration mask in sky coordinates (dome, trees,
        horizon terrain, dead columns).  When present, the dict-shaped
        query methods return status="OBSCURED" for directions falling in
        an occluded sector instead of attributing the loss to cloud.
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
    obscuration: Optional[ObscurationMask] = None

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def age_seconds(self, now=None):
        """Seconds elapsed since ``self.obs_time``.

        Parameters
        ----------
        now : astropy.time.Time, datetime, or ISO string, optional
            Reference time.  Defaults to ``astropy.time.Time.now()`` (UTC).

        Returns
        -------
        float
            Age in seconds.  ``+inf`` if ``obs_time`` is unset.
        """
        from astropy.time import Time as AstroTime

        if self.obs_time is None:
            return float("inf")
        obs = self.obs_time if isinstance(self.obs_time, AstroTime) \
            else AstroTime(self.obs_time)
        if now is None:
            now = AstroTime.now()
        elif not isinstance(now, AstroTime):
            now = AstroTime(now)
        return float((now - obs).to_value("s"))

    def is_stale(self, max_age_seconds=300.0, now=None):
        """True if ``age_seconds(now)`` exceeds ``max_age_seconds``."""
        return self.age_seconds(now) > float(max_age_seconds)

    # ------------------------------------------------------------------
    # Point queries
    # ------------------------------------------------------------------

    def query(self, az_deg, alt_deg, max_age_seconds=None):
        """Look up raw transmission at an (az, alt) sky position.

        Low-level query: returns a single float, NaN if the position is
        outside the grid (or if ``max_age_seconds`` is given and the
        result is stale).  Use :meth:`query_azalt` for a dict that
        carries status and the obscured-vs-cloudy distinction.

        Parameters
        ----------
        az_deg : float
            Azimuth in degrees (0=N, 90=E).
        alt_deg : float
            Altitude in degrees (0=horizon, 90=zenith).
        max_age_seconds : float, optional
            If set and the result is older than this, return NaN instead
            of the (potentially stale) cached transmission.

        Returns
        -------
        float
            Transmission (0=opaque, 1=clear). NaN if outside coverage or
            stale.
        """
        if max_age_seconds is not None and self.is_stale(max_age_seconds):
            return float("nan")
        return self.transmission_map.query(az_deg, alt_deg)

    def _classify(self, az_deg, alt_deg, *, threshold=None,
                  max_age_seconds=None):
        """Internal: classify a sky direction into a status dict.

        Shared by :meth:`query_azalt`, :meth:`query_radec`, and the
        satellite/pass-window queries so the status ladder stays
        consistent across entry points.
        """
        thresh = threshold if threshold is not None else self.threshold

        # 1. Staleness short-circuits everything else.
        if max_age_seconds is not None and self.is_stale(max_age_seconds):
            return {
                "az_deg": round(float(az_deg) % 360.0, 2),
                "alt_deg": round(float(alt_deg), 2),
                "transmission": None,
                "status": STATUS_STALE,
                "age_seconds": round(self.age_seconds(), 1),
            }

        az = float(az_deg) % 360.0
        alt = float(alt_deg)

        # 2. Below the camera's horizon — no useful answer.
        if alt < HORIZON_ALT_DEG:
            return {
                "az_deg": round(az, 2),
                "alt_deg": round(alt, 2),
                "transmission": None,
                "status": STATUS_BELOW_HORIZON,
            }

        # 3. Persistent obscuration trumps any transmission value.
        # The mask was excluded from probing during compute_transmission,
        # but RBF interpolation can still produce a value here — that
        # value would lie.  Authoritative answer: OBSCURED.
        if self.obscuration is not None and not bool(
                self.obscuration.is_visible(az, alt)):
            return {
                "az_deg": round(az, 2),
                "alt_deg": round(alt, 2),
                "transmission": None,
                "status": STATUS_OBSCURED,
            }

        # 4. Measured transmission.
        trans = self.transmission_map.query(az, alt)
        if np.isnan(trans):
            return {
                "az_deg": round(az, 2),
                "alt_deg": round(alt, 2),
                "transmission": None,
                "status": STATUS_NO_DATA,
            }

        return {
            "az_deg": round(az, 2),
            "alt_deg": round(alt, 2),
            "transmission": round(float(trans), 3),
            "status": STATUS_CLEAR if trans >= thresh else STATUS_CLOUDY,
        }

    def query_azalt(self, az_deg, alt_deg, *, threshold=None,
                    max_age_seconds=None):
        """Status query at a sky direction in horizontal coordinates.

        Returns the same dict shape as :meth:`query_radec` — useful when
        the caller already has (az, alt) from another source (e.g. a
        mount encoder, an upstream ephemeris service, or a non-TLE
        satellite tracker).

        Parameters
        ----------
        az_deg, alt_deg : float
            Azimuth and altitude in degrees.
        threshold : float, optional
            Override clear/cloudy threshold (default: use self.threshold).
        max_age_seconds : float, optional
            If set and the result is older than this, returns
            status="STALE" instead of a transmission value.

        Returns
        -------
        dict
            Keys: az_deg, alt_deg, transmission, status.  status is one
            of STALE, BELOW_HORIZON, OBSCURED, NO_DATA, CLEAR, CLOUDY.
        """
        return self._classify(az_deg, alt_deg, threshold=threshold,
                              max_age_seconds=max_age_seconds)

    def query_radec(self, ra_deg, dec_deg, threshold=None,
                    max_age_seconds=None):
        """Look up transmission at a (RA, Dec) position.

        Converts equatorial coordinates to horizontal (az/alt) for the
        result's ``obs_time`` and site, then runs the same status ladder
        as :meth:`query_azalt`.

        Parameters
        ----------
        ra_deg : float
            Right ascension in degrees (J2000).
        dec_deg : float
            Declination in degrees (J2000).
        threshold : float, optional
            Override clear/cloudy threshold (default: use self.threshold).
        max_age_seconds : float, optional
            If set and the result is older than this, returns
            status="STALE" instead of a transmission value.

        Returns
        -------
        dict
            Keys: ra_deg, dec_deg, az_deg, alt_deg, transmission, status.
            status is one of: STALE, BELOW_HORIZON, OBSCURED, NO_DATA,
            CLEAR, CLOUDY.
        """
        # Short-circuit on staleness before doing the coord transform.
        if max_age_seconds is not None and self.is_stale(max_age_seconds):
            return {
                "ra_deg": round(float(ra_deg), 4),
                "dec_deg": round(float(dec_deg), 4),
                "az_deg": None,
                "alt_deg": None,
                "transmission": None,
                "status": STATUS_STALE,
                "age_seconds": round(self.age_seconds(), 1),
            }

        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        import astropy.units as u
        import warnings

        location = EarthLocation(
            lat=self.site_lat * u.deg,
            lon=self.site_lon * u.deg,
        )
        frame = AltAz(obstime=self.obs_time, location=location)
        target = SkyCoord(ra=float(ra_deg) * u.deg,
                          dec=float(dec_deg) * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            altaz = target.transform_to(frame)

        out = self._classify(
            float(altaz.az.deg), float(altaz.alt.deg),
            threshold=threshold,
            max_age_seconds=None,  # already checked above
        )
        out["ra_deg"] = round(float(ra_deg), 4)
        out["dec_deg"] = round(float(dec_deg), 4)
        return out

    # ------------------------------------------------------------------
    # Satellite queries
    # ------------------------------------------------------------------

    def query_satellite(self, tle_line1, tle_line2, *, time=None,
                        name=None, threshold=None, max_age_seconds=None):
        """Query transmission at a satellite's current sky position.

        Propagates the supplied TLE to ``time`` (default: the result's
        ``obs_time``), converts the geocentric TEME position to
        topocentric AltAz at the site, then runs the status ladder.

        For OGS link triage this is the "was the sky clear at the
        satellite's bearing?" question.  Pair with the satellite
        terminal's pointing log to disambiguate weather outages from
        targeting errors: status="CLEAR" + dropped link → look at the
        pointing chain; status="CLOUDY"/"OBSCURED" → weather/structure,
        not pointing.

        Parameters
        ----------
        tle_line1, tle_line2 : str
            The two TLE lines (each 69 characters, exactly as published).
        time : str, datetime, or astropy.time.Time, optional
            Epoch at which to evaluate the satellite position.  Default
            is the result's ``obs_time`` (the frame's timestamp), which
            is the only choice consistent with the cached transmission
            map.  Passing a different ``time`` is allowed but means the
            transmission lookup may be stale relative to the satellite
            position — set ``max_age_seconds`` to enforce.
        name : str, optional
            Satellite name to echo in the result (purely for labelling).
        threshold : float, optional
            Override clear/cloudy threshold.
        max_age_seconds : float, optional
            If set and the result is older than this relative to
            ``time``, returns status="STALE".

        Returns
        -------
        dict
            Keys: sat_name, time, az_deg, alt_deg, transmission, status.
            status may also be "SGP4_ERROR" with ``error_code`` set if
            the TLE propagation failed.

        Raises
        ------
        ImportError
            If the optional ``sgp4`` dependency is not installed.
            Install with ``pip install allclear[satellite]`` or
            ``pip install sgp4``.
        """
        try:
            from sgp4.api import Satrec
        except ImportError as e:
            raise ImportError(
                "Satellite queries require the 'sgp4' package. "
                "Install with: pip install allclear[satellite]  "
                "or: pip install sgp4"
            ) from e

        from astropy.coordinates import (
            TEME, AltAz, EarthLocation, CartesianRepresentation,
        )
        from astropy.time import Time as AstroTime
        import astropy.units as u

        t = _coerce_time(time if time is not None else self.obs_time)

        # Staleness measured relative to the queried time, not "now".
        # If a future pass is being evaluated against a current map, the
        # map is stale relative to the future epoch.
        if max_age_seconds is not None and self.obs_time is not None:
            obs = self.obs_time if isinstance(self.obs_time, AstroTime) \
                else AstroTime(self.obs_time)
            dt = abs(float((t - obs).to_value("s")))
            if dt > float(max_age_seconds):
                return {
                    "sat_name": name,
                    "time": t.isot,
                    "az_deg": None,
                    "alt_deg": None,
                    "transmission": None,
                    "status": STATUS_STALE,
                    "age_seconds": round(dt, 1),
                }

        sat = Satrec.twoline2rv(tle_line1, tle_line2)
        err, r_teme, _ = sat.sgp4(t.jd1, t.jd2)
        if err != 0:
            # sgp4 error codes: 1=mean eccentricity out of range,
            # 2=mean motion out of range, 3=pert elements,
            # 4=semi-latus rectum < 0, 5=epoch elements, 6=decayed.
            return {
                "sat_name": name,
                "time": t.isot,
                "az_deg": None,
                "alt_deg": None,
                "transmission": None,
                "status": STATUS_SGP4_ERROR,
                "error_code": int(err),
            }

        location = EarthLocation(
            lat=self.site_lat * u.deg,
            lon=self.site_lon * u.deg,
        )
        teme = TEME(
            CartesianRepresentation(
                r_teme[0] * u.km, r_teme[1] * u.km, r_teme[2] * u.km,
            ),
            obstime=t,
        )
        altaz = teme.transform_to(AltAz(obstime=t, location=location))

        out = self._classify(
            float(altaz.az.deg), float(altaz.alt.deg),
            threshold=threshold,
            max_age_seconds=None,  # already handled above
        )
        out["sat_name"] = name
        out["time"] = t.isot
        return out

    def query_pass_window(self, tle_line1, tle_line2, *, start, end,
                          step=10.0, name=None, threshold=None,
                          max_age_seconds=None):
        """Sample a satellite's status across a pass window.

        Walks the satellite track from ``start`` to ``end`` at
        ``step``-second cadence, classifying each sample, and returns
        per-sample results plus summary stats useful for scheduling
        ("will this pass have enough contiguous clear sky to close the
        link?").

        .. note::

           The transmission map is frozen at ``obs_time``.  For passes
           that occur well after the frame was taken, cloud motion makes
           the map an increasingly poor predictor; use
           ``max_age_seconds`` to flag stale samples explicitly rather
           than silently relying on out-of-date data.

        Parameters
        ----------
        tle_line1, tle_line2 : str
            TLE lines for the satellite.
        start, end : str, datetime, or astropy.time.Time
            Window boundaries.
        step : float
            Sample cadence in seconds (default 10).
        name : str, optional
            Satellite name to echo in each sample.
        threshold : float, optional
            Override clear/cloudy threshold.
        max_age_seconds : float, optional
            Per-sample staleness gate (vs. the sample's epoch, not now).

        Returns
        -------
        dict
            Keys: n_samples, step_seconds, duration_seconds, n_clear,
            n_cloudy, n_obscured, n_below_horizon, n_no_data, n_stale,
            n_sgp4_error, clear_fraction (over samples above horizon),
            longest_clear_window_seconds, max_alt_deg, samples (list of
            per-sample dicts as returned by :meth:`query_satellite`).
        """
        from astropy.time import Time as AstroTime
        import astropy.units as u

        t_start = _coerce_time(start)
        t_end = _coerce_time(end)
        if (t_end - t_start).to_value("s") <= 0:
            raise ValueError("end must be strictly after start")

        step_s = float(step)
        if step_s <= 0:
            raise ValueError("step must be positive")

        duration_s = float((t_end - t_start).to_value("s"))
        # astropy Time arithmetic loses ~1e-13 relative precision, so an
        # exact-multiple window (e.g. 120 s at 30 s step) reads back as
        # 119.99999... and would drop the boundary sample.  Round up
        # when within 1 ms of a step boundary.
        n_samples = int(np.floor(duration_s / step_s + 1e-3 / step_s)) + 1
        offsets_s = np.arange(n_samples) * step_s
        times = [t_start + float(off) * u.s for off in offsets_s]

        samples = [
            self.query_satellite(
                tle_line1, tle_line2,
                time=t, name=name, threshold=threshold,
                max_age_seconds=max_age_seconds,
            )
            for t in times
        ]

        statuses = [s["status"] for s in samples]
        alts = [s["alt_deg"] for s in samples if s.get("alt_deg") is not None]

        n_clear = statuses.count(STATUS_CLEAR)
        n_cloudy = statuses.count(STATUS_CLOUDY)
        n_obscured = statuses.count(STATUS_OBSCURED)
        n_below = statuses.count(STATUS_BELOW_HORIZON)
        n_no_data = statuses.count(STATUS_NO_DATA)
        n_stale = statuses.count(STATUS_STALE)
        n_err = statuses.count(STATUS_SGP4_ERROR)

        # Longest contiguous CLEAR run, in seconds.
        longest = current = 0
        for st in statuses:
            if st == STATUS_CLEAR:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        # Clear-fraction is computed over samples that are above the
        # horizon — denominator is the realistic visibility window, not
        # the whole user-specified time range.
        n_above = n_samples - n_below
        clear_fraction = (n_clear / n_above) if n_above > 0 else 0.0

        return {
            "n_samples": n_samples,
            "step_seconds": step_s,
            "duration_seconds": round(duration_s, 1),
            "n_clear": n_clear,
            "n_cloudy": n_cloudy,
            "n_obscured": n_obscured,
            "n_below_horizon": n_below,
            "n_no_data": n_no_data,
            "n_stale": n_stale,
            "n_sgp4_error": n_err,
            "clear_fraction": round(clear_fraction, 3),
            "longest_clear_window_seconds": round(longest * step_s, 1),
            "max_alt_deg": round(max(alts), 2) if alts else None,
            "samples": samples,
        }

    def to_dict(self):
        """Serialize to a JSON-compatible dict.

        Returns a summary suitable for JSON serialization, including the
        full gridded transmission map, metadata, and per-star data.
        The obscuration mask is summarized (occluded fraction, frame
        count) rather than serialized in full — the full mask lives in
        the model's sidecar JSON.
        """
        obs_summary = None
        if self.obscuration is not None:
            visible = (self.obscuration.weight >= 0.5)
            obs_summary = {
                "occluded_fraction": round(
                    float(1.0 - np.nanmean(visible.astype(float))), 3),
                "n_frames": int(self.obscuration.n_frames),
                "az_bins": int(self.obscuration.weight.shape[1]),
                "alt_bins": int(self.obscuration.weight.shape[0]),
            }
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
            "obscuration": obs_summary,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_time(t):
    """Normalize an arbitrary time-like input to ``astropy.time.Time`` (UTC).

    Accepts: ``astropy.time.Time`` (returned as-is), ``datetime``
    (naive treated as UTC), or ISO-8601 string.
    """
    from astropy.time import Time as AstroTime
    from datetime import datetime, timezone

    if isinstance(t, AstroTime):
        return t
    if isinstance(t, str):
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return AstroTime(dt, scale="utc")
    if isinstance(t, datetime):
        if t.tzinfo is not None:
            t = t.astimezone(timezone.utc).replace(tzinfo=None)
        return AstroTime(t, scale="utc")
    return AstroTime(t)


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
        # Not enough matches — return empty map with status.  Still
        # carry the obscuration mask through so callers can at least
        # tell "this direction is permanently blocked" from "we don't
        # know" when the solve fails.
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
            obscuration=inst.obscuration,
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
        obscuration=inst.obscuration,
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
