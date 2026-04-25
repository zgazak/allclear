"""Validate WCS round-trip accuracy for each benchmark camera.

For each solved camera model, convert to an astropy WCS via
``CameraModel.to_wcs()`` and measure how well the WCS reproduces the
native fisheye projection in pixel units.

Metric: for a grid of pixel positions inside the image, project
forward via

    (a) the native fisheye model: pixel -> (az, alt) -> (RA, Dec)
    (b) the WCS (affine + SIP):   pixel -> (RA, Dec)

and report the \emph{angular} discrepancy between the two sky
positions, expressed in pixels via the local pixel scale
$(180/\pi)/f$.  Using pixel-in is cheaper and more robust than
iterative sky-to-pixel inversion (which diverges for high-distortion
cameras beyond the SIP polynomial's accurate region).

Reports median / 90th-percentile / max error for each camera and
writes a CSV.

Run:
    uv run python benchmark/scripts/validate_wcs_roundtrip.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

from allclear.instrument import InstrumentModel


MODELS = [
    ("ESO APICAM",        "benchmark/solutions/apicam.json"),
    ("Cloudynight",       "benchmark/solutions/cloudynight.json"),
    ("Liverpool SkyCam",  "benchmark/solutions/liverpool.json"),
    ("Haleakala",         "benchmark/solutions/haleakala.json"),
]

OUT_CSV = Path("benchmark/results/wcs_roundtrip.csv")

ALT_MIN_DEG = 10.0
ALT_INNER_DEG = 60.0   # within 30° of zenith ("inner" usable region)
N_PIX_GRID = 40        # 40x40 pixel grid inside the image


def _obs_time(inst):
    ts = inst.fit_timestamp
    if ts is None or str(ts).strip() == "":
        # Fall back to a nominal epoch; WCS is an astrometric snapshot.
        return Time("2024-01-01T00:00:00")
    s = str(ts)
    # Strip trailing timezone offset (astropy Time does not accept it).
    for suffix in ("+00:00", "Z"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return Time(s)


def _measure(inst):
    cam = inst.to_camera_model()
    wcs = cam.to_wcs(_obs_time(inst), inst.site_lat, inst.site_lon)

    nx = int(inst.image_width or 0)
    ny = int(inst.image_height or 0)

    # 40x40 pixel grid spanning the image.
    xs = np.linspace(0.0, nx - 1, N_PIX_GRID)
    ys = np.linspace(0.0, ny - 1, N_PIX_GRID)
    XX, YY = np.meshgrid(xs, ys)
    xf = XX.ravel()
    yf = YY.ravel()

    # Native projection: pixel -> (az,alt).
    az_rad, alt_rad = cam.pixel_to_sky(xf, yf)
    alt_deg_all = np.degrees(alt_rad)
    valid = (np.isfinite(az_rad) & np.isfinite(alt_rad)
             & (alt_deg_all >= ALT_MIN_DEG))
    if not np.any(valid):
        return np.array([]), np.array([]), 0, 0

    xf, yf = xf[valid], yf[valid]
    az_rad = az_rad[valid]
    alt_rad = alt_rad[valid]
    alt_deg = alt_deg_all[valid]

    # (az,alt) -> (RA,Dec) at the fit epoch.
    location = EarthLocation.from_geodetic(
        lon=inst.site_lon * u.deg, lat=inst.site_lat * u.deg)
    altaz = AltAz(obstime=_obs_time(inst), location=location)
    sc_native = SkyCoord(
        az=np.degrees(az_rad) * u.deg,
        alt=np.degrees(alt_rad) * u.deg,
        frame=altaz,
    ).icrs

    # WCS forward: pixel -> (RA,Dec).
    ra_wcs, dec_wcs = wcs.all_pix2world(xf, yf, 0)
    wcs_ok = np.isfinite(ra_wcs) & np.isfinite(dec_wcs)
    if not np.any(wcs_ok):
        return np.array([]), np.array([]), 0, 0
    sc_native = sc_native[wcs_ok]
    alt_deg_ok = alt_deg[wcs_ok]
    sc_wcs = SkyCoord(ra=ra_wcs[wcs_ok] * u.deg,
                      dec=dec_wcs[wcs_ok] * u.deg, frame="icrs")

    # Angular separation (radians).
    sep_rad = sc_native.separation(sc_wcs).radian

    # Convert to pixel units using the central pixel scale (1/f).  The
    # local scale of a fisheye is altitude-dependent; (1/f) is the
    # zenith value and the convention used when quoting astrometric
    # accuracy in this camera model.
    pix_err = sep_rad * inst.focal_length_px
    return pix_err, alt_deg_ok, int(pix_err.size), int(np.sum(
        alt_deg_ok >= ALT_INNER_DEG))


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    header = (f"{'Camera':18s}  {'N>10':>5s}  {'N>60':>5s}"
              f"  {'med10':>6s}  {'p90_10':>6s}"
              f"  {'med60':>6s}  {'p90_60':>6s}  {'max60':>6s}"
              f"  {'f (px)':>7s}  {'k1':>11s}")
    print(header)
    print("-" * len(header))
    for name, path in MODELS:
        p = Path(path)
        if not p.exists():
            print(f"{name:18s}  MISSING: {path}")
            continue
        inst = InstrumentModel.load(str(p))
        err, alt_deg, n_all, n_inner = _measure(inst)
        if err.size == 0:
            print(f"{name:18s}  no valid grid points")
            continue
        med_all = float(np.median(err))
        p90_all = float(np.percentile(err, 90))

        inner = alt_deg >= ALT_INNER_DEG
        med_inner = float(np.median(err[inner])) if np.any(inner) else float("nan")
        p90_inner = float(np.percentile(err[inner], 90)) if np.any(inner) else float("nan")
        max_inner = float(np.max(err[inner])) if np.any(inner) else float("nan")

        f_px = inst.focal_length_px
        k1 = inst.k1
        print(f"{name:18s}  {n_all:>5d}  {n_inner:>5d}"
              f"  {med_all:>6.3f}  {p90_all:>6.3f}"
              f"  {med_inner:>6.3f}  {p90_inner:>6.3f}  {max_inner:>6.3f}"
              f"  {f_px:>7.1f}  {k1:>11.3e}")
        rows.append({
            "camera": name,
            "n_alt_gt10": n_all, "n_alt_gt60": n_inner,
            "median_px_alt_gt10": round(med_all, 4),
            "p90_px_alt_gt10":    round(p90_all, 4),
            "median_px_alt_gt60": round(med_inner, 4),
            "p90_px_alt_gt60":    round(p90_inner, 4),
            "max_px_alt_gt60":    round(max_inner, 4),
            "f_px": round(float(f_px), 2),
            "k1": float(k1),
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
