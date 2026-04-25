#!/usr/bin/env python3
"""Analyze per-frame pointing drift from AllClear fast-solve results.

Processes a directory of APICAM (or other) frames with a known instrument
model, extracts per-frame pointing parameters, and produces:
  1. A CSV file with per-frame deltas and metadata.
  2. Time-series plots of pointing drift parameters.

This is designed to reveal:
  - Nightly thermal cooling curves (alt, cx, cy, f drift over hours).
  - Seasonal flexure trends (month-to-month pointing variation).

Usage:
    # Full-night analysis
    python analyze_pointing_drift.py \
        --frames "benchmark/data/apicam_drift_nightly/*.fits" \
        --model benchmark/solutions/apicam_dark.json \
        --output drift_nightly

    # Seasonal analysis
    python analyze_pointing_drift.py \
        --frames "benchmark/data/apicam_drift_seasonal/*.fits" \
        --model benchmark/solutions/apicam_dark.json \
        --output drift_seasonal
"""

import argparse
import csv
import glob
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

log = logging.getLogger(__name__)

ESO_TAP_URL = "http://archive.eso.org/tap_obs"


def fetch_paranal_weather(date_start, date_end):
    """Fetch Paranal ambient weather data from ESO archive.

    Returns a list of dicts with keys: datetime, hour_ut, temp_2m, temp_30m,
    rhum_30m, wind_speed_10m.  Returns empty list on failure.
    """
    try:
        import pyvo
    except ImportError:
        print("  Warning: pyvo not installed, skipping weather data")
        return []

    try:
        tap = pyvo.dal.TAPService(ESO_TAP_URL)
        result = tap.search(f"""
        SELECT midpoint_date, temp_2m, temp_30m, rhum_30m, wind_speed_10m
        FROM asm.meteo_paranal
        WHERE midpoint_date BETWEEN '{date_start}' AND '{date_end}'
        ORDER BY midpoint_date
        """, maxrec=100000)
        table = result.to_table()
    except Exception as e:
        print(f"  Warning: weather query failed: {e}")
        return []

    weather = []
    for row in table:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(row["midpoint_date"]).replace("Z", "+00:00"))
        weather.append({
            "datetime": dt.replace(tzinfo=None),
            "hour_ut": dt.hour + dt.minute / 60.0 + dt.second / 3600.0,
            "temp_2m": float(row["temp_2m"]) if row["temp_2m"] is not None else None,
            "temp_30m": float(row["temp_30m"]) if row["temp_30m"] is not None else None,
            "rhum_30m": float(row["rhum_30m"]) if row["rhum_30m"] is not None else None,
            "wind_speed_10m": float(row["wind_speed_10m"]) if row["wind_speed_10m"] is not None else None,
        })
    print(f"  Weather: {len(weather)} measurements from ESO meteo_paranal")
    return weather


def save_frame_images(data, result, meta, cat, inst, fpath, output_dir,
                      backend="matplotlib"):
    """Save annotated solved + transmission images for a frame."""
    from allclear.transmission import compute_transmission

    if backend == "inkblot":
        from allclear.plotting_inkblot import plot_frame as _plot_frame
    else:
        from allclear.plotting import plot_frame as _plot_frame

    def _save_annotated_image(data, model, det, cat, output_path,
                              matched_pairs=None, transmission_data=None,
                              obs_time=None, lat=None, lon=None):
        from allclear.matching import match_sources
        if matched_pairs is None:
            cat_az = np.radians(np.asarray(cat["az_deg"], dtype=np.float64))
            cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=np.float64))
            cat_xy = np.column_stack(model.sky_to_pixel(cat_az, cat_alt))
            det_xy = np.column_stack([
                np.asarray(det["x"], dtype=np.float64),
                np.asarray(det["y"], dtype=np.float64),
            ])
            matched_pairs, _ = match_sources(det_xy, cat_xy, max_dist=15.0)
        _plot_frame(
            data, model, det_table=det, cat_table=cat,
            matched_pairs=matched_pairs, show_grid=True,
            transmission_data=transmission_data,
            obs_time=obs_time, lat_deg=lat, lon_deg=lon,
            output_path=output_path,
        )

    use_det = result.guided_det_table if (
        result.guided_det_table is not None and len(result.guided_det_table) > 0
    ) else None

    out_base = output_dir / fpath.stem

    # Solved image with crosshairs
    solved_path = str(out_base) + "_solved.png"
    _save_annotated_image(
        data, result.camera_model, use_det, cat, solved_path,
        matched_pairs=result.matched_pairs,
        obs_time=meta["obs_time"],
        lat=inst.site_lat, lon=inst.site_lon,
    )

    # Transmission overlay
    if result.n_matched >= 3 and use_det is not None:
        ref_zp = inst.photometric_zeropoint or None
        az, alt, trans, zp = compute_transmission(
            use_det, cat, result.matched_pairs, result.camera_model,
            image=data, reference_zeropoint=ref_zp,
        )
        trans_path = str(out_base) + "_transmission.png"
        _save_annotated_image(
            data, result.camera_model, use_det, cat, trans_path,
            matched_pairs=result.matched_pairs,
            transmission_data=(az, alt, trans),
            obs_time=meta["obs_time"],
            lat=inst.site_lat, lon=inst.site_lon,
        )


def load_and_solve_frame(fpath, inst, camera_ref, save_images_dir=None,
                         fallback_model=None):
    """Load a single frame, run fast_solve, return per-frame results.

    Returns (row_dict, solve_result) or (None, None) on failure.
    The solve_result is the full SolveResult object, retained for
    deferred image rendering.
    """
    from allclear.cli import _load_frame
    from allclear.solver import fast_solve
    from astropy.io import fits

    try:
        data, meta, cat, det, _ = _load_frame(
            str(fpath), inst.site_lat, inst.site_lon,
        )
    except (ValueError, Exception) as e:
        log.warning("SKIP %s: %s", fpath.name, e)
        return None, None

    # Mirror if needed (same logic as cli.py cmd_solve)
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

    result = fast_solve(data, det, cat, camera_ref, guided=True,
                        refit_rotation=False,
                        fallback_model=fallback_model)

    if result.n_matched < 5:
        log.warning("LOW MATCHES %s: %d matches", fpath.name, result.n_matched)
        return None, None

    # Save images if requested (immediate mode — used by outlier re-rendering)
    if save_images_dir is not None:
        try:
            save_frame_images(data, result, meta, cat, inst, fpath,
                              save_images_dir)
        except Exception as e:
            log.warning("Image save failed for %s: %s", fpath.name, e)

    m = result.camera_model
    ref = camera_ref

    # Read CCD temperature from FITS header (if available)
    ccd_temp = None
    try:
        hdr = fits.getheader(str(fpath))
        ccd_temp = hdr.get("CCD-TEMP")
        if ccd_temp is not None:
            ccd_temp = float(ccd_temp)
    except Exception:
        pass

    obs_dt = meta["obs_time"].datetime

    row = {
        "filename": fpath.name,
        "datetime": obs_dt.isoformat(),
        "mjd": float(meta["obs_time"].mjd),
        "hour_ut": obs_dt.hour + obs_dt.minute / 60.0 + obs_dt.second / 3600.0,
        # Absolute parameters
        "az0_deg": math.degrees(m.az0),
        "alt0_deg": math.degrees(m.alt0),
        "rho_deg": math.degrees(m.rho),
        "f_px": m.f,
        "cx": m.cx,
        "cy": m.cy,
        # Deltas from reference model
        "daz_deg": math.degrees(m.az0 - ref.az0),
        "dalt_deg": math.degrees(m.alt0 - ref.alt0),
        "drho_deg": math.degrees(m.rho - ref.rho),
        "df_px": m.f - ref.f,
        "dcx": m.cx - ref.cx,
        "dcy": m.cy - ref.cy,
        # Quality
        "n_matched": result.n_matched,
        "n_expected": result.n_expected,
        "rms_px": result.rms_residual,
        "status": result.status,
        # Environment
        "ccd_temp": ccd_temp,
    }
    return row, result


def render_frame_images(fpath, solve_result, inst, output_dir):
    """Re-load a frame and render solved + transmission images.

    Uses the saved SolveResult to avoid re-solving.
    """
    from allclear.cli import _load_frame

    data, meta, cat, det, _ = _load_frame(
        str(fpath), inst.site_lat, inst.site_lon,
    )
    if inst.mirrored:
        data = data[:, ::-1]

    save_frame_images(data, solve_result, meta, cat, inst, fpath, output_dir)


def rematch_and_render(fpath, camera_model, inst, output_dir,
                       backend="matplotlib"):
    """Load a frame, do a quick guided match with a known model, render.

    Used by --render-only to skip the expensive refinement phase.
    """
    from allclear.cli import _load_frame
    from allclear.solver import fast_solve

    data, meta, cat, det, _ = _load_frame(
        str(fpath), inst.site_lat, inst.site_lon,
    )
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

    result = fast_solve(data, det, cat, camera_model,
                        guided=True, refine=False)
    if result.n_matched >= 3:
        save_frame_images(data, result, meta, cat, inst, fpath, output_dir,
                          backend=backend)
    return fpath.name


def _rematch_and_render_wrapper(args_tuple):
    """Top-level wrapper for multiprocessing (must be picklable)."""
    return rematch_and_render(*args_tuple[:4], backend=args_tuple[4] if len(args_tuple) > 4 else "matplotlib")


def _render_frame_wrapper(args_tuple):
    """Top-level wrapper for multiprocessing (must be picklable)."""
    fp, sr, ins, out = args_tuple
    render_frame_images(fp, sr, ins, out)
    return fp.name


def save_csv(results, output_path):
    """Write results list to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV: {output_path} ({len(results)} rows)")


def plot_nightly_drift(results, output_dir, weather=None):
    """Plot pointing drift over a single night (vs. UT hour)."""
    hours = np.array([r["hour_ut"] for r in results])
    sort = np.argsort(hours)
    hours = hours[sort]

    params = {
        "dalt_deg": ("Tilt offset", "deg"),
        "dcx": ("Center X offset", "px"),
        "dcy": ("Center Y offset", "px"),
        "df_px": ("Focal length offset", "px"),
        "drho_deg": ("Roll offset", "deg"),
        "rms_px": ("RMS residual", "px"),
    }

    n_panels = len(params) + (1 if weather else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.5 * n_panels),
                             sharex=True)
    fig.suptitle("APICAM Pointing Drift — Single Night", fontsize=14, y=0.98)

    # Weather panel first (temperature on top for visual context)
    ax_offset = 0
    if weather:
        ax_w = axes[0]
        ax_offset = 1
        w_hours = np.array([w["hour_ut"] for w in weather])
        w_temp = np.array([w["temp_2m"] for w in weather])
        valid = np.isfinite(w_temp)
        ax_w.plot(w_hours[valid], w_temp[valid], "-", color="firebrick",
                  linewidth=1.0, alpha=0.8, label="T (2m)")
        if any(w["temp_30m"] is not None for w in weather):
            w_t30 = np.array([w["temp_30m"] if w["temp_30m"] is not None
                              else np.nan for w in weather])
            v30 = np.isfinite(w_t30)
            ax_w.plot(w_hours[v30], w_t30[v30], "-", color="darkorange",
                      linewidth=1.0, alpha=0.6, label="T (30m)")
        ax_w.set_ylabel("Ambient temp (°C)")
        ax_w.legend(fontsize=8, loc="upper right")
        ax_w.grid(True, alpha=0.3)
        # Annotate temp drop
        if np.sum(valid) > 1:
            t_range = np.nanmax(w_temp[valid]) - np.nanmin(w_temp[valid])
            ax_w.annotate(f"range: {t_range:.1f}°C",
                          xy=(0.98, 0.05), xycoords="axes fraction",
                          ha="right", va="bottom", fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="lightyellow", alpha=0.8))

    for i_ax, (key, (label, unit)) in enumerate(params.items()):
        ax = axes[i_ax + ax_offset]
        vals = np.array([results[i][key] for i in sort])
        ax.plot(hours, vals, "o-", markersize=3, linewidth=0.8)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)

        # Annotate total drift
        if key != "rms_px" and len(vals) > 1:
            drift = vals[-1] - vals[0]
            ax.annotate(f"drift: {drift:+.3f} {unit}",
                        xy=(0.98, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="wheat", alpha=0.8))

    axes[-1].set_xlabel("UT hour")
    fig.tight_layout()
    out = output_dir / "drift_nightly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")

    # Also plot n_matched over time
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    n_matched = np.array([results[i]["n_matched"] for i in sort])
    ax2.plot(hours, n_matched, "o-", markersize=3, color="green")
    ax2.set_xlabel("UT hour")
    ax2.set_ylabel("Stars matched")
    ax2.set_title("Match count over night")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = output_dir / "matches_nightly.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot: {out2}")


def plot_seasonal_drift(results, output_dir, weather=None):
    """Plot pointing drift over months/year (vs. MJD or date)."""
    from datetime import datetime

    dates = np.array([datetime.fromisoformat(r["datetime"]) for r in results])
    sort = np.argsort(dates)
    dates = dates[sort]

    params = {
        "dalt_deg": ("Tilt offset", "deg"),
        "dcx": ("Center X offset", "px"),
        "dcy": ("Center Y offset", "px"),
        "df_px": ("Focal length offset", "px"),
        "rms_px": ("RMS residual", "px"),
        "n_matched": ("Stars matched", "count"),
    }

    n_panels = len(params) + (1 if weather else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels),
                             sharex=True)
    fig.suptitle("APICAM Pointing Drift — Seasonal", fontsize=14, y=0.98)

    ax_offset = 0
    if weather:
        ax_w = axes[0]
        ax_offset = 1
        w_dates = np.array([w["datetime"] for w in weather])
        w_temp = np.array([w["temp_2m"] if w["temp_2m"] is not None
                           else np.nan for w in weather])
        valid = np.isfinite(w_temp)
        ax_w.plot(w_dates[valid], w_temp[valid], ".", color="firebrick",
                  markersize=2, alpha=0.5)
        ax_w.set_ylabel("Ambient temp (°C)")
        ax_w.grid(True, alpha=0.3)
        # Running median if enough points
        if np.sum(valid) > 10:
            from scipy.ndimage import median_filter
            window = max(5, np.sum(valid) // 20)
            smoothed = median_filter(w_temp[valid], size=window)
            ax_w.plot(w_dates[valid], smoothed, "-", color="darkred",
                      linewidth=1.5, alpha=0.7, label=f"median (w={window})")
            ax_w.legend(fontsize=8)

    for i_ax, (key, (label, unit)) in enumerate(params.items()):
        ax = axes[i_ax + ax_offset]
        vals = np.array([results[i][key] for i in sort])
        colors = ["green" if results[i]["status"] == "ok" else "red"
                  for i in sort]
        ax.scatter(dates, vals, c=colors, s=10, alpha=0.7)
        ax.set_ylabel(f"{label} ({unit})")
        ax.grid(True, alpha=0.3)

        # Running median
        if len(vals) > 10:
            window = max(5, len(vals) // 20)
            from scipy.ndimage import median_filter
            smoothed = median_filter(vals, size=window)
            ax.plot(dates, smoothed, "k-", linewidth=1.5, alpha=0.5,
                    label=f"median (w={window})")
            ax.legend(fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date")

    fig.tight_layout()
    out = output_dir / "drift_seasonal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out}")


def detect_mode(results):
    """Auto-detect seasonal vs nightly from the data span."""
    if len(results) < 2:
        return "nightly"
    mjds = [r["mjd"] for r in results]
    span_days = max(mjds) - min(mjds)
    return "seasonal" if span_days > 5 else "nightly"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pointing drift from AllClear fast-solve"
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Glob pattern for FITS frames (e.g. 'data/apicam_drift_nightly/*.fits')",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="AllClear instrument model JSON (from instrument-fit)",
    )
    parser.add_argument(
        "--output",
        default="drift_results",
        help="Output directory for CSV and plots (default: drift_results)",
    )
    parser.add_argument(
        "--mode",
        choices=["seasonal", "nightly", "auto"],
        default="auto",
        help="Plot style: seasonal (months), nightly (hours), or auto-detect",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-frame progress",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Skip fetching ESO Paranal weather data",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save solved + transmission images for each frame",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save images every N frames (default: 1 = all frames)",
    )
    parser.add_argument(
        "--save-outliers",
        action="store_true",
        help="Retroactively save images for outlier frames (low matches, "
             "high RMS, or non-ok status) after solving all frames",
    )
    parser.add_argument(
        "--outlier-rms-factor",
        type=float,
        default=2.0,
        help="Save images for frames with RMS > median + factor*MAD (default: 2.0)",
    )
    parser.add_argument(
        "--outlier-match-factor",
        type=float,
        default=2.0,
        help="Save images for frames with matches < median - factor*MAD (default: 2.0)",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel workers for image rendering (default: 1). "
             "Set to number of CPU cores for fastest rendering.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip solving — load models from CSV and render images only. "
             "Requires a previous solve run (pointing_drift.csv).",
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "inkblot"],
        default="matplotlib",
        help="Plotting backend (default: matplotlib).",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load instrument model
    from allclear.instrument import InstrumentModel
    inst = InstrumentModel.load(args.model)
    camera_ref = inst.to_camera_model()

    print(f"Reference model: f={camera_ref.f:.1f}px, "
          f"alt0={math.degrees(camera_ref.alt0):.2f}°, "
          f"rho={math.degrees(camera_ref.rho):.2f}°, "
          f"cx={camera_ref.cx:.1f}, cy={camera_ref.cy:.1f}")

    # --render-only: skip solving, load models from CSV, render images
    if args.render_only:
        output_dir = Path(args.output)
        csv_path = output_dir / "pointing_drift.csv"
        if not csv_path.exists():
            print(f"No CSV found at {csv_path} — run solve first.",
                  file=sys.stderr)
            return 1

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Load CSV rows and build per-frame CameraModels
        import csv as csv_mod
        from allclear.projection import CameraModel
        rows = []
        with open(csv_path) as f:
            for r in csv_mod.DictReader(f):
                rows.append(r)
        print(f"Loaded {len(rows)} solved frames from CSV")

        # Resolve frame paths → CSV rows by filename
        frame_paths = sorted(Path(p) for p in glob.glob(args.frames))
        csv_by_name = {r["filename"]: r for r in rows}

        render_args = []
        for fpath in frame_paths:
            r = csv_by_name.get(fpath.name)
            if r is None:
                continue
            model = CameraModel(
                cx=float(r["cx"]), cy=float(r["cy"]),
                az0=math.radians(float(r["az0_deg"])),
                alt0=math.radians(float(r["alt0_deg"])),
                rho=math.radians(float(r["rho_deg"])),
                f=float(r["f_px"]),
                proj_type=camera_ref.proj_type,
                k1=camera_ref.k1, k2=camera_ref.k2,
            )
            render_args.append((fpath, model, inst, images_dir, args.backend))

        n_render = len(render_args)
        n_jobs = min(args.jobs, n_render)
        print(f"Rendering {n_render} frames ({n_jobs} workers) ...",
              flush=True)

        import time as _time
        t0 = _time.monotonic()
        if n_jobs > 1:
            from multiprocessing import Pool
            with Pool(n_jobs) as pool:
                for j, name in enumerate(
                        pool.imap_unordered(
                            _rematch_and_render_wrapper,
                            render_args)):
                    if (j + 1) % 10 == 0 or j == 0:
                        elapsed = _time.monotonic() - t0
                        rate = (j + 1) / max(elapsed, 1)
                        remaining = (n_render - j - 1) / max(rate, 0.01)
                        print(f"  [{j+1}/{n_render}] {name} "
                              f"({remaining:.0f}s remaining)", flush=True)
        else:
            for j, args_t in enumerate(render_args):
                fpath = args_t[0]
                if (j + 1) % 10 == 0 or j == 0:
                    elapsed = _time.monotonic() - t0
                    rate = (j + 1) / max(elapsed, 1)
                    remaining = (n_render - j - 1) / max(rate, 0.01)
                    print(f"  [{j+1}/{n_render}] {fpath.name} "
                          f"({remaining:.0f}s remaining)", flush=True)
                rematch_and_render(*args_t[:4], backend=args_t[4])

        elapsed = _time.monotonic() - t0
        print(f"Done: {n_render} frames in {elapsed:.0f}s "
              f"({elapsed/max(n_render,1):.1f}s/frame)")
        return 0

    # Resolve frames
    frame_paths = sorted(Path(p) for p in glob.glob(args.frames))
    if not frame_paths:
        print(f"No files match: {args.frames}", file=sys.stderr)
        return 1
    print(f"Processing {len(frame_paths)} frames ...")

    # Set up output and image saving
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = None
    if args.save_images:
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving images every {args.save_every} frame(s) to {images_dir}/")

    # Phase 1: Solve all frames (fast — no image rendering)
    results = []
    result_paths = []  # parallel list: frame path for each result
    solve_results = []  # SolveResult objects for deferred rendering
    failed_paths = []  # frames that returned None
    last_good_model = None  # most recent solve with low RMS
    _GOOD_RMS_THRESHOLD = 4.0  # update last_good_model only from clean solves
    for i, fpath in enumerate(frame_paths):
        if args.verbose or (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_paths)}] {fpath.name} ...", flush=True)
        row, solve_result = load_and_solve_frame(
            fpath, inst, camera_ref,
            fallback_model=last_good_model)
        if row is not None:
            results.append(row)
            result_paths.append(fpath)
            solve_results.append(solve_result)
            # Update fallback model from good solves only
            if row["rms_px"] < _GOOD_RMS_THRESHOLD and row["n_matched"] >= 100:
                from allclear.projection import CameraModel
                last_good_model = CameraModel(
                    cx=row["cx"], cy=row["cy"],
                    az0=math.radians(row["az0_deg"]),
                    alt0=math.radians(row["alt0_deg"]),
                    rho=math.radians(row["rho_deg"]),
                    f=row["f_px"],
                    proj_type=camera_ref.proj_type,
                    k1=camera_ref.k1, k2=camera_ref.k2,
                )
            if args.verbose:
                print(f"    -> {row['n_matched']} matches, RMS={row['rms_px']:.2f}, "
                      f"dalt={row['dalt_deg']:+.3f}°, df={row['df_px']:+.1f}px")
        else:
            failed_paths.append(fpath)

    print(f"\nSuccessfully solved {len(results)}/{len(frame_paths)} frames",
          flush=True)
    if failed_paths:
        print(f"  Failed frames: {len(failed_paths)}")

    if not results:
        print("No successful solves — nothing to plot.", file=sys.stderr)
        return 1

    # Save CSV and plots FIRST (before slow outlier phase)
    weather = []
    if not args.no_weather:
        from datetime import datetime
        datetimes = [datetime.fromisoformat(r["datetime"]) for r in results]
        date_min = min(datetimes)
        date_max = max(datetimes)
        span_days = (date_max - date_min).total_seconds() / 86400
        if span_days > 5:
            print("Fetching weather data for each frame timestamp ...")
            all_weather = []
            for r in results:
                dt = datetime.fromisoformat(r["datetime"])
                t0 = dt.strftime("%Y-%m-%dT%H:%M:%S")
                from datetime import timedelta
                t1 = (dt + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
                w = fetch_paranal_weather(t0, t1)
                if w:
                    all_weather.append(w[0])
            weather = all_weather
        else:
            t0 = date_min.strftime("%Y-%m-%dT%H:%M:%S")
            t1 = date_max.strftime("%Y-%m-%dT%H:%M:%S")
            weather = fetch_paranal_weather(t0, t1)

    if weather:
        from datetime import datetime
        for r in results:
            r_dt = datetime.fromisoformat(r["datetime"])
            nearest = min(weather, key=lambda w: abs((w["datetime"] - r_dt).total_seconds()))
            r["ambient_temp_C"] = nearest.get("temp_2m")
            r["wind_speed_ms"] = nearest.get("wind_speed_10m")
    else:
        for r in results:
            r["ambient_temp_C"] = None
            r["wind_speed_ms"] = None

    csv_path = output_dir / "pointing_drift.csv"
    save_csv(results, csv_path)

    mode = args.mode if args.mode != "auto" else detect_mode(results)
    print(f"Plotting mode: {mode}")

    if mode == "nightly":
        plot_nightly_drift(results, output_dir, weather=weather)
    else:
        plot_seasonal_drift(results, output_dir, weather=weather)

    # Print summary statistics
    print("\n--- Drift Summary ---")
    for key, label in [("dalt_deg", "Tilt (alt0)"),
                        ("dcx", "Center X"),
                        ("dcy", "Center Y"),
                        ("df_px", "Focal length")]:
        vals = [r[key] for r in results]
        print(f"  {label:15s}: mean={np.mean(vals):+.4f}, "
              f"std={np.std(vals):.4f}, "
              f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    rms_vals = [r["rms_px"] for r in results]
    print(f"  {'RMS residual':15s}: median={np.median(rms_vals):.2f}px, "
          f"range=[{min(rms_vals):.2f}, {max(rms_vals):.2f}]px")

    print(f"\nResults in: {output_dir}/", flush=True)

    # Phase 2: Batch render images (deferred from solve phase)
    if images_dir and solve_results:
        render_items = [
            (fpath, sr) for j, (fpath, sr)
            in enumerate(zip(result_paths, solve_results))
            if j % args.save_every == 0
        ]
        n_render = len(render_items)
        n_jobs = min(args.jobs, n_render)
        print(f"\nRendering {n_render} frame images to {images_dir}/ "
              f"({n_jobs} workers) ...", flush=True)
        import time as _time
        t0_render = _time.monotonic()

        if n_jobs > 1:
            from multiprocessing import Pool
            # Wrap args for starmap
            render_args = [
                (fpath, sr, inst, images_dir)
                for fpath, sr in render_items
            ]
            with Pool(n_jobs) as pool:
                for j, name in enumerate(
                        pool.imap_unordered(_render_frame_wrapper, render_args)):
                    if (j + 1) % 10 == 0 or j == 0:
                        elapsed = _time.monotonic() - t0_render
                        rate = (j + 1) / max(elapsed, 1)
                        remaining = (n_render - j - 1) / max(rate, 0.01)
                        print(f"  [{j+1}/{n_render}] {name} "
                              f"({remaining:.0f}s remaining)", flush=True)
        else:
            for j, (fpath, sr) in enumerate(render_items):
                if (j + 1) % 10 == 0 or j == 0:
                    elapsed = _time.monotonic() - t0_render
                    rate = (j + 1) / max(elapsed, 1)
                    remaining = (n_render - j - 1) / max(rate, 0.01)
                    print(f"  [{j+1}/{n_render}] {fpath.name} "
                          f"({remaining:.0f}s remaining)", flush=True)
                try:
                    render_frame_images(fpath, sr, inst, images_dir)
                except Exception as e:
                    log.warning("Render failed for %s: %s", fpath.name, e)

        elapsed = _time.monotonic() - t0_render
        print(f"  Rendered {n_render} images in {elapsed:.0f}s "
              f"({elapsed/max(n_render,1):.1f}s/frame)")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
