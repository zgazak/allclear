#!/usr/bin/env python3
"""Run fully blind instrument_fit_pipeline on each APICAM seasonal frame.

Each frame is solved independently from scratch (no reference model).
Results go to benchmark/results/apicam_seasonal_blind/.

Usage:
    python run_seasonal_blind.py \
        --frames "benchmark/data/apicam_drift_seasonal/*.fits" \
        --lat -24.6272 --lon -70.4048 \
        --output benchmark/results/apicam_seasonal_blind \
        --jobs 12

    # Compare against fixed-model results:
    python run_seasonal_blind.py \
        --frames "benchmark/data/apicam_drift_seasonal/*.fits" \
        --lat -24.6272 --lon -70.4048 \
        --output benchmark/results/apicam_seasonal_blind \
        --jobs 12 \
        --fixed-csv benchmark/results/apicam_seasonal/pointing_drift.csv
"""

import argparse
import csv
import math
import multiprocessing
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def solve_one_frame(args_tuple):
    """Run blind instrument-fit on a single frame. Worker function for Pool.

    Returns a dict with results, or a dict with status='failed'/'skipped'.
    """
    fpath, lat, lon, output_dir, save_images = args_tuple
    stem = fpath.stem

    # Check if already solved (resumable)
    model_path = output_dir / f"{stem}_model.json"
    if model_path.exists():
        # Load existing result for CSV
        try:
            from allclear.instrument import InstrumentModel
            inst = InstrumentModel.load(str(model_path))
            return {
                "filename": fpath.name,
                "datetime": inst.fit_timestamp,
                "f": inst.focal_length_px,
                "cx": inst.center_x,
                "cy": inst.center_y,
                "az0_deg": inst.az0_deg,
                "alt0_deg": inst.alt0_deg,
                "rho_deg": inst.roll_deg,
                "k1": inst.k1,
                "k2": inst.k2,
                "n_matched": inst.n_stars_matched,
                "rms": inst.rms_residual_px,
                "mirrored": inst.mirrored,
                "projection_type": inst.projection,
                "status": "cached",
            }
        except Exception:
            pass  # Re-solve if JSON is corrupt

    # Load frame
    try:
        from allclear.cli import _load_frame
        data, meta, cat, det, initial_f = _load_frame(str(fpath), lat, lon)
    except Exception as e:
        print(f"  SKIP {fpath.name}: {e}", flush=True)
        return {
            "filename": fpath.name, "datetime": "", "status": f"load_error: {e}",
            "f": 0, "cx": 0, "cy": 0, "az0_deg": 0, "alt0_deg": 0,
            "rho_deg": 0, "k1": 0, "k2": 0, "n_matched": 0, "rms": 0,
            "mirrored": False, "projection_type": "",
        }

    obs_dt = meta["obs_time"].iso

    # Run blind instrument-fit
    try:
        from allclear.strategies import instrument_fit_pipeline
        from allclear.instrument import InstrumentModel

        model, n_matched, rms, diag = instrument_fit_pipeline(
            data, det, cat, initial_f=initial_f, verbose=False, meta=meta,
        )
    except Exception as e:
        print(f"  FAIL {fpath.name}: {e}", flush=True)
        traceback.print_exc()
        return {
            "filename": fpath.name, "datetime": obs_dt, "status": f"error: {e}",
            "f": 0, "cx": 0, "cy": 0, "az0_deg": 0, "alt0_deg": 0,
            "rho_deg": 0, "k1": 0, "k2": 0, "n_matched": 0, "rms": 0,
            "mirrored": False, "projection_type": "",
        }

    # Quality gate
    ny, nx = data.shape
    max_rms = max(6.0, min(nx, ny) * 0.004)
    is_mirrored = diag.get("mirrored", False)

    if n_matched < 30 or rms > max_rms:
        reason = []
        if n_matched < 30:
            reason.append(f"n={n_matched}<30")
        if rms > max_rms:
            reason.append(f"rms={rms:.1f}>{max_rms:.1f}")
        status = f"failed: {', '.join(reason)}"
        print(f"  FAIL {fpath.name}: {status}", flush=True)
        return {
            "filename": fpath.name, "datetime": obs_dt, "status": status,
            "f": model.f, "cx": model.cx, "cy": model.cy,
            "az0_deg": math.degrees(model.az0),
            "alt0_deg": math.degrees(model.alt0),
            "rho_deg": math.degrees(model.rho),
            "k1": model.k1, "k2": model.k2,
            "n_matched": n_matched, "rms": rms,
            "mirrored": is_mirrored,
            "projection_type": model.proj_type.value,
        }

    # Save model JSON
    inst = InstrumentModel.from_camera_model(
        model,
        site_lat=lat, site_lon=lon,
        image_width=nx, image_height=ny,
        mirrored=is_mirrored,
        n_stars_matched=n_matched,
        n_stars_expected=len(cat),
        rms_residual_px=rms,
        fit_timestamp=obs_dt,
        frame_used=fpath.name,
    )
    inst.save(str(model_path))

    # Save diagnostic PNG
    if save_images:
        try:
            from allclear.cli import _save_diagnostic_plot
            plot_data = data
            if is_mirrored:
                plot_data = plot_data[:, ::-1]
            png_path = output_dir / f"{stem}.png"
            _save_diagnostic_plot(
                plot_data, model, det, cat, n_matched, rms, diag,
                str(png_path), meta=meta,
            )
        except Exception as e:
            print(f"  WARN {fpath.name}: plot failed: {e}", flush=True)

    print(f"  OK {fpath.name}: {n_matched} matches, RMS={rms:.2f}px"
          f"{' (mirrored)' if is_mirrored else ''}", flush=True)

    return {
        "filename": fpath.name,
        "datetime": obs_dt,
        "f": model.f,
        "cx": model.cx,
        "cy": model.cy,
        "az0_deg": math.degrees(model.az0),
        "alt0_deg": math.degrees(model.alt0),
        "rho_deg": math.degrees(model.rho),
        "k1": model.k1,
        "k2": model.k2,
        "n_matched": n_matched,
        "rms": rms,
        "mirrored": is_mirrored,
        "projection_type": model.proj_type.value,
        "status": "ok",
    }


def write_csv(rows, output_path):
    """Write results to CSV."""
    cols = [
        "filename", "datetime", "n_matched", "rms", "f", "cx", "cy",
        "az0_deg", "alt0_deg", "rho_deg", "k1", "mirrored",
        "projection_type", "status",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["filename"]):
            writer.writerow(row)


def _read_csv_cols(path, cols):
    """Read specific columns from a CSV into a dict of numpy arrays."""
    with open(path) as fh:
        reader = csv.DictReader(fh)
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(row.get(c, ""))
    return data


def _parse_datetimes(date_strings):
    """Parse datetime strings to matplotlib date numbers."""
    from datetime import datetime as dt
    dates = []
    for s in date_strings:
        try:
            # Handle both "2019-01-01 01:05:33.000" and "2019-01-01T01:05:33"
            s_clean = s.replace("T", " ").split(".")[0]
            dates.append(dt.strptime(s_clean, "%Y-%m-%d %H:%M:%S"))
        except (ValueError, AttributeError):
            dates.append(None)
    return dates


def plot_comparison(blind_csv, fixed_csv, output_dir):
    """Generate comparison plots: blind-solve vs fixed-model results."""
    # Read blind results
    blind_cols = ["filename", "datetime", "n_matched", "rms", "f", "cx",
                  "cy", "rho_deg", "status"]
    blind_raw = _read_csv_cols(blind_csv, blind_cols)

    # Filter to successful
    ok_mask = [s in ("ok", "cached") for s in blind_raw["status"]]
    blind = {}
    for c in blind_cols:
        blind[c] = [v for v, m in zip(blind_raw[c], ok_mask) if m]
    blind_dt = _parse_datetimes(blind["datetime"])

    # Read fixed results
    fixed_cols = ["datetime", "n_matched", "rms_px", "f_px", "cx", "cy",
                  "rho_deg", "status"]
    fixed = _read_csv_cols(fixed_csv, fixed_cols)
    fixed_dt = _parse_datetimes(fixed["datetime"])

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Blind Solve vs Fixed-Model Solve — APICAM Seasonal",
                 fontsize=14)

    params = [
        ("cx", "cx", "Center X (px)"),
        ("cy", "cy", "Center Y (px)"),
        ("f", "f_px", "Focal Length (px)"),
        ("rms", "rms_px", "RMS Residual (px)"),
        ("n_matched", "n_matched", "Stars Matched"),
        ("rho_deg", "rho_deg", "Roll (deg)"),
    ]

    for idx, (bcol, fcol, label) in enumerate(params):
        ax = axes[idx // 2, idx % 2]

        # Fixed-model
        if fcol in fixed:
            fvals = []
            fdts = []
            for d, v in zip(fixed_dt, fixed[fcol]):
                try:
                    fvals.append(float(v))
                    fdts.append(d)
                except (ValueError, TypeError):
                    pass
            if fdts:
                ax.scatter(fdts, fvals, s=8, alpha=0.5, color="C0",
                           label="Fixed model")

        # Blind solve
        if bcol in blind:
            bvals = []
            bdts = []
            for d, v in zip(blind_dt, blind[bcol]):
                try:
                    bvals.append(float(v))
                    bdts.append(d)
                except (ValueError, TypeError):
                    pass
            if bdts:
                ax.scatter(bdts, bvals, s=12, alpha=0.7, color="C1",
                           label="Blind solve")

        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    out_path = output_dir / "blind_vs_fixed_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot: {out_path}")

    n_total = len(blind_raw["status"])
    n_ok = sum(ok_mask)
    n_fail = n_total - n_ok
    print(f"\nBlind solve summary: {n_ok}/{n_total} succeeded "
          f"({n_fail} failed, {100*n_ok/n_total:.0f}% success rate)")


def main():
    parser = argparse.ArgumentParser(
        description="Run blind instrument-fit on APICAM seasonal frames")
    parser.add_argument("--frames", type=str, required=True,
                        help="Glob pattern for FITS frames")
    parser.add_argument("--lat", type=float, default=-24.6272,
                        help="Site latitude (default: APICAM/Paranal)")
    parser.add_argument("--lon", type=float, default=-70.4048,
                        help="Site longitude (default: APICAM/Paranal)")
    parser.add_argument("--output", type=str,
                        default="benchmark/results/apicam_seasonal_blind",
                        help="Output directory")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--fixed-csv", type=str, default=None,
                        help="Path to fixed-model CSV for comparison plots")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving diagnostic PNGs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N frames (for testing)")
    args = parser.parse_args()

    # Resolve frames
    from allclear.cli import _resolve_frames
    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No frames found: {args.frames}", file=sys.stderr)
        return 1

    if args.limit:
        frames = frames[:args.limit]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Blind instrument-fit: {len(frames)} frames, {args.jobs} workers")
    print(f"Output: {output_dir}")

    save_images = not args.no_images
    work_items = [
        (f, args.lat, args.lon, output_dir, save_images)
        for f in frames
    ]

    # Run in parallel or serial
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(solve_one_frame, work_items)
    else:
        results = [solve_one_frame(item) for item in work_items]

    # Write summary CSV
    csv_path = output_dir / "blind_solve_results.csv"
    write_csv(results, csv_path)
    print(f"\nCSV written: {csv_path}")

    # Summary stats
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    failed = [r for r in results if r["status"] not in ("ok", "cached")]
    print(f"Succeeded: {len(ok)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed[:10]:
            print(f"  {r['filename']}: {r['status']}")

    # Comparison plots
    fixed_csv = args.fixed_csv
    if fixed_csv is None:
        default_fixed = Path("benchmark/results/apicam_seasonal/pointing_drift.csv")
        if default_fixed.exists():
            fixed_csv = str(default_fixed)

    if fixed_csv and Path(fixed_csv).exists() and len(ok) > 5:
        plot_comparison(csv_path, fixed_csv, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
