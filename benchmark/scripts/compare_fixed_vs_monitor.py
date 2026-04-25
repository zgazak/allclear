#!/usr/bin/env python3
"""Compare fixed-model solve vs operational monitor on APICAM seasonal data.

Runs the same 260 frames two ways:
  1. Fixed: every frame solved against the original Feb 10 model (already done)
  2. Monitor: each good solve updates the model for the next frame

Produces:
  - CSV with both runs' results side by side
  - Comparison plot: RMS and matches over time for both modes
  - Images for: 5% of good frames + all bad/spurious frames

Usage:
    python compare_fixed_vs_monitor.py \
        --frames "benchmark/data/apicam_drift_seasonal/*.fits" \
        --model benchmark/solutions/apicam.json \
        --fixed-csv benchmark/results/apicam_seasonal/pointing_drift.csv \
        --output benchmark/results/apicam_comparison
"""

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)


def run_monitor(frame_paths, model_path, output_dir, image_dir):
    """Run all frames through the operational monitor."""
    from allclear.monitor import OperationalMonitor, QualityThresholds

    thresholds = QualityThresholds(
        min_matched_good=800,
        max_rms_good=4.0,
        min_matched_bad=200,
    )

    mon = OperationalMonitor(
        model_path, thresholds=thresholds, output_dir=output_dir)

    # Pre-select 5% random sample for "good" images (deterministic seed)
    random.seed(42)
    n_frames = len(frame_paths)
    good_sample_indices = set(random.sample(range(n_frames),
                                            max(1, n_frames // 20)))
    # Always include first, last, and some evenly-spaced frames
    good_sample_indices.add(0)
    good_sample_indices.add(n_frames - 1)
    for k in range(0, n_frames, n_frames // 10):
        good_sample_indices.add(k)

    image_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, fpath in enumerate(frame_paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_paths)}] {fpath.name} ...", flush=True)

        # Decide whether to save image for this frame DURING processing
        # (so it uses the model-at-this-time, not the end-of-run model)
        save_dir = None
        if i in good_sample_indices:
            save_dir = image_dir  # always save sampled frames

        r = mon.process_frame(fpath, save_image_dir=save_dir)
        results.append(r)

        # Also save image for bad/spurious frames (decided after solve)
        if save_dir is None and (r.quality in ("bad", "refit_needed")
                                 or r.n_matched < 500 or r.rms > 3.5):
            mon._save_image_from_last(fpath, image_dir)

        if r.quality != "bad":
            print(f"    -> {r.n_matched} matches, RMS={r.rms:.2f}, "
                  f"q={r.quality}, upd={r.model_updated}", flush=True)
        else:
            print(f"    -> {r.quality}: {r.notes}", flush=True)

    summary = mon.get_summary()
    return results, summary


def load_fixed_results(csv_path):
    """Load the fixed-model results from the earlier run."""
    rows = list(csv.DictReader(open(csv_path)))
    return rows


def plot_comparison(fixed_rows, monitor_results, output_dir):
    """Plot side-by-side comparison of fixed vs monitor modes."""
    # Parse fixed results
    f_dates = [datetime.fromisoformat(r["datetime"]) for r in fixed_rows]
    f_rms = [float(r["rms_px"]) for r in fixed_rows]
    f_match = [int(r["n_matched"]) for r in fixed_rows]
    f_dalt = [float(r["dalt_deg"]) for r in fixed_rows]
    f_dcy = [float(r["dcy"]) for r in fixed_rows]

    # Parse monitor results
    m_dates = [datetime.fromisoformat(r.timestamp) for r in monitor_results
               if r.timestamp]
    m_rms = [r.rms for r in monitor_results if r.timestamp]
    m_match = [r.n_matched for r in monitor_results if r.timestamp]
    m_dalt = [r.dalt_deg for r in monitor_results if r.timestamp]
    m_dcy = [r.dcy for r in monitor_results if r.timestamp]
    m_quality = [r.quality for r in monitor_results if r.timestamp]
    m_updated = [r.model_updated for r in monitor_results if r.timestamp]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("APICAM 2019: Fixed Model vs Operational Monitor",
                 fontsize=14, y=0.98)

    # 1. RMS comparison
    ax = axes[0]
    ax.plot(f_dates, f_rms, "o", markersize=3, color="tab:red",
            alpha=0.6, label="Fixed model")
    ax.plot(m_dates, m_rms, "o", markersize=3, color="tab:blue",
            alpha=0.6, label="Monitor (auto-update)")
    ax.set_ylabel("RMS residual (px)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Running medians
    from scipy.ndimage import median_filter
    if len(f_rms) > 10:
        w = max(5, len(f_rms) // 20)
        ax.plot(f_dates, median_filter(f_rms, size=w), "-",
                color="darkred", linewidth=2, alpha=0.7)
        ax.plot(m_dates, median_filter(m_rms, size=w), "-",
                color="darkblue", linewidth=2, alpha=0.7)

    # 2. Match count comparison
    ax = axes[1]
    ax.plot(f_dates, f_match, "o", markersize=3, color="tab:red",
            alpha=0.6, label="Fixed")
    ax.plot(m_dates, m_match, "o", markersize=3, color="tab:blue",
            alpha=0.6, label="Monitor")
    ax.set_ylabel("Stars matched")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Tilt offset (both vs original model)
    ax = axes[2]
    ax.plot(f_dates, f_dalt, "o", markersize=3, color="tab:red",
            alpha=0.6, label="Fixed dalt")
    ax.plot(m_dates, m_dalt, "o", markersize=3, color="tab:blue",
            alpha=0.6, label="Monitor dalt")
    ax.set_ylabel("Tilt offset (°)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Monitor quality timeline
    ax = axes[3]
    q_colors = {"good": "green", "marginal": "orange",
                "bad": "red", "refit_needed": "purple"}
    for i, (d, q) in enumerate(zip(m_dates, m_quality)):
        ax.bar(d, 1, width=1.5, color=q_colors.get(q, "gray"), alpha=0.7)
    # Mark model updates
    upd_dates = [d for d, u in zip(m_dates, m_updated) if u]
    ax.plot(upd_dates, [0.5] * len(upd_dates), "|", color="black",
            markersize=8, markeredgewidth=1.5, label="Model updated")
    ax.set_ylabel("Quality")
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper right")
    # Legend for colors
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=l) for l, c in q_colors.items()]
    ax.legend(handles=patches, fontsize=8, ncol=4, loc="upper left")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[-1].set_xlabel("Date")

    fig.tight_layout()
    out = output_dir / "fixed_vs_monitor.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare fixed-model vs operational monitor on APICAM data"
    )
    parser.add_argument("--frames", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixed-csv", required=True,
                        help="CSV from the fixed-model run")
    parser.add_argument("--output", default="benchmark/results/apicam_comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import glob
    frame_paths = sorted(Path(p) for p in glob.glob(args.frames))
    if not frame_paths:
        print(f"No files match: {args.frames}", file=sys.stderr)
        return 1
    print(f"Processing {len(frame_paths)} frames with operational monitor ...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"

    # Run monitor mode
    monitor_results, summary = run_monitor(
        frame_paths, args.model, output_dir, image_dir)

    print(f"\n=== Monitor Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Load fixed results for comparison
    fixed_rows = load_fixed_results(args.fixed_csv)

    # Save monitor results CSV
    csv_path = output_dir / "monitor_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "timestamp", "n_matched", "rms", "status",
            "clear_fraction", "dcx", "dcy", "dalt_deg", "drho_deg",
            "df_px", "quality", "model_updated", "notes",
        ])
        writer.writeheader()
        for r in monitor_results:
            writer.writerow({
                "frame": r.frame_path, "timestamp": r.timestamp,
                "n_matched": r.n_matched, "rms": f"{r.rms:.3f}",
                "status": r.status, "clear_fraction": f"{r.clear_fraction:.3f}",
                "dcx": f"{r.dcx:.2f}", "dcy": f"{r.dcy:.2f}",
                "dalt_deg": f"{r.dalt_deg:.4f}",
                "drho_deg": f"{r.drho_deg:.4f}",
                "df_px": f"{r.df_px:.2f}",
                "quality": r.quality,
                "model_updated": r.model_updated,
                "notes": r.notes,
            })
    print(f"  Monitor CSV: {csv_path}")

    # Plot comparison
    plot_comparison(fixed_rows, monitor_results, output_dir)

    # Print head-to-head stats
    f_rms = [float(r["rms_px"]) for r in fixed_rows]
    m_rms = [r.rms for r in monitor_results if r.rms > 0]
    print(f"\n=== Head-to-Head ===")
    print(f"  Fixed:   RMS median={np.median(f_rms):.2f}, "
          f"range=[{min(f_rms):.2f}, {max(f_rms):.2f}]")
    print(f"  Monitor: RMS median={np.median(m_rms):.2f}, "
          f"range=[{min(m_rms):.2f}, {max(m_rms):.2f}]")

    # Late-year comparison (after July shift)
    f_late = [float(r["rms_px"]) for r in fixed_rows
              if r["datetime"] > "2019-08-01"]
    m_late = [r.rms for r in monitor_results
              if r.timestamp > "2019-08-01" and r.rms > 0]
    if f_late and m_late:
        print(f"  Fixed  (Aug-Nov): RMS median={np.median(f_late):.2f}")
        print(f"  Monitor(Aug-Nov): RMS median={np.median(m_late):.2f}")

    print(f"\nResults in: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
