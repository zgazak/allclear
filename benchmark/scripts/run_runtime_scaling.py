"""Runtime scaling of blind and fast solves across benchmark cameras.

For each of the four benchmark cameras, runs:

    1. ``instrument_fit_pipeline`` from scratch (blind, timed)
    2. ``fast_solve`` against the solved model (timed)

and records ``(n_matched, rms_px, elapsed_s, n_det, n_cat, n_pixels)``.
Writes ``benchmark/results/runtime_scaling.csv`` and a two-panel
PNG showing runtime vs. image pixel count and vs. catalog size.

Run:
    uv run python benchmark/scripts/run_runtime_scaling.py
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allclear.cli import _load_frame
from allclear.detection import detect_stars
from allclear.instrument import InstrumentModel
from allclear.matching import match_sources
from allclear.solver import fast_solve
from allclear.strategies import instrument_fit_pipeline


OUT_DIR = Path("benchmark/results")
OUT_CSV = OUT_DIR / "runtime_scaling.csv"
OUT_PNG = OUT_DIR / "runtime_scaling.png"


CAMERAS = [
    {
        "name": "Liverpool",
        "frame": "benchmark/data/liverpool_skycam/a_e_20240414_268_1_1_1.fits",
        "lat": 28.7624, "lon": -17.8792,
    },
    {
        "name": "Cloudynight",
        "frame": "benchmark/data/cloudynight/000.fits",
        "lat": 32.4420, "lon": -110.7880,
    },
    {
        "name": "Haleakala",
        "frame": "benchmark/data/haleakala/2023_11_19__00_01_19.fits",
        "lat": 20.7458, "lon": -156.4317,
    },
    {
        "name": "APICAM",
        "frame": ("benchmark/data/apicam_drift_seasonal/"
                  "APICAM.2019-06-06T00:59:45.000.fits"),
        "lat": -24.6272, "lon": -70.4048,
    },
]


@dataclass
class TimingRow:
    camera: str
    mode: str
    n_pixels: int
    n_det: int
    n_cat: int
    n_matched: int
    rms_px: float
    elapsed_s: float


def _time_blind(data, det, cat, initial_f):
    t0 = time.time()
    model, n_matched, rms, _ = instrument_fit_pipeline(
        data, det, cat, initial_f=initial_f, verbose=False,
    )
    return time.time() - t0, int(n_matched), float(rms), model


def _time_solve(data, det, cat, camera):
    t0 = time.time()
    result = fast_solve(data, det, cat, camera)
    return (time.time() - t0,
            int(result.n_matched),
            float(result.rms_residual))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for cam in CAMERAS:
        print(f"\n=== {cam['name']} ===")
        frame = Path(cam["frame"])
        if not frame.exists():
            print(f"  missing frame: {frame}")
            continue

        data, meta, cat, det, initial_f = _load_frame(
            str(frame), cam["lat"], cam["lon"])
        ny, nx = data.shape
        n_pixels = nx * ny
        print(f"  image: {data.shape}, catalog: {len(cat)}, "
              f"detections: {len(det)}")

        # Blind
        print(f"  [blind] running...", flush=True)
        blind_t, blind_n, blind_rms, model = _time_blind(
            data, det, cat, initial_f)
        print(f"  [blind] {blind_n} matches, RMS {blind_rms:.2f}px, "
              f"{blind_t:.1f}s")
        rows.append(TimingRow(
            camera=cam["name"], mode="blind",
            n_pixels=n_pixels, n_det=len(det), n_cat=len(cat),
            n_matched=blind_n, rms_px=blind_rms, elapsed_s=blind_t,
        ))

        # Solve using the just-fitted model (same frame; apples-to-apples
        # for how fast a subsequent-frame solve would be).  Re-detect on
        # the same data (and mirror if model says so).
        data_solve = data
        det_solve = det
        if hasattr(model, "mirrored") and getattr(model, "mirrored", False):
            data_solve = data[:, ::-1]
            det_solve = det.copy()
            det_solve["x"] = (data_solve.shape[1] - 1
                              - np.asarray(det_solve["x"],
                                           dtype=np.float64))

        print(f"  [solve] running...", flush=True)
        solve_t, solve_n, solve_rms = _time_solve(
            data_solve, det_solve, cat, model)
        print(f"  [solve] {solve_n} matches, RMS {solve_rms:.2f}px, "
              f"{solve_t:.2f}s")
        rows.append(TimingRow(
            camera=cam["name"], mode="solve",
            n_pixels=n_pixels, n_det=len(det), n_cat=len(cat),
            n_matched=solve_n, rms_px=solve_rms, elapsed_s=solve_t,
        ))

    # CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["camera", "mode", "n_pixels", "n_det", "n_cat",
                    "n_matched", "rms_px", "elapsed_s"])
        for r in rows:
            w.writerow([r.camera, r.mode, r.n_pixels, r.n_det, r.n_cat,
                        r.n_matched, round(r.rms_px, 3),
                        round(r.elapsed_s, 2)])
    print(f"\nWrote: {OUT_CSV}")

    # Plot runtime vs image size (log-log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    blind = [r for r in rows if r.mode == "blind"]
    solve = [r for r in rows if r.mode == "solve"]

    for ax, mode_rows, title in [
        (ax1, blind, "Blind instrument-fit"),
        (ax2, solve, "Fast solve"),
    ]:
        for r in mode_rows:
            ax.scatter(r.n_pixels, r.elapsed_s, s=70, zorder=3)
            ax.annotate(r.camera, (r.n_pixels, r.elapsed_s),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("image pixels")
        ax.set_ylabel("elapsed (s)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Runtime vs. image size across benchmark cameras",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")

    print("\nResults:")
    print(f"{'camera':12s}  {'mode':6s}  {'matched':>8s}  "
          f"{'RMS px':>7s}  {'time s':>7s}")
    for r in rows:
        print(f"{r.camera:12s}  {r.mode:6s}  {r.n_matched:>8d}  "
              f"{r.rms_px:>7.2f}  {r.elapsed_s:>7.2f}")


if __name__ == "__main__":
    main()
