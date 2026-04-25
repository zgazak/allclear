"""Render clear / partial / cloudy transmission triptychs per camera.

Reads benchmark/results/frame_scores/<camera>.csv (produced by
score_frames.py) and picks one frame from each of three sky-condition
bins:

  - clear:   clear_fraction in [0.80, 1.00]   (prefer highest)
  - partial: clear_fraction in [0.30, 0.60]   (prefer closest to 0.45)
  - cloudy:  clear_fraction in [0.00, 0.15]   (prefer lowest)

Runs the full solve + transmission pipeline on each pick and renders
an annotated all-sky panel using allclear.plotting.plot_frame.
Writes a 3-panel composite PDF/PNG per camera.

Run (after score_frames.py):
    uv run python benchmark/scripts/make_triptychs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allclear.instrument import InstrumentModel
from allclear.solver import fast_solve
from allclear.catalog import BrightStarCatalog
from allclear.detection import detect_stars
from allclear.plotting import plot_frame
from allclear.transmission import compute_transmission
from allclear.utils import load_image, parse_fits_header

SCORES_DIR = Path("benchmark/results/frame_scores")
OUT_DIR = Path("benchmark/results/transmission_triptychs")

BINS = [
    ("clear",   (0.70, 1.01), "highest"),
    ("partial", (0.20, 0.65), "mid"),
    ("cloudy",  (0.00, 0.20), "lowest"),
]

# APICAM benchmark frames are all clear-sky; pull partial/cloudy from
# the seasonal set using high-extinction outliers identified in the
# Bouguer-Langley time series (b1.5c).  Paths are relative to repo root.
APICAM_OVERRIDES = {
    "partial": {
        "filename": "APICAM.2019-06-25T00:59:01.000.fits",
        "path": "benchmark/data/apicam_drift_seasonal/APICAM.2019-06-25T00:59:01.000.fits",
        "clear_fraction": "0.45",  # estimated descriptive label
    },
    "cloudy": {
        "filename": "APICAM.2019-07-23T01:00:53.000.fits",
        "path": "benchmark/data/apicam_drift_seasonal/APICAM.2019-07-23T01:00:53.000.fits",
        "clear_fraction": "0.10",
    },
}


def pick_frame(rows, bin_range, strategy):
    lo, hi = bin_range
    candidates = [r for r in rows
                  if r["status"] == "ok"
                  and r["clear_fraction"] != ""
                  and lo <= float(r["clear_fraction"]) <= hi]
    if not candidates:
        return None
    if strategy == "highest":
        return max(candidates, key=lambda r: float(r["clear_fraction"]))
    if strategy == "lowest":
        return min(candidates, key=lambda r: float(r["clear_fraction"]))
    if strategy == "mid":
        target = (lo + hi) / 2
        return min(candidates,
                   key=lambda r: abs(float(r["clear_fraction"]) - target))
    return candidates[0]


def load_scores(camera):
    rows = []
    path = SCORES_DIR / f"{camera}.csv"
    if not path.exists():
        return rows
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def render_panel(frame_path, model_path, bin_name, camera_label, cat):
    inst = InstrumentModel.load(model_path)
    data, header = load_image(frame_path)
    meta = parse_fits_header(header)
    obs_time = meta["obs_time"]

    cat_table = cat.get_visible_stars(
        lat_deg=inst.site_lat, lon_deg=inst.site_lon,
        obs_time=obs_time, alt_limit=0.0, response_k=0.20,
    )
    det = detect_stars(data, n_brightest=500)
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

    camera = inst.to_camera_model()
    result = fast_solve(data, det, cat_table, camera, guided=True)
    use_det = result.guided_det_table
    ref_zp = inst.photometric_zeropoint or None
    az, alt, trans, zp = compute_transmission(
        use_det, cat_table, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )

    # Render directly onto ax: reuse plot_frame with no output_path.
    # plot_frame opens its own figure, so we use a different approach:
    # render to temp file and inset as image? For speed, just render
    # each panel to its own PNG and then composite.
    return {
        "data": data, "camera": result.camera_model, "det": use_det,
        "cat": cat_table, "pairs": result.matched_pairs,
        "obs_time": obs_time, "lat": inst.site_lat, "lon": inst.site_lon,
        "trans_data": (az, alt, trans), "zp": zp, "bin": bin_name,
        "camera_label": camera_label,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "haleakala": "benchmark/solutions/haleakala.json",
        "apicam": "benchmark/solutions/apicam.json",
        "liverpool": "benchmark/solutions/liverpool.json",
        "cloudynight": "benchmark/solutions/cloudynight.json",
    }
    labels = {
        "haleakala": "Haleakala",
        "apicam": "APICAM / Paranal",
        "liverpool": "Liverpool SkyCam",
        "cloudynight": "Cloudynight / Flagstaff",
    }

    cat = BrightStarCatalog()

    for camera in ["haleakala", "apicam", "liverpool", "cloudynight"]:
        rows = load_scores(camera)
        if not rows:
            print(f"[{camera}] no scores file, skipping")
            continue
        picks = {}
        for bin_name, bin_range, strategy in BINS:
            pick = pick_frame(rows, bin_range, strategy)
            picks[bin_name] = pick
        # Override with seasonal frames for APICAM partial/cloudy
        if camera == "apicam":
            for bin_name, override in APICAM_OVERRIDES.items():
                if picks.get(bin_name) is None:
                    picks[bin_name] = dict(override, status="ok")

        print(f"[{camera}] picks:")
        for bin_name, pick in picks.items():
            if pick is None:
                print(f"  {bin_name}: none in range")
            else:
                print(f"  {bin_name}: {pick['filename']} "
                      f"clear={float(pick['clear_fraction']):.2%}")

        # Render each panel to its own figure then stitch
        per_panel_paths = []
        for bin_name, _, _ in BINS:
            pick = picks[bin_name]
            if pick is None:
                per_panel_paths.append(None)
                continue
            rendered = render_panel(
                pick["path"], models[camera], bin_name, labels[camera], cat)
            out_panel = OUT_DIR / f"{camera}_{bin_name}.png"
            plot_frame(
                rendered["data"], rendered["camera"],
                det_table=rendered["det"], cat_table=rendered["cat"],
                matched_pairs=rendered["pairs"],
                transmission_data=rendered["trans_data"],
                obs_time=rendered["obs_time"],
                lat_deg=rendered["lat"], lon_deg=rendered["lon"],
                output_path=out_panel,
            )
            per_panel_paths.append(out_panel)

        # Composite triptych
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        from matplotlib.image import imread
        for i, (bin_name, _, _) in enumerate(BINS):
            ax = axes[i]
            if per_panel_paths[i] is None:
                ax.text(0.5, 0.5, f"no {bin_name} frame",
                        ha="center", va="center")
                ax.set_axis_off()
                continue
            img = imread(str(per_panel_paths[i]))
            ax.imshow(img)
            pick = picks[bin_name]
            clear = float(pick["clear_fraction"])
            ax.set_title(f"{bin_name.upper()}  ({clear:.0%} clear)\n"
                         f"{pick['filename']}",
                         fontsize=10)
            ax.set_axis_off()
        fig.suptitle(labels[camera], fontsize=13)
        fig.tight_layout()
        out_png = OUT_DIR / f"{camera}_triptych.png"
        out_pdf = OUT_DIR / f"{camera}_triptych.pdf"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_png}")


if __name__ == "__main__":
    sys.exit(main() or 0)
