"""Score every frame in each camera's benchmark dataset by clear_fraction.

For each of the four benchmark cameras, runs fast-solve on every frame
in benchmark/data/<camera>/ using the calibrated model in
benchmark/solutions/<camera>.json, computes the transmission map, and
records the clear fraction (transmission > 0.7).

Writes benchmark/results/frame_scores/<camera>.csv per camera and a
combined benchmark/results/frame_scores/all_scores.csv.

Run:
    uv run python benchmark/scripts/score_frames.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from allclear.instrument import InstrumentModel
from allclear.solver import fast_solve
from allclear.catalog import BrightStarCatalog
from allclear.detection import detect_stars
from allclear.transmission import compute_transmission, interpolate_transmission
from allclear.utils import load_image, parse_fits_header

CAMERAS = [
    ("haleakala", "benchmark/data/haleakala", "benchmark/solutions/haleakala.json"),
    ("apicam", "benchmark/data/eso_apicam", "benchmark/solutions/apicam.json"),
    ("liverpool", "benchmark/data/liverpool_skycam",
     "benchmark/solutions/liverpool.json"),
    ("cloudynight", "benchmark/data/cloudynight",
     "benchmark/solutions/cloudynight.json"),
]

OUT_DIR = Path("benchmark/results/frame_scores")
THRESHOLD = 0.7


def score_frame(frame_path: Path, inst: InstrumentModel, cat: BrightStarCatalog):
    data, header = load_image(str(frame_path))
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

    if result.n_matched < 5:
        return {
            "status": "low_matches",
            "n_match": int(result.n_matched),
            "rms_px": float(result.rms_residual),
            "clear_fraction": np.nan,
            "median_transmission": np.nan,
            "zeropoint": np.nan,
        }

    use_det = result.guided_det_table
    ref_zp = inst.photometric_zeropoint or None
    az, alt, trans, zp = compute_transmission(
        use_det, cat_table, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )
    tmap = interpolate_transmission(az, alt, trans)
    clear_mask = tmap.get_observability_mask(threshold=THRESHOLD)
    clear_frac = float(np.nanmean(clear_mask))
    median_trans = float(np.nanmedian(tmap.transmission))

    return {
        "status": "ok",
        "n_match": int(result.n_matched),
        "rms_px": float(result.rms_residual),
        "clear_fraction": clear_frac,
        "median_transmission": median_trans,
        "zeropoint": float(zp),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat = BrightStarCatalog()  # shared across frames

    combined = []

    for camera_name, data_dir, model_path in CAMERAS:
        data_dir = Path(data_dir)
        model_path = Path(model_path)
        if not data_dir.exists() or not model_path.exists():
            print(f"[{camera_name}] missing data dir or model")
            continue

        inst = InstrumentModel.load(model_path)
        frames = sorted([p for p in data_dir.iterdir()
                         if p.suffix.lower() in (".fits", ".fit")])
        print(f"[{camera_name}] {len(frames)} frames")

        rows = []
        for i, frame in enumerate(frames):
            try:
                score = score_frame(frame, inst, cat)
            except Exception as e:
                print(f"  [{i+1}/{len(frames)}] {frame.name}: "
                      f"ERROR {type(e).__name__}: {e}")
                score = {
                    "status": f"error:{type(e).__name__}",
                    "n_match": 0, "rms_px": np.nan,
                    "clear_fraction": np.nan,
                    "median_transmission": np.nan,
                    "zeropoint": np.nan,
                }
            row = {
                "camera": camera_name,
                "filename": frame.name,
                "path": str(frame),
                **score,
            }
            rows.append(row)
            combined.append(row)
            if score["status"] == "ok":
                print(f"  [{i+1}/{len(frames)}] {frame.name}: "
                      f"clear={score['clear_fraction']:.2%}, "
                      f"n_match={score['n_match']}, "
                      f"median_T={score['median_transmission']:.3f}")
            else:
                print(f"  [{i+1}/{len(frames)}] {frame.name}: "
                      f"{score['status']} (n_match={score['n_match']})")

        fields = ["camera", "filename", "path", "status", "n_match",
                  "rms_px", "clear_fraction", "median_transmission",
                  "zeropoint"]
        out_path = OUT_DIR / f"{camera_name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {out_path}")

    all_path = OUT_DIR / "all_scores.csv"
    with all_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "camera", "filename", "path", "status", "n_match",
            "rms_px", "clear_fraction", "median_transmission", "zeropoint",
        ])
        writer.writeheader()
        writer.writerows(combined)
    print(f"Combined: {all_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
