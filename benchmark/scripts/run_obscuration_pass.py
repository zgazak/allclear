"""Per-star detection-status accumulator for the APICAM seasonal data.

For each frame in the seasonal set:
  1. Load the solved per-night model.
  2. Re-match stars on the frame and compute the transmission map.
  3. For every catalog star that is predicted in-frame (inside image
     bounds, altitude above 5 deg, vmag < 7), record whether it was
     matched by the guided solver, plus its projected sky and pixel
     positions, the frame's clear fraction, and the measured flux.

Output: benchmark/results/obscuration/per_star_observations.csv
        (approximately 260 frames x 500 stars/frame = 130k rows)

This dataset underpins the obscuration map (B4b) — positions where a
bright in-frame catalog star is predicted under clear-sky conditions
but consistently goes undetected indicate permanent obstructions
(dome, tree, neighbor building, horizon terrain).

Run:
    uv run python benchmark/scripts/run_obscuration_pass.py
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

FRAMES_DIR = Path("benchmark/data/apicam_drift_seasonal")
MODELS_DIR = Path("benchmark/results/apicam_seasonal_blind")
OUT_CSV = Path("benchmark/results/obscuration/per_star_observations.csv")

VMAG_LIMIT = 6.5
ALT_MIN = 5.0
THRESHOLD = 0.7


def process_frame(frame_path, model_path, cat):
    inst = InstrumentModel.load(model_path)
    camera = inst.to_camera_model()
    data, header = load_image(str(frame_path))
    meta = parse_fits_header(header)
    obs_time = meta["obs_time"]

    cat_table = cat.get_visible_stars(
        lat_deg=inst.site_lat, lon_deg=inst.site_lon,
        obs_time=obs_time, alt_limit=0.0, response_k=0.20,
    )
    det_table = detect_stars(data, n_brightest=500)
    if inst.mirrored:
        data = data[:, ::-1]
        det_table["x"] = (data.shape[1] - 1) - np.asarray(
            det_table["x"], dtype=np.float64)

    result = fast_solve(data, det_table, cat_table, camera)
    if result.n_matched < 30:
        return None

    # Transmission for clear-fraction gating
    use_det = result.guided_det_table
    ref_zp = inst.photometric_zeropoint or None
    az_t, alt_t, trans_t, zp = compute_transmission(
        use_det, cat_table, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )
    tmap = interpolate_transmission(az_t, alt_t, trans_t)
    clear_frac = float(np.nanmean(
        tmap.get_observability_mask(threshold=THRESHOLD)))

    # Project every catalog star through the solved model
    camera_solved = result.camera_model
    vmag = np.asarray(cat_table["vmag"], dtype=np.float64)
    alt_deg = np.asarray(cat_table["alt_deg"], dtype=np.float64)
    az_deg = np.asarray(cat_table["az_deg"], dtype=np.float64)

    in_catalog = (vmag < VMAG_LIMIT) & (alt_deg > ALT_MIN)
    idx = np.where(in_catalog)[0]
    if idx.size == 0:
        return None

    px, py = camera_solved.sky_to_pixel(
        np.radians(az_deg[idx]), np.radians(alt_deg[idx]))
    ny, nx = data.shape
    in_frame = (np.isfinite(px) & np.isfinite(py)
                & (px >= 0) & (px < nx) & (py >= 0) & (py < ny))

    matched_cat_idx = {ci for _, ci in result.matched_pairs}

    rows = []
    for k, ci in enumerate(idx):
        if not in_frame[k]:
            continue
        detected = 1 if int(ci) in matched_cat_idx else 0
        rows.append({
            "frame": frame_path.name,
            "hip_idx": int(ci),
            "vmag": round(float(vmag[ci]), 3),
            "az_deg": round(float(az_deg[ci]), 3),
            "alt_deg": round(float(alt_deg[ci]), 3),
            "pixel_x": round(float(px[k]), 1),
            "pixel_y": round(float(py[k]), 1),
            "detected": detected,
            "clear_fraction": round(clear_frac, 4),
        })
    return rows


def main(argv=None):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cat = BrightStarCatalog()

    fields = ["frame", "hip_idx", "vmag", "az_deg", "alt_deg",
              "pixel_x", "pixel_y", "detected", "clear_fraction"]

    # Stream results to CSV as we go
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        model_files = sorted(MODELS_DIR.glob("*_model.json"))
        n_frames = 0
        n_frames_ok = 0
        n_rows = 0
        for i, model_path in enumerate(model_files):
            stem = model_path.name.replace("_model.json", "")
            frame_path = FRAMES_DIR / f"{stem}.fits"
            if not frame_path.exists():
                continue
            try:
                rows = process_frame(frame_path, model_path, cat)
            except Exception as e:
                print(f"  [{i+1}/{len(model_files)}] {frame_path.name}: "
                      f"EXCEPTION {type(e).__name__}: {e}", flush=True)
                continue
            n_frames += 1
            if rows is None:
                continue
            writer.writerows(rows)
            n_frames_ok += 1
            n_rows += len(rows)
            if (i + 1) % 10 == 0 or (i + 1) == len(model_files):
                print(f"  [{i+1}/{len(model_files)}] ok_frames={n_frames_ok}"
                      f" rows={n_rows}  clear={rows[0]['clear_fraction']:.2f}"
                      f" n_stars={len(rows)}", flush=True)

    print(f"\nDone. {n_frames_ok} frames, {n_rows} star observations.")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main() or 0)
