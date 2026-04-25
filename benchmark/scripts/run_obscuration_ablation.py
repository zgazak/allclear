"""Obscuration-mask ablation on an APICAM frame.

Four runs on a single frame:

    blind    without mask        (current default: raw detections + full catalog)
    blind    with mask           (pixel-mask dome via prior camera, re-detect,
                                  then fit — the "rainy-season return" scenario)
    solve    without mask        (fast_solve, obscuration=None)
    solve    with mask           (fast_solve, obscuration=mask)

For each run the fit and transmission-map images are saved, plus a
composite 4x2 panel figure and a bar chart of match count / RMS.

CLI:
    uv run python benchmark/scripts/run_obscuration_ablation.py
        --frame path/to/frame.fits
        --model path/to/model.json
        --suffix stress          # outputs go to ablation_stress_*
        --mask path/to/obscuration_map.json
"""
from __future__ import annotations

import argparse
import csv
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allclear.catalog import BrightStarCatalog
from allclear.cli import _load_frame
from allclear.detection import detect_stars
from allclear.instrument import InstrumentModel
from allclear.matching import match_sources
from allclear.obscuration import ObscurationMask
from allclear.solver import fast_solve
from allclear.strategies import instrument_fit_pipeline
from allclear.transmission import compute_transmission, interpolate_transmission


DEFAULT_FRAME = Path("benchmark/data/apicam_drift_seasonal/"
                     "APICAM.2019-06-06T00:59:45.000.fits")
DEFAULT_MODEL = Path("benchmark/results/apicam_seasonal_blind/"
                     "APICAM.2019-06-06T00:59:45.000_model.json")
DEFAULT_MASK = Path("benchmark/results/obscuration/obscuration_map.json")

OUT_DIR = Path("benchmark/results/obscuration")

LAT, LON = -24.6272, -70.4048


def _out_paths(suffix):
    # suffix='' preserves the original layout: ablation_results.{csv,png,pkl}
    # suffix='stress' → ablation_stress_results.{csv,png,pkl}
    tag = "ablation" if not suffix else f"ablation_{suffix}"
    return {
        "csv": OUT_DIR / f"{tag}_results.csv",
        "bars": OUT_DIR / f"{tag}_results.png",
        "panels": OUT_DIR / f"{tag}_panels.png",
        "pkl": OUT_DIR / f"{tag}_results.pkl",
    }


@dataclass
class RunResult:
    mode: str            # "blind" | "solve"
    mask: str            # "without" | "with"
    n_matched: int
    rms_px: float
    n_catalog: int
    n_masked_pixels: int   # obscured pixels zeroed (blind-with-mask only)
    elapsed_s: float
    camera_model: object = field(default=None, repr=False)
    matched_pairs: list = field(default=None, repr=False)
    image: np.ndarray = field(default=None, repr=False)
    det_table: object = field(default=None, repr=False)
    cat_table: object = field(default=None, repr=False)
    trans: tuple = field(default=None, repr=False)   # (az, alt, trans)
    zp: float = 0.0


# ---------------------------------------------------------------------
# Blind / solve runners
# ---------------------------------------------------------------------

def _mirror(data, det):
    """Return (mirrored_data, mirrored_det). Leaves inputs untouched."""
    out = data[:, ::-1].copy()
    det2 = det.copy()
    det2["x"] = (out.shape[1] - 1
                 - np.asarray(det2["x"], dtype=np.float64))
    return out, det2


def run_blind(data_m, cat, det_m, initial_f, mask=None,
              prior_camera=None):
    """Run blind instrument-fit on a (pre-mirrored) image.

    If ``mask`` is given, the mask is projected onto the image via
    ``prior_camera`` and obscured pixels are replaced with the local
    background before detection — simulating the operational
    "rainy-season return" where you still have the pre-shutdown
    model + mask but want a fresh pointing solve.
    """
    data_in = data_m
    det_in = det_m
    n_masked = 0
    if mask is not None:
        assert prior_camera is not None, \
            "prior_camera required for blind-with-mask"
        pix_mask = mask.project_to_pixel_mask(prior_camera, data_in.shape)
        n_masked = int(np.sum(pix_mask))
        bg = float(np.median(data_in[~pix_mask]))
        data_in = np.where(pix_mask, bg, data_in)
        # Re-detect on the cleaned image so dome peaks are not in the
        # top-N detection list.
        det_in = detect_stars(data_in, fwhm=5.0, threshold_sigma=5.0,
                              n_brightest=len(det_m))

    t0 = time.time()
    model, n_matched, rms, diag = instrument_fit_pipeline(
        data_in, det_in, cat, initial_f=initial_f, verbose=False,
    )
    elapsed = time.time() - t0

    # Recompute matched pairs (for plotting) from the final model.
    cat_az = np.radians(np.asarray(cat["az_deg"], dtype=np.float64))
    cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=np.float64))
    cat_x, cat_y = model.sky_to_pixel(cat_az, cat_alt)
    cat_xy = np.column_stack([cat_x, cat_y])
    det_xy = np.column_stack([
        np.asarray(det_in["x"], dtype=np.float64),
        np.asarray(det_in["y"], dtype=np.float64),
    ])
    pairs, _ = match_sources(det_xy, cat_xy, max_dist=15.0)

    # Transmission
    az, alt, tvals, zp = compute_transmission(
        det_in, cat, pairs, model, image=data_in,
        reference_zeropoint=None,
    )

    return RunResult(
        mode="blind",
        mask="with" if mask is not None else "without",
        n_matched=int(n_matched),
        rms_px=float(rms),
        n_catalog=len(cat),
        n_masked_pixels=n_masked,
        elapsed_s=elapsed,
        camera_model=model,
        matched_pairs=pairs,
        image=data_in,
        det_table=det_in,
        cat_table=cat,
        trans=(az, alt, tvals),
        zp=float(zp) if zp is not None else 0.0,
    )


def run_solve(data_m, cat, det_m, camera, ref_zp, mask=None):
    t0 = time.time()
    result = fast_solve(data_m, det_m, cat, camera, obscuration=mask)
    elapsed = time.time() - t0

    use_det = (result.guided_det_table
               if result.guided_det_table is not None
               and len(result.guided_det_table) > 0
               else det_m)
    az, alt, tvals, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data_m, reference_zeropoint=ref_zp,
        obscuration=mask,
    )

    return RunResult(
        mode="solve",
        mask="with" if mask is not None else "without",
        n_matched=int(result.n_matched),
        rms_px=float(result.rms_residual),
        n_catalog=len(cat),
        n_masked_pixels=0,
        elapsed_s=elapsed,
        camera_model=result.camera_model,
        matched_pairs=result.matched_pairs,
        image=data_m,
        det_table=use_det,
        cat_table=cat,
        trans=(az, alt, tvals),
        zp=float(zp) if zp is not None else 0.0,
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _thumb_ax(ax, data, title):
    finite = np.isfinite(data)
    vmin = float(np.nanpercentile(data[finite], 2))
    vmax = float(np.nanpercentile(data[finite], 99.5))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10)


def _draw_fit(ax, r):
    _thumb_ax(ax, r.image, f"{r.mode} {r.mask} — fit")
    model = r.camera_model
    cat = r.cat_table

    # All catalog stars projected (dim red circles, scaled by vmag).
    cat_az = np.radians(np.asarray(cat["az_deg"], dtype=np.float64))
    cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=np.float64))
    cx, cy = model.sky_to_pixel(cat_az, cat_alt)
    vmag = np.asarray(cat["vmag"], dtype=np.float64)
    ny, nx = r.image.shape
    in_frame = (np.isfinite(cx) & np.isfinite(cy)
                & (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny))
    size = np.clip(20.0 * (6.0 - vmag), 2.0, 40.0) ** 0.7
    ax.scatter(cx[in_frame], cy[in_frame], s=size[in_frame],
               facecolors="none", edgecolors="#cc5555",
               linewidths=0.4, alpha=0.55)

    # Matched stars: green crosshairs at catalog-projected position.
    if r.matched_pairs:
        cat_idx = np.array([ci for _, ci in r.matched_pairs], dtype=int)
        ax.scatter(cx[cat_idx], cy[cat_idx], s=24,
                   marker="+", color="#2a7f62",
                   linewidths=0.5, alpha=0.9)

    ax.text(0.02, 0.98,
            f"matched {r.n_matched}\nRMS {r.rms_px:.2f} px",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.55,
                      edgecolor="none", pad=3))


def _draw_transmission(ax, r, tmap_cache):
    _thumb_ax(ax, r.image, f"{r.mode} {r.mask} — transmission")
    model = r.camera_model
    az, alt, tvals = r.trans
    if az is None or len(az) < 3:
        return

    # Cache interpolated map per run
    key = id(r)
    tmap = tmap_cache.get(key)
    if tmap is None:
        tmap = interpolate_transmission(az, alt, tvals)
        tmap_cache[key] = tmap

    # Render RdYlGn overlay at sparse sampled grid onto pixel image.
    ny, nx = r.image.shape
    step = max(1, min(nx, ny) // 120)
    yy, xx = np.mgrid[0:ny:step, 0:nx:step]
    with np.errstate(invalid="ignore"):
        az_rad, alt_rad = model.pixel_to_sky(
            xx.astype(np.float64), yy.astype(np.float64))
    az_deg = np.degrees(az_rad)
    alt_deg = np.degrees(alt_rad)
    valid = np.isfinite(az_deg) & np.isfinite(alt_deg) & (alt_deg > 5)
    tpix = np.full(az_deg.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        ai = np.clip(
            np.searchsorted(tmap.alt_grid, alt_deg[valid]),
            0, len(tmap.alt_grid) - 1,
        )
        zi = np.clip(
            np.searchsorted(tmap.az_grid, np.mod(az_deg[valid], 360.0)),
            0, len(tmap.az_grid) - 1,
        )
        tpix[valid] = tmap.transmission[ai, zi]
    ax.imshow(
        tpix, origin="lower", cmap="RdYlGn", vmin=0.0, vmax=1.0,
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
        alpha=0.55, interpolation="nearest",
    )

    # Clear fraction text
    clear_frac = float(np.nanmean(
        tmap.get_observability_mask(threshold=0.7)))
    ax.text(0.02, 0.98,
            f"clear {clear_frac:.0%}\nzp {r.zp:.2f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, color="white",
            bbox=dict(facecolor="black", alpha=0.55,
                      edgecolor="none", pad=3))


def render_panels(results, out_path, title):
    """4 columns (runs) × 2 rows (fit, transmission)."""
    order = [
        ("blind", "without"),
        ("blind", "with"),
        ("solve", "without"),
        ("solve", "with"),
    ]
    by_key = {(r.mode, r.mask): r for r in results}

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    tmap_cache = {}
    for col, key in enumerate(order):
        r = by_key[key]
        _draw_fit(axes[0, col], r)
        _draw_transmission(axes[1, col], r, tmap_cache)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_bars(results, out_path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    modes = ["blind", "solve"]
    variants = ["without", "with"]
    x = np.arange(len(modes))
    w = 0.36

    def val(metric):
        out = {"without": [], "with": []}
        for mode in modes:
            for v in variants:
                rr = next(r for r in results
                          if r.mode == mode and r.mask == v)
                out[v].append(getattr(rr, metric))
        return out

    m = val("n_matched")
    ax1.bar(x - w / 2, m["without"], w, label="without mask",
            color="#999999")
    ax1.bar(x + w / 2, m["with"], w, label="with mask",
            color="#2a7f62")
    ax1.set_xticks(x); ax1.set_xticklabels([s.capitalize() for s in modes])
    ax1.set_ylabel("stars matched"); ax1.set_title("Match count")
    ax1.legend(frameon=False)
    for xi, _ in enumerate(modes):
        for s, v in enumerate(variants):
            dx = (s - 0.5) * w
            n = m[v][xi]
            ax1.text(xi + dx, n, f"{n}", ha="center", va="bottom",
                     fontsize=9)

    r = val("rms_px")
    ax2.bar(x - w / 2, r["without"], w, label="without mask",
            color="#999999")
    ax2.bar(x + w / 2, r["with"], w, label="with mask",
            color="#2a7f62")
    ax2.set_xticks(x); ax2.set_xticklabels([s.capitalize() for s in modes])
    ax2.set_ylabel("RMS residual (px)"); ax2.set_title("Astrometric RMS")
    ax2.legend(frameon=False)
    for xi, _ in enumerate(modes):
        for s, v in enumerate(variants):
            dx = (s - 0.5) * w
            val_ = r[v][xi]
            ax2.text(xi + dx, val_, f"{val_:.2f}", ha="center",
                     va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_table(results):
    hdr = (f"{'mode':6s}  {'mask':8s}  {'matched':>8s}  {'RMS px':>7s}"
           f"  {'cat_in':>7s}  {'mask px':>9s}  {'time s':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.mode:6s}  {r.mask:8s}  {r.n_matched:>8d}  "
              f"{r.rms_px:>7.2f}  {r.n_catalog:>7d}  "
              f"{r.n_masked_pixels:>9d}  {r.elapsed_s:>7.1f}")


def write_csv(results, out_path):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "mask", "n_matched", "rms_px",
                    "n_catalog", "n_masked_pixels", "elapsed_s",
                    "clear_fraction", "zeropoint"])
        for r in results:
            tmap = None
            cf = 0.0
            if r.trans is not None and r.trans[0] is not None:
                az, alt, tv = r.trans
                if len(az) >= 3:
                    tmap = interpolate_transmission(az, alt, tv)
                    cf = float(np.nanmean(
                        tmap.get_observability_mask(threshold=0.7)))
            w.writerow([r.mode, r.mask, r.n_matched,
                        round(r.rms_px, 3),
                        r.n_catalog, r.n_masked_pixels,
                        round(r.elapsed_s, 2),
                        round(cf, 4), round(r.zp, 4)])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--suffix", type=str, default="",
                    help="Output suffix, e.g. 'stress' → ablation_stress_*")
    ap.add_argument("--lat", type=float, default=LAT)
    ap.add_argument("--lon", type=float, default=LON)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = _out_paths(args.suffix)

    title_base = args.title or f"{args.frame.stem} obscuration-mask ablation"

    print(f"Loading {args.frame.name}")
    data, meta, cat, det, initial_f = _load_frame(
        str(args.frame), args.lat, args.lon)
    print(f"  image: {data.shape}, catalog: {len(cat)}, "
          f"detections: {len(det)}")

    inst = InstrumentModel.load(str(args.model))
    prior_camera = inst.to_camera_model()
    mask = ObscurationMask.load(str(args.mask))
    ref_zp = inst.photometric_zeropoint or None

    # Pre-mirror the frame + detections because the camera is mirrored
    # — both modes use the same input so the fits are comparable.
    if inst.mirrored:
        data_m, det_m = _mirror(data, det)
    else:
        data_m, det_m = data, det

    # Re-detect on mirrored image with the solver-standard n_brightest
    # to match blind_solve's expected input.
    det_m = detect_stars(data_m, fwhm=5.0, threshold_sigma=5.0,
                         n_brightest=len(det_m))

    if paths["pkl"].exists():
        print(f"Loading cached results from {paths['pkl']}")
        with paths["pkl"].open("rb") as f:
            results = pickle.load(f)
    else:
        results = []

        print("[1/4] blind, without mask")
        results.append(run_blind(data_m, cat, det_m, initial_f, mask=None))

        print("[2/4] blind, with mask (pixel-mask dome via prior camera)")
        results.append(run_blind(data_m, cat, det_m, initial_f,
                                 mask=mask, prior_camera=prior_camera))

        print("[3/4] solve, without mask")
        results.append(run_solve(data_m, cat, det_m, prior_camera,
                                 ref_zp, mask=None))

        print("[4/4] solve, with mask")
        results.append(run_solve(data_m, cat, det_m, prior_camera,
                                 ref_zp, mask=mask))

        with paths["pkl"].open("wb") as f:
            pickle.dump(results, f)

    print()
    print_table(results)

    write_csv(results, paths["csv"])
    render_bars(results, paths["bars"], f"{title_base} (metrics)")
    render_panels(results, paths["panels"],
                  f"{title_base} (fit / transmission)")

    print(f"\nWrote: {paths['csv']}")
    print(f"Wrote: {paths['bars']}")
    print(f"Wrote: {paths['panels']}")


if __name__ == "__main__":
    main()
