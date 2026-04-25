"""Illustration of the guided-matching step.

Three panels showing the algorithm on a real APICAM crop:

  1. Raw image crop with catalog-projected star position marked.
  2. Search box with the local peak-finding result.
  3. Sub-pixel centroid refinement overlaid.

Designed to make the "project catalog → search box → centroid" idea
visually obvious in one figure.

Output:
  benchmark/results/paper_figures_v3/fig_guided_matching.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from allclear.catalog import BrightStarCatalog
from allclear.detection import detect_stars
from allclear.instrument import InstrumentModel
from allclear.solver import fast_solve
from allclear.utils import load_image, parse_fits_header

FRAME = Path("benchmark/data/eso_apicam/APICAM.2019-02-10T01:52:35.000.fits")
MODEL = Path("benchmark/solutions/apicam.json")

OUT_DIR = Path("benchmark/results/paper_figures_v3")

SEARCH_RADIUS = 12   # px — displayed search box half-size
CROP_RADIUS = 40     # px — zoomed crop around chosen star


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inst = InstrumentModel.load(MODEL)
    camera = inst.to_camera_model()
    data, header = load_image(str(FRAME))
    meta = parse_fits_header(header)
    obs_time = meta["obs_time"]

    cat = BrightStarCatalog()
    cat_table = cat.get_visible_stars(
        lat_deg=inst.site_lat, lon_deg=inst.site_lon,
        obs_time=obs_time, alt_limit=0.0, response_k=0.20,
    )
    det_table = detect_stars(data, n_brightest=500)
    if inst.mirrored:
        data = data[:, ::-1]
        det_table["x"] = (data.shape[1] - 1) - np.asarray(
            det_table["x"], dtype=np.float64)

    result = fast_solve(data, det_table, cat_table, camera, guided=True)
    gdt = result.guided_det_table
    camera_solved = result.camera_model

    # Pick a moderately bright, well-isolated matched star for the demo
    # (avoid the very brightest to prevent saturation dominance).
    pairs = result.matched_pairs
    vmag_cat = np.asarray(cat_table["vmag"], dtype=np.float64)
    candidates = []
    for di, ci in pairs:
        v = float(vmag_cat[ci])
        if 2.5 < v < 3.5:
            candidates.append((di, ci, v))
    if not candidates:
        # fallback: any match
        candidates = [(pairs[0][0], pairs[0][1], float(vmag_cat[pairs[0][1]]))]

    # Use the first candidate
    di_use, ci_use, v_use = candidates[0]

    # Catalog-projected pixel position
    az_rad = np.radians(float(cat_table["az_deg"][ci_use]))
    alt_rad = np.radians(float(cat_table["alt_deg"][ci_use]))
    cx_proj, cy_proj = camera_solved.sky_to_pixel(az_rad, alt_rad)
    cx_proj = float(cx_proj)
    cy_proj = float(cy_proj)

    # Centroided position from guided matching
    cx_cen = float(gdt["x"][di_use])
    cy_cen = float(gdt["y"][di_use])

    # For the middle panel: show the search box and the peak.  Recompute
    # locally: find max in search box (clip to image bounds).
    r = SEARCH_RADIUS
    ny, nx = data.shape
    xi = int(round(cx_proj))
    yi = int(round(cy_proj))
    x0 = max(0, xi - r)
    x1 = min(nx, xi + r + 1)
    y0 = max(0, yi - r)
    y1 = min(ny, yi + r + 1)
    box = data[y0:y1, x0:x1]
    peak_local = np.unravel_index(int(np.argmax(box)), box.shape)
    peak_x = peak_local[1] + x0
    peak_y = peak_local[0] + y0

    # Zoom window for all panels
    cx0 = int(round(cx_proj)) - CROP_RADIUS
    cx1 = int(round(cx_proj)) + CROP_RADIUS
    cy0 = int(round(cy_proj)) - CROP_RADIUS
    cy1 = int(round(cy_proj)) + CROP_RADIUS
    crop = data[cy0:cy1, cx0:cx1]

    vmin = float(np.nanpercentile(crop, 5))
    vmax = float(np.nanpercentile(crop, 99.5))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    extent = (cx0, cx1, cy0, cy1)

    # Panel 1: raw crop with catalog projection marker
    ax = axes[0]
    ax.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              extent=extent)
    ax.scatter([cx_proj], [cy_proj], s=150, marker="x", color="#d62728",
               linewidths=2, label="catalog projection")
    ax.set_title("1. Project catalog star\nto pixel plane")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Panel 2: search box and peak
    ax = axes[1]
    ax.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              extent=extent)
    rect = mpatches.Rectangle((cx_proj - r, cy_proj - r),
                              2 * r, 2 * r, edgecolor="#1f77b4",
                              facecolor="none", lw=1.5,
                              label=f"search box ({2*r}$\\times${2*r} px)")
    ax.add_patch(rect)
    ax.scatter([peak_x], [peak_y], s=150, marker="o",
               facecolors="none", edgecolors="#2ca02c", linewidths=2,
               label="local peak")
    ax.scatter([cx_proj], [cy_proj], s=80, marker="x", color="#d62728",
               linewidths=1.5, alpha=0.6)
    ax.set_title(f"2. Search peak within box\n(accept nearest bright source)")
    ax.set_xlabel("pixel x")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Panel 3: centroided position
    ax = axes[2]
    ax.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              extent=extent)
    ax.scatter([cx_cen], [cy_cen], s=200, marker="+", color="#ff7f0e",
               linewidths=2,
               label=f"sub-pixel centroid\n({cx_cen:.1f}, {cy_cen:.1f})")
    # Vector from projection to centroid
    dx = cx_cen - cx_proj
    dy = cy_cen - cy_proj
    ax.annotate("", xy=(cx_cen, cy_cen), xytext=(cx_proj, cy_proj),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    ax.scatter([cx_proj], [cy_proj], s=80, marker="x", color="#d62728",
               linewidths=1.5, alpha=0.6,
               label=f"catalog proj\nresidual=({dx:.1f}, {dy:.1f}) px")
    ax.set_title("3. Centroid for\nsub-pixel accuracy")
    ax.set_xlabel("pixel x")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    star_name = f"catalog idx={ci_use}, $V$={v_use:.2f}"
    fig.suptitle(f"Guided-matching workflow ({star_name})",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_guided_matching.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
