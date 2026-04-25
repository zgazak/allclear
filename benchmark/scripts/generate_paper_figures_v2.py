#!/usr/bin/env python3
"""Generate publication-quality figures for the AllClear paper (v2).

Three figures:
  1. Solved frame showcase — APICAM full-width, grid + matched stars
  2. Transmission triptych — Cloudynight clear / partial / heavy cloud
  3. Hardware diversity strip — all 4 cameras, one panel each

Usage:
    python generate_paper_figures_v2.py --output benchmark/results/paper_figures_v2
    python generate_paper_figures_v2.py --figure 2 --output /tmp/figs  # single figure
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

log = logging.getLogger(__name__)

# ---- Import from the existing script for shared helpers ----
sys.path.insert(0, str(Path(__file__).parent))
from generate_paper_figures import (
    CAMERAS, stretch_image, crop_to_sky, load_and_solve,
    draw_grid, draw_matched_stars, overlay_transmission, draw_planets,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
PASP_FULL_WIDTH = 7.1    # inches, double-column
PASP_COL_WIDTH = 3.4     # inches, single-column
DPI = 300

FONT = {
    "panel_label": 14,
    "title": 11,
    "subtitle": 9,
    "grid_label": 7,
    "stats": 9,
    "legend": 8,
    "colorbar": 8,
    "strip_label": 9,
}

# Transmission figure — Cloudynight frames
TRANSMISSION_FRAMES = {
    "clear": {
        "frame": "benchmark/data/cloudynight/008.fits",
        "label": "Clear sky",
    },
    "partial": {
        "frame": "benchmark/data/cloudynight/006.fits",
        "label": "Partial cloud",
    },
    "heavy": {
        "frame": "benchmark/data/cloudynight/000.fits",
        "label": "Heavy cloud",
    },
}


def _make_crop_model(model, x0, y0):
    """Create a camera model shifted to cropped coordinates."""
    from allclear.projection import CameraModel
    return CameraModel(
        cx=model.cx - x0, cy=model.cy - y0,
        az0=model.az0, alt0=model.alt0, rho=model.rho, f=model.f,
        proj_type=model.proj_type, k1=model.k1, k2=model.k2,
    )


def _solve_frame(frame_path, model_path, lat, lon):
    """Load a frame, solve with given model, return solve dict."""
    from allclear.instrument import InstrumentModel
    from allclear.cli import _load_frame
    from allclear.solver import fast_solve
    from allclear.transmission import compute_transmission

    inst = InstrumentModel.load(model_path)
    camera_model = inst.to_camera_model()

    data, meta, cat, det, _ = _load_frame(str(frame_path), lat, lon)
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

    result = fast_solve(data, det, cat, camera_model)
    use_det = result.guided_det_table if result.guided_det_table is not None else det

    ref_zp = inst.photometric_zeropoint if inst.photometric_zeropoint != 0.0 else None
    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )

    return {
        "image": data, "model": result.camera_model,
        "det": use_det, "cat": cat, "pairs": result.matched_pairs,
        "n_matched": result.n_matched, "rms": result.rms_residual,
        "trans_az": az, "trans_alt": alt, "trans_vals": trans,
        "meta": meta, "inst": inst,
        "obs_time": meta.get("obs_time"),
    }


# =========================================================================
# Figure 1: Solved frame showcase (APICAM)
# =========================================================================

def figure_showcase(output_dir, root):
    """Single large panel: APICAM solved frame with grid + matched stars."""
    print("\n" + "=" * 60)
    print("  Figure 1: Solved frame showcase (APICAM)")
    print("=" * 60)

    cam_cfg = CAMERAS["apicam"]
    solved = load_and_solve(cam_cfg, root)
    if solved is None:
        print("  SKIPPED")
        return

    image = solved["image"]
    model = solved["model"]

    cropped, x0, y0 = crop_to_sky(image, model, padding_frac=0.02)
    cny, cnx = cropped.shape
    crop_model = _make_crop_model(model, x0, y0)

    from astropy.table import Table
    crop_det = Table(solved["det"])
    crop_det["x"] = np.asarray(crop_det["x"], dtype=float) - x0
    crop_det["y"] = np.asarray(crop_det["y"], dtype=float) - y0

    stretched = stretch_image(cropped, method="asinh")

    # Single panel, full width
    fig_w = PASP_FULL_WIDTH
    fig_h = fig_w * (cny / cnx)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(stretched, cmap="gray", origin="lower", aspect="equal",
              extent=[0, cnx, 0, cny])
    draw_grid(ax, crop_model, cnx, cny)
    draw_matched_stars(ax, crop_det, solved["cat"], solved["pairs"],
                       crop_model, cnx, cny, vmag_limit=5.5)
    if solved["obs_time"] is not None:
        draw_planets(ax, crop_model, solved["obs_time"],
                     cam_cfg["lat"], cam_cfg["lon"], cnx, cny)

    ax.set_xlim(0, cnx)
    ax.set_ylim(0, cny)
    ax.set_xticks([])
    ax.set_yticks([])

    # Stats overlay
    ax.text(0.98, 0.02,
            f"$n$ = {solved['n_matched']}    RMS = {solved['rms']:.2f} px",
            transform=ax.transAxes, fontsize=FONT["stats"],
            color="#44ff44", ha="right", va="bottom",
            fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Legend
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#44ff44",
               markerfacecolor="none", markersize=5, markeredgewidth=0.8,
               label="Matched"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#ff4444",
               markerfacecolor="none", markersize=5, markeredgewidth=0.8,
               label="Unmatched"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=FONT["legend"], framealpha=0.6,
              facecolor="black", edgecolor="gray", labelcolor="white")

    fig.savefig(output_dir / "fig_showcase_apicam.pdf",
                dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / "fig_showcase_apicam.png",
                dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> fig_showcase_apicam.pdf/png")


# =========================================================================
# Figure 2: Transmission triptych (Cloudynight)
# =========================================================================

def figure_transmission(output_dir, root):
    """Three-panel figure: clear / partial cloud / heavy cloud."""
    print("\n" + "=" * 60)
    print("  Figure 2: Transmission triptych (Cloudynight)")
    print("=" * 60)

    cam_cfg = CAMERAS["cloudynight"]
    model_path = root / cam_cfg["model"]

    panels = []
    for key in ["clear", "partial", "heavy"]:
        frame_cfg = TRANSMISSION_FRAMES[key]
        frame_path = root / frame_cfg["frame"]
        print(f"  Solving {key}: {frame_path.name}")

        solved = _solve_frame(frame_path, model_path,
                              cam_cfg["lat"], cam_cfg["lon"])
        solved["label"] = frame_cfg["label"]
        panels.append(solved)
        print(f"    n={solved['n_matched']}, rms={solved['rms']:.2f}")

    # All panels share the same crop (same camera model)
    ref_model = panels[0]["model"]
    _, x0, y0 = crop_to_sky(panels[0]["image"], ref_model, padding_frac=0.02)

    # Use 2/3 of full width for 3 panels — leaves room for colorbar
    # and keeps panels large enough to read at print
    usable_w = PASP_FULL_WIDTH - 0.6  # reserve for colorbar
    panel_w = usable_w / 3
    panel_h = panel_w  # square

    fig, axes = plt.subplots(1, 3, figsize=(PASP_FULL_WIDTH, panel_h + 0.6),
                             gridspec_kw={"wspace": 0.05,
                                          "left": 0.01, "right": 0.88,
                                          "bottom": 0.02, "top": 0.92})

    for i, (ax, solved) in enumerate(zip(axes, panels)):
        image = solved["image"]
        model = solved["model"]
        cropped, x0_i, y0_i = crop_to_sky(image, model, padding_frac=0.02)
        cny, cnx = cropped.shape
        crop_model = _make_crop_model(model, x0_i, y0_i)

        stretched = stretch_image(cropped, method="asinh")

        # Base image
        ax.imshow(stretched, cmap="gray", origin="lower", aspect="equal",
                  extent=[0, cnx, 0, cny])

        # Transmission overlay
        overlay_transmission(ax, crop_model,
                             solved["trans_az"], solved["trans_alt"],
                             solved["trans_vals"], cnx, cny)

        # Faint grid for orientation
        draw_grid(ax, crop_model, cnx, cny)

        if solved["obs_time"] is not None:
            draw_planets(ax, crop_model, solved["obs_time"],
                         cam_cfg["lat"], cam_cfg["lon"], cnx, cny)

        ax.set_xlim(0, cnx)
        ax.set_ylim(0, cny)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel title
        ax.set_title(solved["label"], fontsize=FONT["subtitle"],
                     fontweight="bold", pad=4)

        # Stats
        ax.text(0.97, 0.03,
                f"$n$ = {solved['n_matched']}",
                transform=ax.transAxes, fontsize=FONT["stats"] - 1,
                color="white", ha="right", va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Shared colorbar
    sm = ScalarMappable(cmap="RdYlGn", norm=Normalize(0, 1.2))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Transmission", fontsize=FONT["colorbar"])
    cb.ax.tick_params(labelsize=FONT["colorbar"])

    fig.savefig(output_dir / "fig_transmission_triptych.pdf",
                dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "fig_transmission_triptych.png",
                dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  -> fig_transmission_triptych.pdf/png")


# =========================================================================
# Figure 3: Hardware diversity strip (all 4 cameras)
# =========================================================================

def figure_hardware_strip(output_dir, root):
    """Four-panel horizontal strip: one solved frame per camera."""
    print("\n" + "=" * 60)
    print("  Figure 3: Hardware diversity strip")
    print("=" * 60)

    cam_order = ["haleakala", "apicam", "liverpool", "cloudynight"]

    solved_list = []
    for cam_name in cam_order:
        cam_cfg = CAMERAS[cam_name]
        print(f"  Solving {cam_cfg['label']}...")
        solved = load_and_solve(cam_cfg, root)
        if solved is None:
            print(f"    SKIPPED")
            continue
        solved["cam_name"] = cam_name
        solved["cam_cfg"] = cam_cfg
        solved_list.append(solved)
        print(f"    n={solved['n_matched']}, rms={solved['rms']:.2f}")

    n_cams = len(solved_list)
    fig_w = PASP_FULL_WIDTH
    panel_w = (fig_w - 0.3) / 2  # 2 columns
    panel_h = panel_w  # square panels

    fig, axes_2d = plt.subplots(2, 2,
                                figsize=(fig_w, fig_w + 0.5),
                                gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    axes = axes_2d.ravel()

    for ax, solved in zip(axes, solved_list):
        image = solved["image"]
        model = solved["model"]
        cam_cfg = solved["cam_cfg"]

        cropped, x0, y0 = crop_to_sky(image, model, padding_frac=0.02)
        cny, cnx = cropped.shape
        crop_model = _make_crop_model(model, x0, y0)

        from astropy.table import Table
        crop_det = Table(solved["det"])
        crop_det["x"] = np.asarray(crop_det["x"], dtype=float) - x0
        crop_det["y"] = np.asarray(crop_det["y"], dtype=float) - y0

        stretched = stretch_image(cropped, method="asinh")

        ax.imshow(stretched, cmap="gray", origin="lower", aspect="equal",
                  extent=[0, cnx, 0, cny])
        draw_grid(ax, crop_model, cnx, cny)
        draw_matched_stars(ax, crop_det, solved["cat"], solved["pairs"],
                           crop_model, cnx, cny, vmag_limit=4.5)

        if solved["obs_time"] is not None:
            draw_planets(ax, crop_model, solved["obs_time"],
                         cam_cfg["lat"], cam_cfg["lon"], cnx, cny)

        ax.set_xlim(0, cnx)
        ax.set_ylim(0, cny)
        ax.set_xticks([])
        ax.set_yticks([])

        # Camera label above, stats below
        ax.set_title(cam_cfg["label"], fontsize=FONT["strip_label"],
                     fontweight="bold", pad=4)

        # Stats inside the panel (bottom center) — avoids clipping
        ax.text(0.5, 0.02,
                f"{solved['n_matched']} stars, {solved['rms']:.2f} px",
                transform=ax.transAxes, fontsize=FONT["stats"] - 1,
                color="#44ff44", ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    fig.savefig(output_dir / "fig_hardware_strip.pdf",
                dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "fig_hardware_strip.png",
                dpi=150, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  -> fig_hardware_strip.pdf/png")


# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures (v2) for AllClear")
    parser.add_argument("--figure", type=int, nargs="*",
                        help="Which figures to generate (1,2,3). Default: all")
    parser.add_argument("--output", default="benchmark/results/paper_figures_v2",
                        help="Output directory")
    parser.add_argument("--root", default=".", help="Project root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = args.figure or [1, 2, 3]

    if 1 in figures:
        figure_showcase(output_dir, root)
    if 2 in figures:
        figure_transmission(output_dir, root)
    if 3 in figures:
        figure_hardware_strip(output_dir, root)

    print(f"\nDone. Figures in {output_dir}/")


if __name__ == "__main__":
    sys.exit(main() or 0)
