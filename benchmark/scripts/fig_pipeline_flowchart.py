"""Schematic flowchart of the blind instrument-fit pipeline.

Uses matplotlib's FancyBboxPatch for pipeline stages and annotated
arrows for data flow.  Intended as a single-column figure in the paper.

Pipeline stages (per methods.tex):
  1. Horizon circle detection           → optical center, initial f
  2. Rotation estimate (density/MW)     → rho seed
  3. Pattern-match hypothesis-and-verify → n_match, (f, rho), mirror?
  4. Score-map fallback                 → alternate path if low matches
  5. Joint f + k1 estimation             → refined scale + distortion
  6. Multi-phase guided refinement (A–E) → final model
  7. Projection-type selection          → equidistant / stereographic / ...

Output:
  benchmark/results/paper_figures_v3/fig_pipeline_flowchart.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = Path("benchmark/results/paper_figures_v3")


def _box(ax, xy, width, height, text, facecolor="#dbe9f4",
         edgecolor="#1f4e79", textsize=9, bold=False):
    x, y = xy
    p = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2, facecolor=facecolor, edgecolor=edgecolor)
    ax.add_patch(p)
    ax.text(x + width / 2, y + height / 2, text,
            ha="center", va="center", fontsize=textsize,
            weight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1, label=None, rad=0.0, color="#555555"):
    arr = mpatches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0, color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(mx + 0.05, my, label, fontsize=8, color=color,
                ha="left", va="center")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Colours
    C_INPUT = "#ffeab0"     # pale yellow
    C_STAGE = "#dbe9f4"     # pale blue
    C_ALT = "#f3d4d4"       # pale red (fallback)
    C_OUT = "#d6e9c6"       # pale green
    EDGE = "#1f4e79"

    # Inputs at top
    _box(ax, (0.2, 12.0), 2.2, 0.7,
         "FITS frame", facecolor=C_INPUT, edgecolor=EDGE, bold=True)
    _box(ax, (2.5, 12.0), 2.2, 0.7,
         "(lat, lon, time)", facecolor=C_INPUT, edgecolor=EDGE, bold=True)
    _box(ax, (4.8, 12.0), 2.0, 0.7,
         "Hipparcos / BSC5", facecolor=C_INPUT, edgecolor=EDGE, bold=True)

    # Stage 1: horizon detection
    _box(ax, (0.5, 10.5), 6.0, 0.9,
         "1.  Horizon-circle detection\n"
         "$(c_x, c_y)$, initial $f$ from radius",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Stage 2: rotation estimate
    _box(ax, (0.5, 9.2), 6.0, 0.7,
         "2.  Density / Milky-Way rotation estimate  →  $\\rho$ seed",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Stage 3: pattern-match (with mirror test)
    _box(ax, (0.5, 7.6), 6.0, 1.2,
         "3.  Hypothesis-and-verify pattern match\n"
         "(both mirror orientations, $\\phi_0 \\times c_x \\times c_y$ grid,\n"
         " choose best by $n_\\mathrm{match}$)",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Stage 4: score-map fallback (side path)
    _box(ax, (0.5, 6.2), 6.0, 0.9,
         "3$'$.  Score-map fallback  (if pattern-match $< $ threshold)",
         facecolor=C_ALT, edgecolor="#a13a3a")

    # Stage 5: joint f + k1
    _box(ax, (0.5, 4.9), 6.0, 0.9,
         "4.  Joint $(f, k_1)$ from horizon radius + mid-alt matches",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Stage 6: multi-phase refinement
    _box(ax, (0.5, 3.1), 6.0, 1.4,
         "5.  Multi-phase guided refinement\n"
         " A  mid-altitude geometry    B  widen to full disk\n"
         " C  full-field distortion    D/E  full catalog pass",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Stage 7: projection-type selection
    _box(ax, (0.5, 1.8), 6.0, 0.9,
         "6.  Projection-type selection\n"
         "(equidistant / stereographic / equisolid / ortho)",
         facecolor=C_STAGE, edgecolor=EDGE)

    # Output
    _box(ax, (0.5, 0.2), 6.0, 1.2,
         "Instrument model (JSON)\n"
         "$(c_x, c_y, \\phi_0, \\alpha_0, \\rho, f, k_1, k_2,$ "
         "mirror, projection, $\\mathrm{ZP})$",
         facecolor=C_OUT, edgecolor="#2f6a2f", bold=True)

    # Arrows (vertical flow)
    _arrow(ax, 3.5, 12.0, 3.5, 11.4)
    _arrow(ax, 3.5, 10.5, 3.5, 9.9)
    _arrow(ax, 3.5, 9.2, 3.5, 8.8)
    _arrow(ax, 3.5, 7.6, 3.5, 7.1)     # 3 → fallback
    _arrow(ax, 3.5, 6.2, 3.5, 5.8)     # fallback → stage 4
    _arrow(ax, 3.5, 4.9, 3.5, 4.5)
    _arrow(ax, 3.5, 3.1, 3.5, 2.7)
    _arrow(ax, 3.5, 1.8, 3.5, 1.4)

    # Title
    ax.text(3.5, 12.85, "Blind instrument-fit pipeline",
            ha="center", va="bottom", fontsize=13, weight="bold")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_pipeline_flowchart.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
