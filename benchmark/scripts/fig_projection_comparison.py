"""Schematic comparison of the four supported fisheye projections.

Left panel: radial mapping r(theta) for equidistant, equisolid,
stereographic, and orthographic projections, with a common focal
length.  Right panel: how a uniform altitude grid maps to the image
plane under each projection.

Output:
  benchmark/results/paper_figures_v3/fig_projection_comparison.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("benchmark/results/paper_figures_v3")


def r_equidistant(theta, f):
    return f * theta


def r_equisolid(theta, f):
    return 2 * f * np.sin(theta / 2)


def r_stereographic(theta, f):
    return 2 * f * np.tan(theta / 2)


def r_orthographic(theta, f):
    return f * np.sin(theta)


PROJECTIONS = [
    ("equidistant",  r_equidistant,  "#1f77b4", "solid"),
    ("equisolid",    r_equisolid,    "#2ca02c", "dashed"),
    ("stereographic", r_stereographic, "#d62728", "dashdot"),
    ("orthographic", r_orthographic, "#9467bd", "dotted"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = 1.0  # normalised focal length
    theta = np.linspace(0, np.pi / 2, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: r vs theta (zenith angle)
    ax = axes[0]
    for name, fn, color, ls in PROJECTIONS:
        ax.plot(np.degrees(theta), fn(theta, f), label=name,
                color=color, ls=ls, lw=2)
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel(r"Image radius $r / f$")
    ax.set_title("Radial mapping $r(\\theta)$")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, np.pi / 2 + 0.3)

    # Right: altitude ring grid projected through each projection.
    ax = axes[1]
    alts = np.array([0, 15, 30, 45, 60, 75])  # deg
    theta_rings = np.radians(90 - alts)
    az = np.linspace(0, 2 * np.pi, 360)

    # Overlay each projection's altitude rings at unique radii; offset
    # the centres horizontally so the four overlap doesn't confuse.
    offsets = [(-1.8, 1.0), (1.8, 1.0), (-1.8, -1.0), (1.8, -1.0)]
    for (name, fn, color, ls), (ox, oy) in zip(PROJECTIONS, offsets):
        for tr in theta_rings:
            r = fn(tr, f)
            x = ox + r * np.cos(az)
            y = oy + r * np.sin(az)
            ax.plot(x, y, color=color, ls=ls, lw=1.0, alpha=0.8)
        # centre label
        ax.text(ox, oy - np.pi / 2 - 0.2, name, color=color,
                ha="center", va="top", fontsize=10, weight="bold")
        # horizon reference circle (theta=pi/2)
        r_horizon = fn(np.pi / 2, f)
        x = ox + r_horizon * np.cos(az)
        y = oy + r_horizon * np.sin(az)
        ax.plot(x, y, color=color, ls="-", lw=2.0, alpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title(
        "Altitude grid (alt = 0, 15, 30, 45, 60, 75$^\\circ$)\n"
        "under each projection, same $f$")
    ax.set_xlabel("image x / f")
    ax.set_ylabel("image y / f")
    ax.grid(alpha=0.2)

    fig.suptitle("Fisheye projection comparison", fontsize=13, y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"fig_projection_comparison.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
