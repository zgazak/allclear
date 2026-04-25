"""Build the APICAM obscuration mask from seasonal per-star observations.

Aggregates the per-star detection outcomes (produced by
``run_obscuration_pass.py``) in **sky coordinates (az, alt)** rather
than pixel coordinates.  Sky-space aggregation is essential for
cameras whose mount drifts between frames: fixed obstructions (dome,
trees, horizon terrain) stay at the same (az, alt) even as their pixel
position shifts.  Pixel-space aggregation smears them.

Saturated bright stars (vmag < 1.5) are excluded because they clip the
16-bit sensor and fail matching for photometric, not obscuration,
reasons.  Cloudy frames are excluded via a clear-fraction gate.

Output:
  benchmark/results/obscuration/obscuration_map.json  — ObscurationMask
  benchmark/results/obscuration/obscuration_map.png
  benchmark/results/obscuration/obscuration_map.pdf

Run:
    uv run python benchmark/scripts/build_obscuration_map.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allclear.instrument import InstrumentModel
from allclear.obscuration import ObscurationMask, build_from_observations
from allclear.utils import load_image

OBS_CSV = Path("benchmark/results/obscuration/per_star_observations.csv")
EXAMPLE_FRAME_MODEL = Path(
    "benchmark/results/apicam_seasonal_blind/"
    "APICAM.2019-06-06T00:59:45.000_model.json")
EXAMPLE_FRAME_IMG = Path(
    "benchmark/data/apicam_drift_seasonal/"
    "APICAM.2019-06-06T00:59:45.000.fits")

OUT_JSON = Path("benchmark/results/obscuration/obscuration_map.json")
OUT_PNG = Path("benchmark/results/obscuration/obscuration_map.png")
OUT_PDF = Path("benchmark/results/obscuration/obscuration_map.pdf")

CLEAR_GATE = 0.70
VMAG_MIN = 1.5    # exclude saturated bright stars (Sirius / Canopus / Vega)
VMAG_MAX = 6.0    # exclude faint end (below detection threshold even clear)
MIN_VISITS = 8
AZ_STEP_DEG = 2.0
ALT_STEP_DEG = 2.0


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load_observations():
    az, alt, det, cf, vmag = [], [], [], [], []
    frames = set()
    with OBS_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            az.append(_float(r["az_deg"]))
            alt.append(_float(r["alt_deg"]))
            det.append(int(r["detected"]))
            cf.append(_float(r["clear_fraction"]))
            vmag.append(_float(r["vmag"]))
            frames.add(r["frame"])
    return (
        np.array(az), np.array(alt),
        np.array(det, dtype=np.int32),
        np.array(cf), np.array(vmag),
        len(frames),
    )


def main():
    az, alt, det, cf, vmag, n_frames = load_observations()
    print(f"Loaded {len(az):,} per-star observations from {n_frames} frames")

    mask = build_from_observations(
        az_deg=az, alt_deg=alt, detected=det,
        clear_fraction=cf, vmag=vmag,
        clear_gate=CLEAR_GATE,
        vmag_min=VMAG_MIN, vmag_max=VMAG_MAX,
        min_visits=MIN_VISITS,
        az_step_deg=AZ_STEP_DEG, alt_step_deg=ALT_STEP_DEG,
        n_frames=n_frames,
    )

    n_bins = mask.weight.size
    n_filled = int(np.isfinite(mask.weight).sum())
    n_obscured = int(((mask.weight < 0.3) & np.isfinite(mask.weight)).sum())
    print(f"Grid: {len(mask.alt_edges_deg)-1} alt × "
          f"{len(mask.az_edges_deg)-1} az = {n_bins:,} bins")
    print(f"  Filled (≥ {MIN_VISITS} visits): {n_filled:,} "
          f"({100.0*n_filled/n_bins:.1f}%)")
    print(f"  Obscured (weight < 0.3): {n_obscured:,} "
          f"({100.0*n_obscured/max(n_filled,1):.1f}% of filled)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    mask.save(OUT_JSON)
    print(f"Wrote: {OUT_JSON}")

    _render(mask)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")


def _render(mask):
    """Render a three-panel diagnostic plot of the mask."""
    alt_centers = 0.5 * (mask.alt_edges_deg[:-1] + mask.alt_edges_deg[1:])
    az_centers = 0.5 * (mask.az_edges_deg[:-1] + mask.az_edges_deg[1:])
    visits = mask.n_visits
    weight = mask.weight

    fig = plt.figure(figsize=(14, 5.2))

    # Panels 1 & 2 use a polar (az, zenith-angle) view matching the
    # pixel-projected panel's left/right orientation for APICAM.
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    AZ, Z = np.meshgrid(np.radians(az_centers), 90.0 - alt_centers)
    im = ax1.pcolormesh(AZ, Z, visits, cmap="viridis", shading="auto")
    ax1.set_theta_zero_location("S")
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(0)
    ax1.set_yticks([0, 30, 60, 90])
    ax1.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax1.set_title(f"Clear-sky visits per bin\n(gate $f_c$≥{CLEAR_GATE},"
                  f" {VMAG_MIN}<V<{VMAG_MAX})")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.08)

    # Panel 2: weight (detection fraction)
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    im = ax2.pcolormesh(AZ, Z, weight, cmap="RdYlGn", vmin=0, vmax=1,
                        shading="auto")
    ax2.set_theta_zero_location("S")
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(0)
    ax2.set_yticks([0, 30, 60, 90])
    ax2.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax2.set_title("Detection fraction\n(1 − obscuration)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.08, label="weight")

    # Panel 3: sky-space mask projected onto the example frame
    ax4 = fig.add_subplot(1, 3, 3)
    try:
        inst = InstrumentModel.load(EXAMPLE_FRAME_MODEL)
        camera = inst.to_camera_model()
        data, _ = load_image(str(EXAMPLE_FRAME_IMG))
        if inst.mirrored:
            data = data[:, ::-1]
        finite = np.isfinite(data)
        vmin = float(np.nanpercentile(data[finite], 2))
        vmax = float(np.nanpercentile(data[finite], 99.5))
        ax4.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        pixel_w = mask.project_to_pixel(camera, data.shape)
        overlay = np.where(pixel_w < 0.3, 1.0 - pixel_w, np.nan)
        ax4.imshow(overlay, origin="lower", cmap="Reds",
                   vmin=0, vmax=1, alpha=0.5)
        ax4.set_title("Mask projected onto\nexample frame")
        ax4.set_xticks([])
        ax4.set_yticks([])
        for spine in ax4.spines.values():
            spine.set_visible(False)
    except Exception as exc:
        ax4.text(0.5, 0.5, f"example frame unavailable:\n{exc}",
                 ha="center", va="center", transform=ax4.transAxes)

    fig.suptitle(
        "APICAM sky-space obscuration mask (2019 seasonal solve)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
