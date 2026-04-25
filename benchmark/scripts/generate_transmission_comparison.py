#!/usr/bin/env python3
"""Generate a 2-panel comparison figure for the paper.

LEFT:  "Before" -- stale zeropoint (11.875), no measured probing
       (old behavior: all matched stars show inflated transmission,
        unmatched stars get binary zero -- result looks all-green)
RIGHT: "After"  -- auto-upgraded zeropoint, measured probing
       (new behavior: cloud structure visible through graded transmission)

Usage:
    uv run python benchmark/scripts/generate_transmission_comparison.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- paths ----
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FRAME_PATH = (ROOT / "benchmark" / "data" / "apicam_drift_nightly"
              / "APICAM.2019-06-15T01:36:15.000.fits")
CLEAR_FRAME_PATH = (ROOT / "benchmark" / "data" / "apicam_drift_nightly"
                    / "APICAM.2019-06-15T00:02:16.000.fits")
MODEL_PATH = ROOT / "benchmark" / "solutions" / "apicam_jun13.json"

OUT_DIR = ROOT / "benchmark" / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_and_solve(frame_path, inst, camera):
    """Load a frame, mirror if needed, and run fast_solve."""
    from allclear.cli import _load_frame
    from allclear.solver import fast_solve

    data, meta, cat, det, _ = _load_frame(
        str(frame_path), inst.site_lat, inst.site_lon)

    # Mirror to match instrument model
    if inst.mirrored:
        data = data[:, ::-1]
        det["x"] = (data.shape[1] - 1) - np.asarray(det["x"],
                                                      dtype=np.float64)

    result = fast_solve(data, det, cat, camera, guided=True, refine=True)
    log.info("  %s: %d matches, RMS=%.2f px",
             frame_path.name, result.n_matched, result.rms_residual)
    return data, meta, cat, det, result


def compute_before(data, cat, result, stale_zp):
    """Compute transmission the OLD way: stale zeropoint, no image probing."""
    from allclear.transmission import compute_transmission

    use_det = (result.guided_det_table
               if result.guided_det_table is not None
               and len(result.guided_det_table) > 0
               else None)

    # OLD behavior:
    #   1) reference_zeropoint = stale value (11.875)
    #   2) image=None  --> unmatched stars get binary zero (not measured)
    #   3) The auto-upgrade check still fires, but we bypass it by
    #      NOT passing the image, and passing image_shape so probing
    #      gives binary zeros.  However the key issue is the stale zp
    #      makes matched-star transmission >> 1, so the RBF interpolation
    #      paints everything green.
    #
    # Actually, to truly simulate the OLD behavior we must prevent the
    # auto-upgrade of zeropoint.  The auto-upgrade triggers when
    # frame_zeropoint > reference_zeropoint.  With stale_zp=11.875 and
    # frame_zp~14.3, the code upgrades.  To prevent this, we compute
    # transmission ourselves with the stale zp forced.

    det_table = use_det
    matched_pairs = result.matched_pairs
    camera_model = result.camera_model

    cat_vmag = np.array([float(cat["vmag_expected"][ci])
                         for _, ci in matched_pairs])
    cat_az = np.array([float(cat["az_deg"][ci])
                       for _, ci in matched_pairs])
    cat_alt = np.array([float(cat["alt_deg"][ci])
                        for _, ci in matched_pairs])

    # Get detection flux (from guided det table -- NOT image-based local bg)
    det_flux = np.array([float(det_table["flux"][di])
                         for di, _ in matched_pairs])

    valid = det_flux > 0
    inst_mag = np.full_like(det_flux, np.nan)
    inst_mag[valid] = -2.5 * np.log10(det_flux[valid])

    # Force the stale zeropoint (no auto-upgrade)
    zeropoint = stale_zp

    # Transmission per matched star
    transmission = np.full_like(det_flux, np.nan)
    transmission[valid] = 10 ** (-0.4 * (inst_mag[valid] + zeropoint
                                          - cat_vmag[valid]))

    # OLD behavior: NO unmatched-star probing.
    # The original code only had matched-star transmission; unmatched
    # stars were simply invisible.  With the stale zeropoint, all
    # matched stars have inflated transmission (>>1, clipped to 1.2),
    # so the RBF interpolation paints the entire sky uniformly green
    # regardless of actual cloud cover.  This is the key failure mode
    # that the measured-probing fix addresses.

    log.info("  BEFORE: zp=%.3f, matched trans median=%.2f, "
             "%d matched stars only (no probing)",
             zeropoint,
             float(np.nanmedian(transmission[valid])),
             int(np.sum(valid)))

    return cat_az, cat_alt, transmission, zeropoint


def compute_after(data, cat, result, ref_zp):
    """Compute transmission the NEW way: correct zeropoint + measured probing."""
    from allclear.transmission import compute_transmission

    use_det = (result.guided_det_table
               if result.guided_det_table is not None
               and len(result.guided_det_table) > 0
               else None)

    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
    )
    log.info("  AFTER:  zp=%.3f, matched trans median=%.2f, %d total points",
             zp, float(np.nanmedian(trans[:len(result.matched_pairs)])),
             len(trans))
    return az, alt, trans, zp


def render_panel(ax, image, camera_model, trans_az, trans_alt, trans_vals,
                 cat, matched_pairs, det_table, meta, lat, lon):
    """Render a single annotated transmission panel onto the given axes."""
    from allclear.plotting import (
        zscale, _overlay_transmission, _draw_altaz_grid, _draw_stars,
        _draw_planets,
    )

    ny, nx = image.shape
    ax.imshow(zscale(image), cmap="gray", origin="lower",
              extent=[0, nx, 0, ny])

    _overlay_transmission(ax, camera_model, trans_az, trans_alt,
                          trans_vals, nx, ny)
    _draw_altaz_grid(ax, camera_model, nx, ny)
    _draw_stars(ax, det_table, cat, matched_pairs, camera_model, nx, ny)

    if meta.get("obs_time") is not None and lat is not None:
        _draw_planets(ax, camera_model, meta["obs_time"], lat, lon, nx, ny)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def main():
    from allclear.instrument import InstrumentModel

    log.info("Loading instrument model from %s", MODEL_PATH)
    inst = InstrumentModel.load(MODEL_PATH)
    camera = inst.to_camera_model()
    stale_zp = inst.photometric_zeropoint  # 11.875
    log.info("  Stale (model) zeropoint: %.3f", stale_zp)

    # --- Solve the clear frame to get the correct zeropoint ---
    log.info("\nSolving clear frame for correct zeropoint...")
    clear_data, clear_meta, clear_cat, clear_det, clear_result = \
        load_and_solve(CLEAR_FRAME_PATH, inst, camera)

    from allclear.transmission import compute_transmission
    clear_use_det = (clear_result.guided_det_table
                     if clear_result.guided_det_table is not None
                     and len(clear_result.guided_det_table) > 0
                     else clear_det)
    _, _, _, clear_zp = compute_transmission(
        clear_use_det, clear_cat, clear_result.matched_pairs,
        clear_result.camera_model, image=clear_data,
        reference_zeropoint=stale_zp)
    log.info("  Clear frame zeropoint: %.3f", clear_zp)

    # --- Solve the target (cloudy) frame ---
    log.info("\nSolving target (cloudy) frame...")
    data, meta, cat, det, result = \
        load_and_solve(FRAME_PATH, inst, camera)

    use_det = (result.guided_det_table
               if result.guided_det_table is not None
               and len(result.guided_det_table) > 0
               else det)

    # --- "Before" transmission ---
    log.info("\nComputing BEFORE (stale zp, binary probing)...")
    before_az, before_alt, before_trans, before_zp = \
        compute_before(data, cat, result, stale_zp)

    # --- "After" transmission ---
    log.info("\nComputing AFTER (correct zp, measured probing)...")
    after_az, after_alt, after_trans, after_zp = \
        compute_after(data, cat, result, clear_zp)

    # --- Build 2-panel figure ---
    log.info("\nRendering 2-panel figure...")

    ny, nx = data.shape
    # PASP column width ~3.4in, full width ~7in
    # Two panels side by side at aspect ~nx/ny each
    aspect = nx / ny
    panel_w = 3.4  # inches per panel
    panel_h = panel_w / aspect
    fig_w = panel_w * 2 + 0.3  # small gap
    fig_h = panel_h + 0.45  # room for labels

    fig, (ax_before, ax_after) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05},
    )

    render_panel(ax_before, data, result.camera_model,
                 before_az, before_alt, before_trans,
                 cat, result.matched_pairs, use_det,
                 meta, inst.site_lat, inst.site_lon)

    render_panel(ax_after, data, result.camera_model,
                 after_az, after_alt, after_trans,
                 cat, result.matched_pairs, use_det,
                 meta, inst.site_lat, inst.site_lon)

    # Panel labels
    label_kw = dict(color="white", fontsize=11, fontweight="bold",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="black", alpha=0.8,
                              edgecolor="none"))
    ax_before.set_title("(a) Stale zeropoint, no unmatched probing",
                        color="white", fontsize=9, pad=3,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="black", alpha=0.8,
                                  edgecolor="none"))
    ax_after.set_title("(b) Auto-upgraded zeropoint, measured probing",
                       color="white", fontsize=9, pad=3,
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="black", alpha=0.8,
                                 edgecolor="none"))

    # Zeropoint annotations
    zp_kw = dict(color="white", fontsize=7, ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor="black", alpha=0.7,
                           edgecolor="none"))
    ax_before.text(nx * 0.02, ny * 0.02,
                   f"ZP = {before_zp:.3f}  (model)",
                   transform=ax_before.transData, **zp_kw)
    ax_after.text(nx * 0.02, ny * 0.02,
                  f"ZP = {after_zp:.3f}  (auto-upgraded)",
                  transform=ax_after.transData, **zp_kw)

    fig.patch.set_facecolor("black")

    # Save
    pdf_path = OUT_DIR / "transmission_comparison.pdf"
    png_path = OUT_DIR / "transmission_comparison.png"

    fig.savefig(str(pdf_path), dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="black")
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="black")
    plt.close(fig)

    log.info("\nSaved:")
    log.info("  PDF: %s", pdf_path)
    log.info("  PNG: %s", png_path)

    # Also save individual panels at higher resolution for flexibility
    for label, trans_data in [("before", (before_az, before_alt, before_trans)),
                               ("after", (after_az, after_alt, after_trans))]:
        from allclear.plotting import plot_frame
        panel_path = OUT_DIR / f"transmission_{label}.png"
        plot_frame(
            data, result.camera_model,
            det_table=use_det,
            cat_table=cat,
            matched_pairs=result.matched_pairs,
            show_grid=True,
            transmission_data=trans_data,
            obs_time=meta.get("obs_time"),
            lat_deg=inst.site_lat,
            lon_deg=inst.site_lon,
            output_path=str(panel_path),
        )
        log.info("  Panel: %s", panel_path)


if __name__ == "__main__":
    main()
