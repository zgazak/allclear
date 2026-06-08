#!/usr/bin/env python3
"""Night observing-efficiency simulation: AllClear target selection vs
blind random pointing, over a night of all-sky transparency maps.

Model of an observatory co-located with the all-sky camera:
  - Targets are sets of 6x 1 s frames at 8192x8192, 16-bit.
  - 30 s slew/settle between targets -> 36 s per target cycle.
  - Random pointing: draw az uniform[0,360), alt uniform[30,90].
  - BLIND  : observe at the drawn target regardless of sky.  Every cycle
             writes data; the clear fraction of those frames equals the
             sky's clear fraction above 30 deg at that time.
  - ALLCLEAR: read the transmission map; redraw (free) until a clear
             target is found; if the whole sky above 30 deg is cloudy,
             sit idle (collect nothing) for that cycle.  Every written
             frame is clear.

Sky truth = per-frame clear fraction above 30 deg (clearfrac CSV, nearest
frame in time to each 36 s cycle).  "Clear" = transmission >= 0.7.  We
assume the solved model is correct.

Accounting is expected-value (the mean over the random pointing), so the
result is deterministic and reproducible — no Monte-Carlo seed.

Inputs:
    clearfrac_night.csv  (from extract_nightly_clearfrac.py)
Outputs (next to the CSV):
    night_efficiency.png        cumulative useful/collected volume
    night_collection_rate.png   good/bad GB-rate per method + transparency

Usage:
    python sim_night_efficiency.py [clearfrac_csv] [out_dir]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Observation model ---
FRAME_PX = 8192
BIT_DEPTH = 16
FRAMES_PER_TARGET = 6
EXPOSURE_S = 6.0          # 6 x 1 s
SLEW_S = 30.0
CYCLE_S = EXPOSURE_S + SLEW_S            # 36 s per target
THRESHOLD_NOTE = 0.7                     # clear-sky threshold used upstream
ALT_MIN_NOTE = 30.0
BYTES_PER_FRAME = FRAME_PX * FRAME_PX * (BIT_DEPTH // 8)
GB_PER_FRAME = BYTES_PER_FRAME / 1e9              # decimal GB
GB_PER_TARGET = GB_PER_FRAME * FRAMES_PER_TARGET
FULL_RATE_GBH = GB_PER_TARGET / CYCLE_S * 3600.0  # GB/h if observing every cycle
BIN_MIN = 15.0                                    # rate-plot time bin


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    mjd = np.array([float(r["mjd"]) for r in rows])
    cf30 = np.array([float(r["clear_frac30"]) for r in rows])
    has30 = np.array([int(r["has_clear30"]) for r in rows], dtype=bool)
    order = np.argsort(mjd)
    return mjd[order], cf30[order], has30[order], rows


def build_cycles(mjd, cf30, has30):
    t0, t1 = mjd.min(), mjd.max()
    night_s = (t1 - t0) * 86400.0
    n_cycles = int(night_s // CYCLE_S)
    cycle_mjd = t0 + (np.arange(n_cycles) * CYCLE_S) / 86400.0
    idx = np.clip(np.searchsorted(mjd, cycle_mjd), 0, len(mjd) - 1)
    left = np.clip(idx - 1, 0, len(mjd) - 1)
    use_left = np.abs(cycle_mjd - mjd[left]) < np.abs(cycle_mjd - mjd[idx])
    nn = np.where(use_left, left, idx)
    hrs = (cycle_mjd - t0) * 24.0
    return hrs, cf30[nn], has30[nn], night_s, n_cycles


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1]
        / "results/apicam_nightly/clearfrac_night.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    mjd, cf30, has30, rows = load(csv_path)
    hrs, p_clear, sky_clear, night_s, n_cycles = build_cycles(mjd, cf30, has30)

    # Expected-value volumes per cycle
    blind_good = GB_PER_TARGET * p_clear           # clear frames written blind
    blind_bad = GB_PER_TARGET * (1.0 - p_clear)    # cloudy frames written blind
    ac_good = GB_PER_TARGET * sky_clear.astype(float)   # full rate iff any clear
    ac_bad = np.zeros(n_cycles)                    # AllClear never writes cloudy

    blind_vol = GB_PER_TARGET * n_cycles
    blind_useful = blind_good.sum()
    ac_vol = ac_good.sum()
    ac_useful = ac_vol

    # --- Report ---
    print(f"Night: {night_s/3600:.2f} h "
          f"({rows[0]['iso'][:19]} -> {rows[-1]['iso'][:19]} UT)")
    print(f"Frame: {FRAME_PX}x{FRAME_PX} {BIT_DEPTH}-bit = {GB_PER_FRAME:.4f} GB; "
          f"target = {FRAMES_PER_TARGET} frames = {GB_PER_TARGET:.4f} GB; "
          f"full rate = {FULL_RATE_GBH:.1f} GB/h")
    print(f"Cycle {CYCLE_S:.0f} s -> {n_cycles} cycles; "
          f"mean clear>{ALT_MIN_NOTE:.0f}deg = {p_clear.mean():.1%}; "
          f"fully-overcast cycles = {np.mean(~sky_clear):.1%}\n")
    f = "  {:<10s} {:>10s} {:>9s} {:>10s}"
    print(f.format("strategy", "collected", "%clear", "useful"))
    print("  " + "-" * 42)
    print("  {:<10s} {:>9.1f}G {:>8.1f}% {:>9.1f}G".format(
        "blind", blind_vol, 100*blind_useful/blind_vol, blind_useful))
    print("  {:<10s} {:>9.1f}G {:>8.1f}% {:>9.1f}G".format(
        "allclear", ac_vol, 100.0, ac_useful))
    print(f"\n  useful gain AllClear/blind = {ac_useful/blind_useful:.2f}x")
    print(f"  blind wasted (cloudy) = {blind_vol-blind_useful:.1f} GB "
          f"({100*(1-blind_useful/blind_vol):.1f}% of collected)")
    print(f"  AllClear idle (overcast) = {int(np.sum(~sky_clear))}/{n_cycles} "
          f"cycles ({100*np.mean(~sky_clear):.1f}%)")

    # ---------- Figure 1: cumulative good/bad buildup ----------
    cb_good = np.cumsum(blind_good)
    cb_bad = np.cumsum(blind_bad)
    ca_good = np.cumsum(ac_good)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    # AllClear: all good, zero bad
    ax.fill_between(hrs, 0, ca_good, color="#27ae60", alpha=0.18)
    ax.plot(hrs, ca_good, color="#27ae60", lw=2.8,
            label=f"AllClear — good/clear ({ac_useful:.0f} GB)")
    ax.plot(hrs, np.zeros_like(hrs), color="#27ae60", lw=1.4, ls=":",
            label="AllClear — bad/cloudy  ≡ 0 GB")
    # Blind: good and bad both accumulate
    ax.plot(hrs, cb_good, color="#16a085", lw=2.2, ls="--",
            label=f"Blind — good/clear ({blind_useful:.0f} GB)")
    ax.plot(hrs, cb_bad, color="#c0392b", lw=2.2, ls="--",
            label=f"Blind — bad/cloudy ({blind_vol-blind_useful:.0f} GB)")
    ax.plot(hrs, np.cumsum(np.full(n_cycles, GB_PER_TARGET)), color="#888",
            lw=1.4, label=f"Blind — total written ({blind_vol:.0f} GB)")
    ax.set_xlabel("Hours into night (UT from first frame)")
    ax.set_ylabel("Cumulative data volume (GB)")
    ax.set_ylim(bottom=0)
    ax.set_title("Cumulative good/bad data buildup: AllClear vs blind "
                 "pointing\n"
                 f"{FRAME_PX}x{FRAME_PX} 16-bit, 6x1 s/target, "
                 f"{CYCLE_S:.0f} s cadence")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "night_efficiency.png", dpi=130)

    # ---------- Figure 2: good/bad collection RATE + transparency ----------
    bin_h = BIN_MIN / 60.0
    nb = int(np.ceil(hrs.max() / bin_h))
    bidx = np.clip((hrs / bin_h).astype(int), 0, nb - 1)
    bin_centers = (np.arange(nb) + 0.5) * bin_h

    def binrate(vals):
        out = np.zeros(nb)
        for b in range(nb):
            sel = bidx == b
            if np.any(sel):
                out[b] = vals[sel].sum() / bin_h     # GB per hour
        return out

    bg, bb = binrate(blind_good), binrate(blind_bad)
    ag, ab = binrate(ac_good), binrate(ac_bad)
    transp = np.array([100*p_clear[bidx == b].mean() if np.any(bidx == b)
                       else np.nan for b in range(nb)])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(bin_centers, ag, color="#27ae60", lw=2.6,
            label="AllClear — good (clear) GB/h")
    ax.plot(bin_centers, ab, color="#27ae60", lw=1.6, ls=":",
            label="AllClear — bad (cloudy) GB/h  ≡ 0")
    ax.plot(bin_centers, bg, color="#16a085", lw=2.0, ls="--",
            label="Blind — good (clear) GB/h")
    ax.plot(bin_centers, bb, color="#c0392b", lw=2.0, ls="--",
            label="Blind — bad (cloudy) GB/h")
    ax.axhline(FULL_RATE_GBH, color="#888", lw=1, ls=":",
               label=f"full observing rate ({FULL_RATE_GBH:.0f} GB/h)")
    ax.set_xlabel("Hours into night (UT from first frame)")
    ax.set_ylabel("Data collection rate (GB / h)")
    ax.set_ylim(bottom=0)
    ax.set_title("Good vs bad data collection rate — AllClear vs blind\n"
                 f"{BIN_MIN:.0f}-min bins; '%clear sky' = transparency above "
                 f"{ALT_MIN_NOTE:.0f}° (transmission ≥ {THRESHOLD_NOTE})")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, transp, color="#2c3e50", lw=2.4, alpha=0.85,
             label="% sky transparency (>30°)")
    ax2.set_ylabel("% sky clear above 30°", color="#2c3e50")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="#2c3e50")

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "night_collection_rate.png", dpi=130)
    print(f"\nPlots:\n  {out_dir/'night_efficiency.png'}"
          f"\n  {out_dir/'night_collection_rate.png'}")


if __name__ == "__main__":
    main()
