"""Per-frame clear-sky fraction above 30 deg altitude, for the nightly
observing-efficiency simulation.  Re-solves each frame from the known
model (transmission now normalized to model reference ZP) and records
the observability mask statistics restricted to alt >= 30 deg.

Output CSV: mjd, iso, n_matched, clear_frac30, has_clear30, clear_frac_all
"""
import csv
import multiprocessing as mp
from pathlib import Path

ROOT = Path("/stars/src/allclear")
DATA = ROOT / "benchmark/data/apicam_drift_nightly"
# Reference = best model from BEFORE this night (2019-02-24 seasonal
# blind solve) with the clear-sky zeropoint carried over.  Solved with
# fix_center=True so the optical centre stays pinned and per-frame
# pointing drift is absorbed by az0/alt0/rho (kills the zenith sawtooth).
MODEL = ROOT / "benchmark/results/apicam_nightly/model_prenight.json"
OUTCSV = ROOT / "benchmark/results/apicam_nightly/clearfrac_night.csv"
THRESHOLD = 0.7
ALT_MIN = 30.0
FIX_CENTER = True


def one(fpath):
    import numpy as np
    from allclear.cli import _load_frame
    from allclear.instrument import InstrumentModel
    from allclear.solver import fast_solve
    from allclear.transmission import (compute_transmission,
                                       interpolate_transmission)
    inst = InstrumentModel.load(str(MODEL))
    camera = inst.to_camera_model()
    try:
        data, meta, cat, det, _ = _load_frame(str(fpath), inst.site_lat,
                                              inst.site_lon)
        mjd = float(meta["obs_time"].mjd)
        iso = meta["obs_time"].iso
        if inst.mirrored:
            data = data[:, ::-1]
            det["x"] = (data.shape[1] - 1) - np.asarray(det["x"],
                                                        dtype=np.float64)
        result = fast_solve(data, det, cat, camera, guided=True,
                            refine=True, obscuration=inst.obscuration,
                            fix_center=FIX_CENTER)
        use_det = (result.guided_det_table
                   if result.guided_det_table is not None
                   and len(result.guided_det_table) > 0 else det)
        nm = int(result.n_matched)
        if nm < 3:
            return (mjd, iso, nm, 0.0, 0, 0.0)
        ref_zp = inst.photometric_zeropoint or None
        az, alt, trans, zp = compute_transmission(
            use_det, cat, result.matched_pairs, result.camera_model,
            image=data, reference_zeropoint=ref_zp,
            obscuration=inst.obscuration)
        tmap = interpolate_transmission(az, alt, trans)
        clear = tmap.transmission >= THRESHOLD          # NaN -> False
        alt_sel = tmap.alt_grid >= ALT_MIN
        block30 = clear[alt_sel, :]
        cf30 = float(np.mean(block30)) if block30.size else 0.0
        cf_all = float(np.mean(clear)) if clear.size else 0.0
        has30 = int(bool(np.any(block30)))
        return (mjd, iso, nm, cf30, has30, cf_all)
    except Exception as e:
        return (-1.0, f"ERR {e}", 0, 0.0, 0, 0.0)


if __name__ == "__main__":
    frames = sorted(DATA.glob("*.fits"))
    print(f"Extracting clear-fraction (alt>={ALT_MIN}, thr={THRESHOLD}) "
          f"for {len(frames)} frames", flush=True)
    rows = []
    done = 0
    with mp.Pool(12) as p:
        for r in p.imap_unordered(one, frames):
            done += 1
            rows.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(frames)}", flush=True)
    rows = [r for r in rows if r[0] > 0]
    rows.sort(key=lambda x: x[0])
    with open(OUTCSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mjd", "iso", "n_matched", "clear_frac30",
                    "has_clear30", "clear_frac_all"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUTCSV}", flush=True)
