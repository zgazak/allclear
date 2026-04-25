"""Operational monitor: wraps solve with model auto-update and history tracking.

Processes frames chronologically, updating the camera model after each good
solve so that subsequent frames start from a better initial guess.  Records
per-frame results to a JSONL history file for drift analysis.

Usage (Python API)::

    mon = OperationalMonitor("camera_dir/model.json")
    for frame in sorted(glob("data/*.fits")):
        result = mon.process_frame(frame)
        print(f"{result.quality}: {result.n_matched} matches, RMS={result.rms:.2f}")

Usage (CLI)::

    allclear monitor --model model.json --frames "data/*.fits"
"""

import json
import logging
import math
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .instrument import InstrumentModel
from .solver import fast_solve

log = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Configurable quality gates for model update decisions."""
    min_matched_good: int = 1500
    max_rms_good: float = 2.5
    min_matched_bad: int = 500
    max_rms_refit: float = 3.0
    min_clear_fraction_refit: float = 0.5
    # Snapshot trigger: parameter drift from last snapshot
    snapshot_dcx: float = 5.0
    snapshot_dcy: float = 5.0
    snapshot_dalt_deg: float = 0.5
    snapshot_df_px: float = 3.0


@dataclass
class FrameResult:
    """Per-frame result from the operational monitor."""
    frame_path: str = ""
    timestamp: str = ""
    n_matched: int = 0
    n_expected: int = 0
    rms: float = 0.0
    status: str = ""
    clear_fraction: float = -1.0
    # Pointing deltas from ORIGINAL model
    dcx: float = 0.0
    dcy: float = 0.0
    dalt_deg: float = 0.0
    drho_deg: float = 0.0
    df_px: float = 0.0
    # Quality assessment
    quality: str = ""         # good, marginal, bad, refit_needed
    model_updated: bool = False
    notes: str = ""


class OperationalMonitor:
    """Process frames with automatic model tracking and update.

    The monitor wraps ``fast_solve`` with three additions:
    1. **Model management**: loads model_latest.json if available,
       falls back to the original model.json.
    2. **Auto-update**: after a good solve, writes refined pointing
       to model_latest.json for the next frame.
    3. **History**: appends a compact JSON record per frame to
       solve_history.jsonl.
    """

    def __init__(self, model_path, thresholds=None, output_dir=None):
        """
        Parameters
        ----------
        model_path : str or Path
            Path to the original instrument model JSON.
        thresholds : QualityThresholds, optional
            Quality gates (uses defaults if omitted).
        output_dir : str or Path, optional
            Directory for model_latest.json, snapshots, and history.
            Defaults to the same directory as model_path.
        """
        self.model_path = Path(model_path)
        self.thresholds = thresholds or QualityThresholds()
        self.output_dir = Path(output_dir) if output_dir else self.model_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load original model (never modified)
        self.original_inst = InstrumentModel.load(self.model_path)
        self.original_camera = self.original_inst.to_camera_model()

        # Load or initialize current model
        self._latest_path = self.output_dir / "model_latest.json"
        if self._latest_path.exists():
            self.current_inst = InstrumentModel.load(self._latest_path)
        else:
            self.current_inst = InstrumentModel.load(self.model_path)

        self._history_path = self.output_dir / "solve_history.jsonl"
        self._last_snapshot_inst = self.current_inst
        self._n_processed = 0
        self._n_updated = 0
        # Cache last solve for deferred image saving
        self._last_solve = None
        self._last_data = None
        self._last_cat = None
        self._last_meta = None
        self._last_fpath = None

    @property
    def current_camera(self):
        return self.current_inst.to_camera_model()

    def process_frame(self, frame_path, save_image_dir=None):
        """Process a single frame: solve, assess, maybe update model.

        Parameters
        ----------
        frame_path : str or Path
            Path to FITS image.
        save_image_dir : Path, optional
            If set, save solved overlay image here.

        Returns
        -------
        FrameResult
        """
        from .cli import _load_frame

        fpath = Path(frame_path)
        result = FrameResult(frame_path=fpath.name)

        # Load frame
        try:
            data, meta, cat, det, _ = _load_frame(
                str(fpath), self.original_inst.site_lat,
                self.original_inst.site_lon,
            )
        except Exception as e:
            log.warning("SKIP %s: %s", fpath.name, e)
            result.quality = "bad"
            result.notes = f"load failed: {e}"
            self._append_history(result)
            return result

        # Mirror if needed
        if self.original_inst.mirrored:
            data = data[:, ::-1]
            det["x"] = (data.shape[1] - 1) - np.asarray(det["x"], dtype=np.float64)

        result.timestamp = meta["obs_time"].datetime.isoformat()

        # Solve using CURRENT (possibly updated) model
        camera = self.current_camera
        solve = fast_solve(data, det, cat, camera, guided=True,
                           refit_rotation=False,
                           obscuration=self.original_inst.obscuration)

        # Cache for deferred image saving
        self._last_solve = solve
        self._last_data = data
        self._last_cat = cat
        self._last_meta = meta
        self._last_fpath = fpath

        result.n_matched = solve.n_matched
        result.n_expected = solve.n_expected
        result.rms = solve.rms_residual
        result.status = solve.status

        # Compute pointing deltas vs ORIGINAL model (stable reference)
        ref = self.original_camera
        m = solve.camera_model
        result.dcx = m.cx - ref.cx
        result.dcy = m.cy - ref.cy
        result.dalt_deg = math.degrees(m.alt0 - ref.alt0)
        result.drho_deg = math.degrees(m.rho - ref.rho)
        result.df_px = m.f - ref.f

        # Compute transmission / clear fraction
        result.clear_fraction = self._compute_clear_fraction(
            data, solve, cat, meta)

        # Assess quality
        result.quality, result.notes = self._assess_quality(result)

        # Update model if good
        if result.quality == "good":
            self._update_model(solve, meta)
            result.model_updated = True

        # Save image if requested
        if save_image_dir is not None:
            self._save_image(data, solve, cat, meta, fpath, save_image_dir)

        self._n_processed += 1
        self._append_history(result)
        return result

    def _assess_quality(self, r):
        """Classify solve quality. Returns (label, notes)."""
        t = self.thresholds

        if r.n_matched < t.min_matched_bad or r.status != "ok":
            return "bad", f"n={r.n_matched}, status={r.status}"

        if (r.n_matched >= t.min_matched_good
                and r.rms <= t.max_rms_good
                and r.status == "ok"):
            # Check refit trigger: good match count but high RMS
            # (can happen when distortion has drifted)
            return "good", ""

        # Marginal — check if refit is needed
        if (r.rms > t.max_rms_refit
                and r.clear_fraction > t.min_clear_fraction_refit):
            return "refit_needed", (
                f"clear sky ({r.clear_fraction:.0%}) but RMS={r.rms:.2f}px — "
                f"consider re-running instrument-fit")

        return "marginal", f"n={r.n_matched}, rms={r.rms:.2f}"

    def _update_model(self, solve, meta):
        """Update model_latest.json with refined pointing from a good solve."""
        m = solve.camera_model

        # Update pointing params, preserve everything else from current model
        self.current_inst.focal_length_px = m.f
        self.current_inst.center_x = m.cx
        self.current_inst.center_y = m.cy
        self.current_inst.alt0_deg = float(np.degrees(m.alt0))
        self.current_inst.roll_deg = float(np.degrees(m.rho))
        # Preserve k1, k2, az0 from original (not refined by fast_solve)

        self.current_inst.n_stars_matched = solve.n_matched
        self.current_inst.rms_residual_px = solve.rms_residual
        self.current_inst.fit_timestamp = (
            datetime.now(timezone.utc).isoformat())

        self.current_inst.save(self._latest_path)
        self._n_updated += 1

        # Maybe create snapshot
        self._maybe_snapshot()

    def _maybe_snapshot(self):
        """Save a timestamped snapshot if drift from last snapshot is large."""
        t = self.thresholds
        cur = self.current_inst
        prev = self._last_snapshot_inst

        drift_cx = abs(cur.center_x - prev.center_x)
        drift_cy = abs(cur.center_y - prev.center_y)
        drift_alt = abs(cur.alt0_deg - prev.alt0_deg)
        drift_f = abs(cur.focal_length_px - prev.focal_length_px)

        if (drift_cx > t.snapshot_dcx or drift_cy > t.snapshot_dcy
                or drift_alt > t.snapshot_dalt_deg
                or drift_f > t.snapshot_df_px):
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            snap_path = self.output_dir / f"model_{ts}.json"
            self.current_inst.save(snap_path)
            self._last_snapshot_inst = InstrumentModel.load(snap_path)
            log.info("Snapshot saved: %s", snap_path.name)

    def _compute_clear_fraction(self, data, solve, cat, meta):
        """Compute fraction of sky that is clear (transmission > 0.7)."""
        if solve.n_matched < 3 or solve.guided_det_table is None:
            return -1.0
        try:
            from .transmission import compute_transmission, interpolate_transmission
            ref_zp = self.original_inst.photometric_zeropoint or None
            az, alt, trans, zp = compute_transmission(
                solve.guided_det_table, cat, solve.matched_pairs,
                solve.camera_model, image=data, reference_zeropoint=ref_zp,
                obscuration=self.original_inst.obscuration,
            )
            if len(trans) == 0:
                return -1.0
            grid = interpolate_transmission(az, alt, trans)
            valid = np.isfinite(grid)
            if valid.sum() == 0:
                return -1.0
            return float(np.mean(grid[valid] > 0.7))
        except Exception:
            return -1.0

    def _save_image_from_last(self, frame_path, out_dir):
        """Save image from the most recent process_frame call (no re-solve)."""
        if self._last_solve is not None and self._last_fpath == Path(frame_path):
            self._save_image(self._last_data, self._last_solve,
                             self._last_cat, self._last_meta,
                             self._last_fpath, out_dir)

    def _save_image(self, data, solve, cat, meta, fpath, out_dir):
        """Save annotated solved + transmission images."""
        try:
            from .cli import _save_annotated_image
            from .transmission import compute_transmission
            use_det = solve.guided_det_table if (
                solve.guided_det_table is not None
                and len(solve.guided_det_table) > 0) else None

            out_path = Path(out_dir) / (fpath.stem + "_solved.png")
            _save_annotated_image(
                data, solve.camera_model, use_det, cat, str(out_path),
                matched_pairs=solve.matched_pairs,
                obs_time=meta["obs_time"],
                lat=self.original_inst.site_lat,
                lon=self.original_inst.site_lon,
            )

            # Transmission overlay
            if solve.n_matched >= 3 and use_det is not None:
                ref_zp = self.original_inst.photometric_zeropoint or None
                az, alt, trans, zp = compute_transmission(
                    use_det, cat, solve.matched_pairs, solve.camera_model,
                    image=data, reference_zeropoint=ref_zp,
                    obscuration=self.original_inst.obscuration,
                )
                trans_path = Path(out_dir) / (fpath.stem + "_transmission.png")
                _save_annotated_image(
                    data, solve.camera_model, use_det, cat, str(trans_path),
                    matched_pairs=solve.matched_pairs,
                    transmission_data=(az, alt, trans),
                    obs_time=meta["obs_time"],
                    lat=self.original_inst.site_lat,
                    lon=self.original_inst.site_lon,
                )
        except Exception as e:
            log.warning("Image save failed for %s: %s", fpath.name, e)

    def _append_history(self, r):
        """Append a compact JSON record to the history file."""
        record = {
            "frame": r.frame_path, "ts": r.timestamp,
            "n": r.n_matched, "rms": round(r.rms, 3),
            "status": r.status, "clear": round(r.clear_fraction, 3),
            "dcx": round(r.dcx, 2), "dcy": round(r.dcy, 2),
            "dalt": round(r.dalt_deg, 4), "drho": round(r.drho_deg, 4),
            "df": round(r.df_px, 2),
            "q": r.quality, "upd": r.model_updated,
        }
        with open(self._history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_summary(self):
        """Return summary statistics from history."""
        if not self._history_path.exists():
            return {}
        records = []
        with open(self._history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return {}
        rms_vals = [r["rms"] for r in records if r.get("rms", 0) > 0]
        n_vals = [r["n"] for r in records if r.get("n", 0) > 0]
        qualities = [r.get("q", "") for r in records]
        return {
            "n_frames": len(records),
            "n_good": qualities.count("good"),
            "n_marginal": qualities.count("marginal"),
            "n_bad": qualities.count("bad"),
            "n_refit": qualities.count("refit_needed"),
            "n_model_updates": sum(1 for r in records if r.get("upd")),
            "rms_median": float(np.median(rms_vals)) if rms_vals else 0,
            "rms_range": (min(rms_vals), max(rms_vals)) if rms_vals else (0, 0),
            "matches_median": float(np.median(n_vals)) if n_vals else 0,
        }
