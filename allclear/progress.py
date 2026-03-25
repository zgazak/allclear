"""Dashboard progress display for allclear instrument-fit.

Activated with --dashboard flag.  Shows a live-updating view of the
pipeline stages, with progress bars for long steps and a final
summary box.  Falls back to plain text if stdout is not a TTY.
"""

import sys
import time

# ANSI escape codes
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
WHITE = "\033[97m"
ERASE_LINE = "\033[K"

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

TOTAL_STEPS = 9
STEP_NAMES = {
    1: "Horizon circle",
    2: "Rotation estimate",
    3: "Blind pattern match",
    4: "f/k1 estimation",
    5: "Projection search",
    6: "Guided refinement",
    7: "Sweep backup",
    8: "Validation",
    9: "Residual diagnostics",
}


class ProgressDisplay:
    """Live dashboard for instrument-fit pipeline progress.

    Usage: pass as ``progress=`` callback to ``instrument_fit_pipeline``.
    The pipeline calls ``progress(event, **kwargs)`` at each stage.
    """

    def __init__(self):
        self.t0 = time.time()
        self.step_t0 = None
        self.tty = sys.stdout.isatty()
        self.spin_idx = 0
        self.best_n = 0
        self.best_rms = 999.0
        self.best_f = 0.0
        self.width = 72
        try:
            import shutil
            self.width = min(100, max(60, shutil.get_terminal_size().columns))
        except Exception:
            pass

    def __call__(self, event, **kw):
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(**kw)

    # -- Helpers --

    def _elapsed(self):
        if self.step_t0:
            return time.time() - self.step_t0
        return 0.0

    def _total_elapsed(self):
        return time.time() - self.t0

    def _spin(self):
        ch = SPINNER[self.spin_idx % len(SPINNER)]
        self.spin_idx += 1
        return ch

    def _step_label(self, num):
        name = STEP_NAMES.get(num, "")
        return f"  {DIM}[{num}/{TOTAL_STEPS}]{RESET} {name}"

    def _dots(self, label, detail, elapsed=None):
        """Print a completed step: label ........... detail  [time]"""
        time_str = f"  {DIM}[{elapsed:.1f}s]{RESET}" if elapsed else ""
        plain_len = len(label) + len(detail) + 8
        n_dots = max(3, self.width - plain_len - 10)
        dots = f" {DIM}{'·' * n_dots}{RESET} "
        print(f"{label}{dots}{detail}{time_str}")

    def _overwrite(self, text):
        if self.tty:
            sys.stdout.write(f"\r{ERASE_LINE}{text}")
            sys.stdout.flush()

    def _progress_bar(self, fraction, width=20):
        filled = int(fraction * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"{DIM}[{RESET}{CYAN}{bar}{RESET}{DIM}]{RESET}"

    def _update_best(self, n, rms, f=None):
        if n > self.best_n or (n == self.best_n and rms < self.best_rms):
            self.best_n = n
            self.best_rms = rms
            if f:
                self.best_f = f
            return True
        return False

    # -- Event handlers --

    def _on_start(self, nx=0, ny=0, obs_time=None, n_cat=0, n_det=0, **kw):
        print()
        print(f"  {BOLD}AllClear instrument-fit{RESET}")
        info_parts = []
        if nx and ny:
            info_parts.append(f"{nx}×{ny}")
        if obs_time:
            info_parts.append(str(obs_time)[:19])
        if n_cat:
            info_parts.append(f"{n_cat} catalog stars")
        if n_det:
            info_parts.append(f"{n_det} detections")
        if info_parts:
            print(f"  {DIM}{' │ '.join(info_parts)}{RESET}")
        print()

    def _on_horizon(self, cx=0, cy=0, radius=0, n_points=0, f_implied=0,
                    **kw):
        self._dots(
            self._step_label(1),
            f"R={radius:.0f}px  center=({cx:.0f},{cy:.0f})  "
            f"f≈{f_implied:.0f}")

    def _on_rotation(self, rho_deg=0, **kw):
        self._dots(
            self._step_label(2),
            f"ρ ≈ {rho_deg:.0f}°")

    def _on_pattern_match_start(self, n_detections=0, **kw):
        self.step_t0 = time.time()
        self._overwrite(
            f"{self._step_label(3)}  "
            f"{DIM}{n_detections} point sources ...{RESET}")

    def _on_pattern_match_candidate(self, orientation="", n_matches=0,
                                     rms=0, f=0, **kw):
        e = self._elapsed()
        self._overwrite(
            f"{self._step_label(3)}  "
            f"{DIM}{orientation} f={f:.0f}{RESET}  "
            f"best: {CYAN}{n_matches}{RESET} @ {rms:.1f}px  "
            f"{DIM}[{e:.0f}s]{RESET}")

    def _on_pattern_match_done(self, n_matches=0, rms=0, f=0,
                                mirrored=False, **kw):
        e = self._elapsed()
        mirror_tag = f"  {YELLOW}(mirrored){RESET}" if mirrored else ""
        if n_matches >= 10:
            self._overwrite("")
            self._dots(
                self._step_label(3),
                f"{GREEN}{n_matches}{RESET} matches  "
                f"RMS={rms:.1f}  f={f:.0f}{mirror_tag}",
                elapsed=e)
        else:
            self._overwrite("")
            self._dots(
                self._step_label(3),
                f"{DIM}{n_matches} matches (weak){RESET}{mirror_tag}",
                elapsed=e)
        self._update_best(n_matches, rms, f)

    def _on_fk1(self, f=0, k1=0, **kw):
        self._dots(
            self._step_label(4),
            f"f={f:.0f}  k1={k1:.2e}")

    def _on_fk1_skip(self, **kw):
        self._dots(
            self._step_label(4),
            f"{DIM}skipped{RESET}")

    def _on_projection_search(self, proj_type="", n_matches=0, rms=0,
                               **kw):
        if n_matches > 0:
            self._dots(
                self._step_label(5),
                f"{proj_type}: {n_matches} matches")
        else:
            self._dots(
                self._step_label(5),
                f"{DIM}equidistant (default){RESET}")

    def _on_refine_start(self, **kw):
        self.step_t0 = time.time()

    def _on_refine_phase(self, phase="", label="", n_matches=0, rms=0,
                          f=0, **kw):
        improved = self._update_best(n_matches, rms, f)
        star = f"  {GREEN}★{RESET}" if improved else ""
        e = self._elapsed()
        phase_names = {
            "A": "mid-altitude",
            "B": "wide field",
            "C": "full field",
            "D": "detection-based",
            "E": "deep catalog",
        }
        pname = phase_names.get(phase, phase)
        if n_matches > 0:
            print(f"        {DIM}Phase {phase}{RESET}  {pname:16s} "
                  f"{n_matches:4d} matches  "
                  f"RMS={rms:.1f}  f={f:.0f}{star}")
        else:
            print(f"        {DIM}Phase {phase}  {pname:16s} "
                  f"  —{RESET}")

    def _on_refine_done(self, n_matches=0, rms=0, f=0, **kw):
        e = self._elapsed()
        self._dots(
            self._step_label(6),
            f"{GREEN}{self.best_n}{RESET} matches  "
            f"RMS={self.best_rms:.1f}  f={self.best_f:.0f}",
            elapsed=e)

    def _on_sweep_start(self, n_models=0, **kw):
        self.step_t0 = time.time()
        self._overwrite(
            f"{self._step_label(7)}  "
            f"{DIM}scanning {n_models} models ...{RESET}")

    def _on_sweep_progress(self, fraction=0, **kw):
        e = self._elapsed()
        bar = self._progress_bar(fraction)
        self._overwrite(
            f"{self._step_label(7)}  "
            f"{bar}  {fraction:.0%}  "
            f"{DIM}[{e:.0f}s]{RESET}")

    def _on_sweep_done(self, rho_deg=0, f=0, alt0_deg=0, **kw):
        pass  # Will be completed by sweep_refine results

    def _on_sweep_refine(self, phase=0, n_matches=0, rms=0, f=0, **kw):
        phase_names = {
            1: "high-alt geom",
            2: "mid-alt +dist",
            3: "wide altitude",
            4: "deep guided",
        }
        pname = phase_names.get(phase, f"phase {phase}")
        if n_matches > 0:
            print(f"        {DIM}Sweep {phase}{RESET}  {pname:16s} "
                  f"{n_matches:4d} matches  "
                  f"RMS={rms:.1f}  f={f:.0f}")
        else:
            print(f"        {DIM}Sweep {phase}  {pname:16s} "
                  f"  —{RESET}")

    def _on_sweep_result(self, n_matches=0, rms=0, f=0, **kw):
        e = self._elapsed()
        self._overwrite("")
        self._dots(
            self._step_label(7),
            f"{n_matches} matches  RMS={rms:.1f}  f={f:.0f}",
            elapsed=e)

    def _on_validation(self, main_frac=0, main_n=0, main_total=0,
                       sweep_frac=0, sweep_n=0, sweep_total=0,
                       winner="", **kw):
        print()
        print(f"  {DIM}[{8}/{TOTAL_STEPS}]{RESET} Validation")

        def _bar(frac):
            if frac >= 0.7:
                color = GREEN
            elif frac >= 0.4:
                color = YELLOW
            else:
                color = RED
            n = int(frac * 15)
            return f"{color}{'█' * n}{'░' * (15 - n)}{RESET}"

        print(f"        Main pipeline  "
              f"{_bar(main_frac)} {main_frac:4.0%}  "
              f"({main_n}/{main_total} bright stars)")
        if sweep_total > 0:
            print(f"        Sweep backup   "
                  f"{_bar(sweep_frac)} {sweep_frac:4.0%}  "
                  f"({sweep_n}/{sweep_total} bright stars)")

        if winner:
            if winner == "sweep":
                print(f"        {YELLOW}▸ Sweep backup wins "
                      f"(better validation){RESET}")
            else:
                print(f"        {GREEN}▸ Main pipeline wins{RESET}")

    def _on_diagnostics(self, pattern="", **kw):
        self._dots(
            self._step_label(9),
            f"{pattern}")

    def _on_done(self, n_matches=0, rms=0, f=0, k1=0, cx=0, cy=0,
                 az0_deg=0, alt0_deg=0, rho_deg=0, proj="",
                 output_path="", zeropoint=0, **kw):
        total = self._total_elapsed()
        print()
        w = min(62, self.width - 4)
        border = "─" * w
        print(f"  {GREEN}┌{border}┐{RESET}")
        print(f"  {GREEN}│{RESET} {BOLD}{'INSTRUMENT MODEL SOLVED':^{w-2}}{RESET}"
              f" {GREEN}│{RESET}")
        print(f"  {GREEN}│{RESET}{' ' * (w)}{GREEN}│{RESET}")
        lines = [
            f"Stars matched:  {n_matches}   RMS: {rms:.2f} px",
            f"Focal length:   {f:.1f} px       k1: {k1:.2e}",
            f"Projection:     {proj}",
            f"Boresight:      az={az0_deg:.1f}°  alt={alt0_deg:.1f}°  "
            f"roll={rho_deg:.1f}°",
            f"Center:         ({cx:.1f}, {cy:.1f})",
        ]
        if zeropoint:
            lines.append(f"Zeropoint:      {zeropoint:.3f}")
        if output_path:
            lines.append(f"Saved:          {output_path}")
        for line in lines:
            padding = w - len(line) - 2
            print(f"  {GREEN}│{RESET}  {line}{' ' * max(0, padding)}"
                  f" {GREEN}│{RESET}")
        print(f"  {GREEN}│{RESET}{' ' * (w)}{GREEN}│{RESET}")
        print(f"  {GREEN}│{RESET} {DIM}{'Total time: %.1fs' % total:^{w-2}}"
              f"{RESET} {GREEN}│{RESET}")
        print(f"  {GREEN}└{border}┘{RESET}")
        print()

    def _on_failed(self, reason="", **kw):
        total = self._total_elapsed()
        print()
        w = min(62, self.width - 4)
        border = "─" * w
        print(f"  {RED}┌{border}┐{RESET}")
        print(f"  {RED}│{RESET} {BOLD}{RED}"
              f"{'INSTRUMENT FIT FAILED':^{w-2}}{RESET}"
              f" {RED}│{RESET}")
        print(f"  {RED}│{RESET}{' ' * (w)}{RED}│{RESET}")
        if reason:
            for line in reason.split("\n"):
                padding = w - len(line) - 2
                print(f"  {RED}│{RESET}  {line}"
                      f"{' ' * max(0, padding)} {RED}│{RESET}")
        print(f"  {RED}│{RESET}{' ' * (w)}{RED}│{RESET}")
        print(f"  {RED}│{RESET} {DIM}{'Total time: %.1fs' % total:^{w-2}}"
              f"{RESET} {RED}│{RESET}")
        print(f"  {RED}└{border}┘{RESET}")
        print()

    def _on_moon_excluded(self, alt_deg=0, n_excluded=0, x=0, y=0, **kw):
        print(f"        {YELLOW}☽{RESET} Moon (alt={alt_deg:.0f}°) — "
              f"excluded {n_excluded} detections near "
              f"({x:.0f}, {y:.0f})")
