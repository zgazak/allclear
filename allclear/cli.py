"""Command-line interface for allclear."""

import argparse
import logging
import pathlib
import sys

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="allclear",
        description="All-sky camera cloud detection via star matching",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- instrument-fit ---
    p_fit = subparsers.add_parser(
        "instrument-fit",
        help="Characterize camera geometry from scratch (slow, thorough)",
    )
    p_fit.add_argument("--frames", type=str, required=True,
                       help="Image file(s) — path or glob (FITS/JPG/PNG/TIFF)")
    p_fit.add_argument("--lat", type=float, default=None,
                       help="Observer latitude (deg). If omitted, read from FITS header.")
    p_fit.add_argument("--lon", type=float, default=None,
                       help="Observer longitude (deg). If omitted, read from FITS header.")
    p_fit.add_argument("--lat-key", type=str, default="SITELAT",
                       help="FITS header key for latitude (default: SITELAT)")
    p_fit.add_argument("--lon-key", type=str, default="SITELONG",
                       help="FITS header key for longitude (default: SITELONG)")
    p_fit.add_argument("--time", type=str, default=None,
                       help="Observation time (UTC), e.g. '2024-01-15 03:30:00'. "
                            "Required for images without FITS headers or EXIF timestamps.")
    p_fit.add_argument("--output", type=str, default="instrument_model.json",
                       help="Output model file (default: instrument_model.json)")
    p_fit.add_argument("--focal", type=float, default=None,
                       help="Initial focal length guess (pixels)")
    p_fit.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    p_fit.add_argument("--dashboard", action="store_true",
                       help="Show live progress dashboard")
    p_fit.add_argument("--diagnostic-plot", type=str, default=None,
                       help="Save diagnostic plot to this path")
    p_fit.add_argument("--write-fits", action="store_true",
                       help="Write calibration frame as FITS with WCS header")

    # --- solve ---
    p_solve = subparsers.add_parser(
        "solve",
        help="Process frames with a known instrument model (fast)",
    )
    p_solve.add_argument("--frames", type=str, required=True,
                         help="Image file(s) — path or glob (FITS/JPG/PNG/TIFF)")
    p_solve.add_argument("--model", type=str, required=True,
                         help="Instrument model JSON file")
    p_solve.add_argument("--time", type=str, default=None,
                         help="Observation time (UTC), e.g. '2024-01-15 03:30:00'. "
                              "Required for images without FITS headers or EXIF timestamps.")
    p_solve.add_argument("--output-dir", type=str, default=None,
                         help="Output directory for maps/images")
    p_solve.add_argument("--threshold", type=float, default=0.7,
                         help="Clear-sky transmission threshold (default 0.7)")
    p_solve.add_argument("--no-plot", action="store_true",
                         help="Suppress image output")
    p_solve.add_argument("--format", type=str, default="png",
                         help="Output format(s): png, fits (comma-separated)")
    p_solve.add_argument("--refit-rotation", action="store_true",
                         help="Allow wide rotation search if initial solve "
                              "fails. Use for cameras on rotating mounts or "
                              "platforms that have been physically moved.")
    p_solve.add_argument("--verbose", "-v", action="store_true",
                         help="Show detailed solver logging")

    # --- check ---
    p_check = subparsers.add_parser(
        "check",
        help="Quick check: is a sky region clear?",
    )
    p_check.add_argument("--frame", type=str, required=True,
                         help="Image file (FITS/JPG/PNG/TIFF)")
    p_check.add_argument("--model", type=str, required=True,
                         help="Instrument model JSON file")
    p_check.add_argument("--time", type=str, default=None,
                         help="Observation time (UTC), e.g. '2024-01-15 03:30:00'. "
                              "Required for images without FITS headers or EXIF timestamps.")
    p_check.add_argument("--target-ra", type=float, default=None,
                         help="Target RA (deg, J2000)")
    p_check.add_argument("--target-dec", type=float, default=None,
                         help="Target Dec (deg, J2000)")
    p_check.add_argument("--target-name", type=str, default=None,
                         help="Target name for display")
    p_check.add_argument("--threshold", type=float, default=0.7,
                         help="Clear/cloudy threshold (default 0.7)")

    # --- calibrate ---
    p_cal = subparsers.add_parser(
        "calibrate",
        help="Tier 2 calibration tools (build improvements from many frames)",
    )
    cal_sub = p_cal.add_subparsers(dest="calibrate_subcommand", required=True)

    p_cal_obs = cal_sub.add_parser(
        "obscuration",
        help="Build persistent obscuration mask from many solved frames",
    )
    p_cal_obs.add_argument("--model", type=str, required=True,
                            help="Instrument model JSON file")
    p_cal_obs.add_argument("--frames", type=str, required=True,
                            help="Frame glob (FITS/JPG/PNG/TIFF)")
    p_cal_obs.add_argument("--output", type=str, default=None,
                            help="Output path (default: <model>_obscuration.json)")
    p_cal_obs.add_argument("--clear-gate", type=float, default=0.70,
                            help="Minimum clear fraction to include a frame")
    p_cal_obs.add_argument("--min-visits", type=int, default=8,
                            help="Bins with fewer clear-sky visits are NaN")
    p_cal_obs.add_argument("--vmag-min", type=float, default=1.5,
                            help="Exclude saturated bright stars (default 1.5)")
    p_cal_obs.add_argument("--vmag-max", type=float, default=6.0,
                            help="Exclude faint stars (default 6.0)")
    p_cal_obs.add_argument("--az-step", type=float, default=2.0,
                            help="Azimuth bin size in degrees (default 2.0)")
    p_cal_obs.add_argument("--alt-step", type=float, default=2.0,
                            help="Altitude bin size in degrees (default 2.0)")
    p_cal_obs.add_argument("--verbose", "-v", action="store_true",
                            help="Per-frame progress output")

    # --- manual-fit ---
    p_manual = subparsers.add_parser(
        "manual-fit",
        help="Interactive GUI: click known objects to bootstrap camera model",
    )
    p_manual.add_argument("--frames", type=str, required=True,
                          help="Image file — path (FITS/JPG/PNG/TIFF)")
    p_manual.add_argument("--lat", type=float, default=None,
                          help="Observer latitude (deg). If omitted, read from FITS header.")
    p_manual.add_argument("--lon", type=float, default=None,
                          help="Observer longitude (deg). If omitted, read from FITS header.")
    p_manual.add_argument("--lat-key", type=str, default="SITELAT",
                          help="FITS header key for latitude (default: SITELAT)")
    p_manual.add_argument("--lon-key", type=str, default="SITELONG",
                          help="FITS header key for longitude (default: SITELONG)")
    p_manual.add_argument("--time", type=str, default=None,
                          help="Observation time (UTC), e.g. '2024-01-15 03:30:00'. "
                               "Required for images without FITS headers or EXIF timestamps.")
    p_manual.add_argument("--output", type=str, default="manual_model.json",
                          help="Output model file (default: manual_model.json)")
    p_manual.add_argument("--model", type=str, default=None,
                          help="Optional initial model JSON to start from")

    args = parser.parse_args(argv)

    # Parse --time if provided
    if hasattr(args, "time") and args.time is not None:
        from astropy.time import Time
        from datetime import datetime, timezone
        try:
            # Parse through datetime.fromisoformat first — handles timezone
            # offsets like "+10:00" or "-05:00" and converts to UTC.
            dt = datetime.fromisoformat(args.time)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc)
            args._obs_time = Time(dt, scale="utc")
        except Exception as exc:
            print(f"Cannot parse --time '{args.time}': {exc}", file=sys.stderr)
            return 1
    else:
        args._obs_time = None

    if args.command == "instrument-fit":
        return cmd_instrument_fit(args)
    elif args.command == "solve":
        return cmd_solve(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "manual-fit":
        return cmd_manual_fit(args)
    elif args.command == "calibrate":
        if args.calibrate_subcommand == "obscuration":
            return cmd_calibrate_obscuration(args)


def _resolve_frames(pattern):
    """Expand a glob pattern or single path to a list of supported image files."""
    from .utils import SUPPORTED_EXTS
    p = pathlib.Path(pattern)
    if p.is_file():
        return [p]
    parent = p.parent
    if not parent.exists():
        parent = pathlib.Path(".")
    files = sorted(parent.glob(p.name))
    # Also try the pattern directly if no matches
    if not files:
        files = sorted(pathlib.Path(".").glob(pattern))
    return [f for f in files if f.suffix.lower() in SUPPORTED_EXTS]


def _load_frame(path, lat, lon, initial_f=None, obs_time=None,
                lat_key="SITELAT", lon_key="SITELONG"):
    """Load an image frame and prepare catalog + detections.

    Parameters
    ----------
    path : str or Path
        Image file (FITS, JPG, PNG, TIFF).
    lat, lon : float or None
        Observer coordinates (degrees).  CLI values override FITS header.
    initial_f : float, optional
        Initial focal length guess (pixels).
    obs_time : astropy.time.Time, optional
        Observation time override.  Required for non-FITS images that
        lack EXIF timestamps.
    lat_key, lon_key : str
        FITS header keys for latitude/longitude.
    """
    from .utils import load_image, extract_obs_time, parse_fits_header
    from .catalog import BrightStarCatalog
    from .detection import detect_stars

    data, header = load_image(path)

    if header is not None:
        # FITS — full metadata available
        meta = parse_fits_header(header, lat_key=lat_key, lon_key=lon_key)
        if obs_time is not None:
            meta["obs_time"] = obs_time
    else:
        # Non-FITS — build minimal metadata
        resolved_time = obs_time or extract_obs_time(path)
        if resolved_time is None:
            raise ValueError(
                f"No observation time for {path}. "
                "Provide --time or use an image with EXIF timestamps."
            )
        meta = {
            "obs_time": resolved_time,
            "lat_deg": lat,
            "lon_deg": lon,
            "exposure": 1.0,
            "focal_mm": 1.8,
            "pixel_um": 2.4,
        }

    # CLI --lat/--lon override FITS header values
    if lat is not None:
        meta["lat_deg"] = lat
    if lon is not None:
        meta["lon_deg"] = lon

    # Ensure we have coordinates from one source or the other
    if meta.get("lat_deg") is None or meta.get("lon_deg") is None:
        hint = "Provide --lat and --lon, or use --lat-key and --lon-key to specify FITS header keys."
        if header is not None:
            keys = [k for k in header.keys() if k.strip()]
            hint += f"\nAvailable FITS header keys: {keys}"
        raise ValueError(
            f"No observer coordinates for {path}. {hint}"
        )

    if initial_f is None:
        if meta.get("focal_mm") is not None:
            initial_f = meta["focal_mm"] / (meta["pixel_um"] / 1000.0)
        else:
            # No focal length in header — use image size as rough guide.
            # Assume fisheye covering roughly a hemisphere.
            ny, nx = data.shape
            initial_f = min(nx, ny) * 0.5

    catalog = BrightStarCatalog()
    cat = catalog.get_visible_stars(meta["lat_deg"], meta["lon_deg"], meta["obs_time"])

    # Scale detection count with image size — large sensors have more stars.
    ny, nx = data.shape
    n_det = max(1000, min(3000, (nx * ny) // 4000))
    det = detect_stars(data, fwhm=5.0, threshold_sigma=5.0, n_brightest=n_det)

    return data, meta, cat, det, initial_f


# ---- instrument-fit ----

def cmd_instrument_fit(args):
    from .strategies import instrument_fit_pipeline
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission
    from datetime import datetime, timezone

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching image files: {args.frames}", file=sys.stderr)
        return 1

    MIN_MATCHES = 30

    # ---- Fit each frame independently ----
    results = []  # list of dicts with per-frame results
    for fi, fits_path in enumerate(frames):
        label = (f"[{fi+1}/{len(frames)}] " if len(frames) > 1
                 else "")
        print(f"{label}Instrument fit using {fits_path.name}")

        try:
            data, meta, cat, det, initial_f = _load_frame(
                str(fits_path), args.lat, args.lon, args.focal,
                obs_time=args._obs_time,
                lat_key=args.lat_key, lon_key=args.lon_key,
            )
        except ValueError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        if args.focal is not None:
            initial_f = args.focal

        ny, nx = data.shape
        if not args.dashboard:
            print(f"  Image: {nx}x{ny}")
            print(f"  Obs time: {meta['obs_time'].iso}")
            print(f"  Catalog stars: {len(cat)}")
            print(f"  Detected sources: {len(det)}")
            print(f"  Initial f estimate: {initial_f:.0f} px")

        # Set up dashboard progress display
        progress_cb = None
        if args.dashboard:
            from .progress import ProgressDisplay
            progress_cb = ProgressDisplay()
            progress_cb("start", nx=nx, ny=ny, obs_time=meta["obs_time"],
                        n_cat=len(cat), n_det=len(det))

        model, n_matched, rms, diag = instrument_fit_pipeline(
            data, det, cat, initial_f=initial_f, verbose=args.verbose,
            meta=meta, progress=progress_cb,
        )

        # Quality gate — scale RMS limit with image size since coarser
        # plate scales (larger images, shorter focal lengths) naturally
        # have higher pixel-domain residuals at the same angular accuracy.
        max_rms = max(6.0, min(nx, ny) * 0.004)
        fail_reasons = []
        if n_matched < MIN_MATCHES:
            fail_reasons.append(
                f"too few matches ({n_matched} < {MIN_MATCHES})")
        if rms > max_rms:
            fail_reasons.append(
                f"RMS too high ({rms:.1f} > {max_rms:.1f} px)")

        if fail_reasons:
            print(f"  FAILED: {'; '.join(fail_reasons)}")
            if args.diagnostic_plot:
                diag_path = _per_frame_diag_path(
                    args.diagnostic_plot, fits_path, len(frames))
                _save_diagnostic_plot(data, model, det, cat, n_matched, rms,
                                      diag, diag_path, meta=meta)
                print(f"  Diagnostic plot: {diag_path}")
            continue

        # Compute zeropoint via guided matching
        fit_result = fast_solve(
            data, det, cat, model, refine=False, guided=True)
        fit_det = fit_result.guided_det_table
        fit_pairs = fit_result.matched_pairs
        _, _, _, cal_zp = compute_transmission(
            fit_det, cat, fit_pairs, model, image=data,
        )

        print(f"  OK: {n_matched} matches, RMS={rms:.2f}px, "
              f"zeropoint={cal_zp:.3f}")

        results.append({
            "path": fits_path, "data": data, "meta": meta,
            "cat": cat, "det": det, "model": model,
            "n_matched": n_matched, "rms": rms, "diag": diag,
            "zeropoint": cal_zp, "fit_det": fit_det,
            "fit_pairs": fit_pairs, "nx": nx, "ny": ny,
        })

    # ---- No successful fits ----
    if not results:
        print("\nInstrument fit FAILED: no frames passed quality gate.",
              file=sys.stderr)
        print("  All frames may be cloudy or have too few visible stars.",
              file=sys.stderr)
        return 1

    # ---- Select best geometry (most matches, then lowest RMS) ----
    best = max(results, key=lambda r: (r["n_matched"], -r["rms"]))
    model = best["model"]
    n_matched = best["n_matched"]
    rms = best["rms"]

    # ---- Select best zeropoint (largest = clearest sky) ----
    best_zp_result = max(results, key=lambda r: r["zeropoint"])
    best_zp = best_zp_result["zeropoint"]

    lat = best["meta"]["lat_deg"]
    lon = best["meta"]["lon_deg"]
    nx, ny = best["nx"], best["ny"]

    # Build and save instrument model
    frames_used = [r["path"].name for r in results]
    inst = InstrumentModel.from_camera_model(
        model,
        site_lat=lat,
        site_lon=lon,
        image_width=nx,
        image_height=ny,
        mirrored=best["diag"].get("mirrored", False),
        n_stars_matched=n_matched,
        n_stars_expected=len(best["cat"]),
        rms_residual_px=rms,
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
        frame_used=best["path"].name,
    )
    inst.photometric_zeropoint = best_zp
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    inst.save(args.output)

    # ---- Report ----
    n_total = len(frames)
    n_ok = len(results)
    print(f"\nResults ({n_ok}/{n_total} frames passed):")
    print(f"  Best geometry: {best['path'].name} "
          f"({n_matched} matches, RMS={rms:.2f}px)")
    print(f"  Best zeropoint: {best_zp:.3f} "
          f"(from {best_zp_result['path'].name})")
    print(f"  Projection:   {model.proj_type.value}")
    print(f"  Center:       ({model.cx:.1f}, {model.cy:.1f})")
    print(f"  Focal length: {model.f:.1f} pixels")
    print(f"  Boresight:    az={np.degrees(model.az0):.2f}\u00b0 "
          f"alt={np.degrees(model.alt0):.2f}\u00b0")
    print(f"  Roll:         {np.degrees(model.rho):.2f}\u00b0")
    print(f"  Distortion:   k1={model.k1:.2e}")
    print(f"  Model saved:  {args.output}")

    if best["diag"].get("projection_results"):
        print("  Per-projection RMS:")
        for name, info in sorted(
                best["diag"]["projection_results"].items()):
            print(f"    {name}: {info['rms']:.2f} ({info['n']} matches)")

    # Diagnostic plot (best frame)
    if args.diagnostic_plot:
        plot_data = best["data"]
        # If the pipeline detected a mirrored image, flip for display
        if best["diag"].get("mirrored", False):
            plot_data = plot_data[:, ::-1]
        _save_diagnostic_plot(
            plot_data, model, best["det"], best["cat"],
            n_matched, rms, best["diag"], args.diagnostic_plot,
            meta=best["meta"])
        print(f"  Diagnostic plot: {args.diagnostic_plot}")

    # Write FITS with WCS header
    if args.write_fits:
        from .utils import write_fits_with_wcs
        fits_out = str(pathlib.Path(args.output).with_suffix('')) + "_solved.fits"
        # Reload original header if input was FITS
        orig_hdr = None
        if best["path"].suffix.lower() in {".fits", ".fit", ".fts"}:
            from astropy.io import fits as pyfits
            orig_hdr = pyfits.getheader(str(best["path"]))
        write_data = best["data"]
        if best["diag"].get("mirrored", False):
            write_data = write_data[:, ::-1]
        write_fits_with_wcs(
            write_data, model,
            obs_time=best["meta"]["obs_time"],
            site_lat=lat, site_lon=lon,
            output_path=fits_out,
            original_header=orig_hdr,
            mirrored=best["diag"].get("mirrored", False),
            extra_keys={
                "AC_NSTAR": (n_matched, "AllClear matched stars"),
                "AC_RMS": (round(rms, 3), "[px] AllClear astrometric RMS"),
                "AC_ZP": (round(best_zp, 4),
                          "[mag] AllClear photometric zeropoint"),
            },
        )
        print(f"  Solved FITS: {fits_out}")

    # Save annotated image using guided matches from best frame
    plot_path = str(pathlib.Path(args.output).with_suffix('')) + "_solved.png"
    plot_data_solved = best["data"]
    if best["diag"].get("mirrored", False):
        plot_data_solved = plot_data_solved[:, ::-1]
    _save_annotated_image(
        plot_data_solved, model, best["fit_det"], best["cat"], plot_path,
        matched_pairs=best["fit_pairs"],
        obs_time=best["meta"]["obs_time"],
        lat=lat, lon=lon)
    print(f"  Annotated image: {plot_path}")

    return 0


def _per_frame_diag_path(base_path, frame_path, n_frames):
    """Generate a per-frame diagnostic plot path.

    Single frame: use base_path as-is.
    Multiple frames: insert frame stem before extension.
    """
    if n_frames <= 1:
        return base_path
    base = pathlib.Path(base_path)
    return str(base.parent / f"{base.stem}_{frame_path.stem}{base.suffix}")


# ---- solve ----

def cmd_solve(args):
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission, interpolate_transmission

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_file = pathlib.Path(args.model)
    if not model_file.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        print("Run 'allclear instrument-fit' first to create a model.",
              file=sys.stderr)
        return 1

    inst = InstrumentModel.load(model_file)
    camera = inst.to_camera_model()

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching image files: {args.frames}", file=sys.stderr)
        return 1

    output_dir = None
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(frames)} frame(s) with model {args.model}")

    for i, fpath in enumerate(frames):
        try:
            data, meta, cat, det, _ = _load_frame(
                str(fpath), inst.site_lat, inst.site_lon,
                obs_time=args._obs_time,
            )
        except ValueError as e:
            print(f"  [{i+1}/{len(frames)}] SKIP {fpath.name}: {e}")
            continue

        # If the instrument model was fitted to a mirrored image,
        # flip this frame to match.
        if inst.mirrored:
            data = data[:, ::-1]
            det["x"] = (data.shape[1] - 1) - np.asarray(det["x"],
                                                          dtype=np.float64)

        result = fast_solve(data, det, cat, camera, guided=True,
                            refit_rotation=args.refit_rotation,
                            obscuration=inst.obscuration)

        status_str = f"  [{i+1}/{len(frames)}] {fpath.name}: "
        status_str += (f"{result.n_matched}/{result.n_expected} matches, "
                       f"RMS={result.rms_residual:.2f}")

        # Use guided_det_table (centroided star positions) for transmission
        use_det = result.guided_det_table if result.guided_det_table is not None and len(result.guided_det_table) > 0 else det

        if result.n_matched >= 3:
            ref_zp = inst.photometric_zeropoint or None
            az, alt, trans, zp = compute_transmission(
                use_det, cat, result.matched_pairs, result.camera_model,
                image=data, reference_zeropoint=ref_zp,
                obscuration=inst.obscuration,
            )
            tmap = interpolate_transmission(az, alt, trans)
            clear_frac = float(np.nanmean(
                tmap.get_observability_mask(threshold=args.threshold)
            ))
            status_str += f", clear={clear_frac:.0%}"

            # Note when the zeropoint was auto-upgraded from frame
            if ref_zp and ref_zp != 0.0 and zp > ref_zp + 0.15:
                status_str += (
                    f"\n    NOTE: using frame zeropoint ({zp:.3f}) — "
                    f"better than model ({ref_zp:.3f})"
                )

            if output_dir:
                out_base = output_dir / fpath.stem
            else:
                out_base = pathlib.Path(fpath.stem)

            # Per-frame model JSON (lightweight — solved geometry + zeropoint)
            model_path = str(out_base) + "_model.json"
            frame_inst = InstrumentModel.from_camera_model(
                result.camera_model,
                site_lat=inst.site_lat, site_lon=inst.site_lon,
                mirrored=inst.mirrored,
                photometric_zeropoint=zp,
                n_stars_matched=result.n_matched,
                n_stars_expected=result.n_expected,
                rms_residual=result.rms_residual,
            )
            frame_inst.save(model_path)

            # FITS output with WCS header
            formats = {f.strip().lower()
                       for f in args.format.split(",")}
            if "fits" in formats:
                from .utils import write_fits_with_wcs
                fits_path_out = str(out_base) + "_solved.fits"
                # Reload original header if input was FITS
                orig_hdr = None
                if fpath.suffix.lower() in {".fits", ".fit", ".fts"}:
                    from astropy.io import fits as pyfits
                    orig_hdr = pyfits.getheader(str(fpath))
                write_fits_with_wcs(
                    data, result.camera_model,
                    obs_time=meta["obs_time"],
                    site_lat=inst.site_lat,
                    site_lon=inst.site_lon,
                    output_path=fits_path_out,
                    original_header=orig_hdr,
                    mirrored=inst.mirrored,
                    extra_keys={
                        "AC_NSTAR": (result.n_matched,
                                     "AllClear matched stars"),
                        "AC_RMS": (round(result.rms_residual, 3),
                                   "[px] AllClear astrometric RMS"),
                        "AC_CLEAR": (round(clear_frac, 3),
                                     "AllClear clear-sky fraction"),
                        "AC_ZP": (round(zp, 4),
                                  "[mag] AllClear photometric zeropoint"),
                    },
                )
                status_str += f"\n    -> {fits_path_out}"

            if not args.no_plot and "png" in formats:
                # Annotated image with matches
                solved_path = str(out_base) + "_solved.png"
                _save_annotated_image(
                    data, result.camera_model, use_det, cat,
                    solved_path,
                    matched_pairs=result.matched_pairs,
                    obs_time=meta["obs_time"],
                    lat=inst.site_lat, lon=inst.site_lon,
                )

                # Transmission overlay
                trans_path = str(out_base) + "_transmission.png"
                _save_annotated_image(
                    data, result.camera_model, use_det, cat,
                    trans_path,
                    matched_pairs=result.matched_pairs,
                    transmission_data=(az, alt, trans),
                    obs_time=meta["obs_time"],
                    lat=inst.site_lat, lon=inst.site_lon,
                    obscuration=inst.obscuration,
                )

                # Blink GIF: slow alternation between solved and transmission
                blink_path = str(out_base) + "_blink.gif"
                _save_blink_gif(solved_path, trans_path, blink_path)
                status_str += (f"\n    -> {solved_path}"
                               f"\n    -> {trans_path}"
                               f"\n    -> {blink_path}")
        else:
            status_str += " (too few matches)"

        if result.status != "ok":
            status_str += f" [{result.status}]"

        print(status_str)

        if result.status_detail:
            print(f"    {result.status_detail}")

    print("Done.")
    return 0


# ---- check ----

def cmd_check(args):
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission, interpolate_transmission

    model_file = pathlib.Path(args.model)
    if not model_file.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    inst = InstrumentModel.load(model_file)
    camera = inst.to_camera_model()

    try:
        data, meta, cat, det, _ = _load_frame(
            args.frame, inst.site_lat, inst.site_lon,
            obs_time=args._obs_time,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = fast_solve(data, det, cat, camera, guided=True,
                        obscuration=inst.obscuration)

    if result.n_matched < 3:
        print(f"Insufficient matches ({result.n_matched}) — "
              "cannot determine sky conditions.")
        return 1

    use_det = result.guided_det_table if result.guided_det_table is not None and len(result.guided_det_table) > 0 else det
    ref_zp = inst.photometric_zeropoint or None
    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
        obscuration=inst.obscuration,
    )
    tmap = interpolate_transmission(az, alt, trans)
    clear_frac = float(np.nanmean(
        tmap.get_observability_mask(threshold=args.threshold)
    ))

    print(f"Frame: {args.frame}")
    print(f"  Matched: {result.n_matched}/{result.n_expected} stars")
    print(f"  Overall clear fraction: {clear_frac:.0%}")

    # Check specific target if given
    if args.target_ra is not None and args.target_dec is not None:
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        import astropy.units as u
        import warnings

        location = EarthLocation(
            lat=inst.site_lat * u.deg,
            lon=inst.site_lon * u.deg,
        )
        frame = AltAz(obstime=meta["obs_time"], location=location)
        target = SkyCoord(ra=args.target_ra * u.deg,
                          dec=args.target_dec * u.deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            target_altaz = target.transform_to(frame)

        t_alt = target_altaz.alt.deg
        t_az = target_altaz.az.deg
        name = args.target_name or f"RA={args.target_ra:.1f} Dec={args.target_dec:.1f}"

        if t_alt < 5:
            print(f"  {name}: below horizon (alt={t_alt:.1f}\u00b0)")
        else:
            # Find nearest grid point in transmission map
            az_idx = np.argmin(np.abs(tmap.az_grid - t_az))
            alt_idx = np.argmin(np.abs(tmap.alt_grid - t_alt))
            t_val = tmap.transmission[alt_idx, az_idx]

            if np.isnan(t_val):
                status = "NO DATA"
            elif t_val >= args.threshold:
                status = "CLEAR"
            else:
                status = "CLOUDY"

            print(f"  {name} (alt={t_alt:.0f}\u00b0 az={t_az:.0f}\u00b0): "
                  f"transmission {t_val:.2f} ({status})")

    return 0


# ---- manual-fit ----

def cmd_manual_fit(args):
    from .manual_fit import get_identifiable_objects
    from .manual_fit_web import ManualFitWeb

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching image files: {args.frames}", file=sys.stderr)
        return 1

    fits_path = frames[0]
    print(f"Manual fit using {fits_path.name}")

    try:
        data, meta, cat, det, initial_f = _load_frame(
            str(fits_path), args.lat, args.lon,
            obs_time=args._obs_time,
            lat_key=args.lat_key, lon_key=args.lon_key,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ny, nx = data.shape
    print(f"  Image: {nx}x{ny}")
    print(f"  Obs time: {meta['obs_time'].iso}")
    print(f"  Catalog stars: {len(cat)}")

    objects = get_identifiable_objects(
        meta["lat_deg"], meta["lon_deg"], meta["obs_time"])
    print(f"  Identifiable objects: {len(objects)}")

    # Load initial model if provided
    initial_model = None
    mirrored = False
    if args.model:
        from .instrument import InstrumentModel
        model_path = pathlib.Path(args.model)
        if model_path.exists():
            inst = InstrumentModel.load(model_path)
            initial_model = inst.to_camera_model()
            mirrored = inst.mirrored
            print(f"  Initial model loaded from {args.model}")
            if mirrored:
                print(f"  Model is mirrored (image will be flipped)")
        else:
            print(f"  Warning: model file not found: {args.model}",
                  file=sys.stderr)

    viewer = ManualFitWeb(data, objects, cat, meta, args.output,
                          initial_model=initial_model, mirrored=mirrored)
    viewer.run()
    return 0


# ---- Plotting helpers ----

def _save_blink_gif(path_a, path_b, output_path, duration_ms=1500,
                     max_width=1200):
    """Create a slow-blink animated GIF alternating between two PNGs."""
    from PIL import Image
    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB").resize(img_a.size)
    # Downscale for reasonable GIF size
    w, h = img_a.size
    if w > max_width:
        scale = max_width / w
        new_size = (max_width, int(h * scale))
        img_a = img_a.resize(new_size, Image.LANCZOS)
        img_b = img_b.resize(new_size, Image.LANCZOS)
    # Quantize to shared 256-color palette for smaller file
    img_a_q = img_a.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    img_b_q = img_b.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    img_a_q.save(
        output_path, save_all=True, append_images=[img_b_q],
        duration=duration_ms, loop=0, optimize=True,
    )


def _save_annotated_image(data, model, det, cat, output_path,
                          matched_pairs=None, transmission_data=None,
                          obs_time=None, lat=None, lon=None,
                          obscuration=None):
    """Save an annotated frame image."""
    from .plotting import plot_frame
    from .matching import match_sources

    if matched_pairs is None:
        cat_az = np.radians(np.asarray(cat["az_deg"], dtype=np.float64))
        cat_alt = np.radians(np.asarray(cat["alt_deg"], dtype=np.float64))
        cat_x, cat_y = model.sky_to_pixel(cat_az, cat_alt)
        cat_xy = np.column_stack([cat_x, cat_y])
        det_xy = np.column_stack([
            np.asarray(det["x"], dtype=np.float64),
            np.asarray(det["y"], dtype=np.float64),
        ])
        matched_pairs, _ = match_sources(det_xy, cat_xy, max_dist=15.0)

    plot_frame(
        data, model,
        det_table=det,
        cat_table=cat,
        matched_pairs=matched_pairs,
        show_grid=True,
        transmission_data=transmission_data,
        obs_time=obs_time,
        lat_deg=lat,
        lon_deg=lon,
        output_path=output_path,
        obscuration=obscuration,
    )


def _save_diagnostic_plot(data, model, det, cat, n_matched, rms, diag, path,
                          meta=None):
    """Save an annotated frame as a diagnostic plot."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    from .plotting import plot_frame
    from .matching import match_sources

    # Compute matched pairs from model
    cat_xy = np.column_stack(model.sky_to_pixel(
        np.radians(np.array(cat["az_deg"], dtype=float)),
        np.radians(np.array(cat["alt_deg"], dtype=float)),
    ))
    det_xy = np.column_stack([np.array(det["x"], dtype=float),
                              np.array(det["y"], dtype=float)])
    pairs, _ = match_sources(det_xy, cat_xy, max_dist=15.0)

    obs_time = meta.get("obs_time") if meta else None
    lat_deg = meta.get("lat_deg") if meta else None
    lon_deg = meta.get("lon_deg") if meta else None
    horizon_r = diag.get("horizon_R") if diag else None
    horizon_cx = diag.get("horizon_cx") if diag else None
    horizon_cy = diag.get("horizon_cy") if diag else None
    plot_frame(data, model, det_table=det, cat_table=cat,
               matched_pairs=pairs, obs_time=obs_time,
               lat_deg=lat_deg, lon_deg=lon_deg, output_path=path,
               horizon_r=horizon_r, horizon_center=(horizon_cx, horizon_cy)
               if horizon_cx is not None else None)


# ---- calibrate obscuration ----

def cmd_calibrate_obscuration(args):
    """Build a sky-space ObscurationMask from many solved frames.

    For each frame:
      1. Solve against the supplied instrument model.
      2. Record each bright catalog star's (az, alt, detected) outcome.
      3. Compute the frame's clear fraction from its transmission map.
    After all frames are processed, aggregate outcomes in (az, alt)
    bins via ``obscuration.build_from_observations`` and write the
    resulting mask to the model's sidecar.
    """
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .catalog import BrightStarCatalog
    from .detection import detect_stars
    from .transmission import compute_transmission, interpolate_transmission
    from .utils import load_image, parse_fits_header
    from .obscuration import build_from_observations

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    model_file = pathlib.Path(args.model)
    if not model_file.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1
    inst = InstrumentModel.load(model_file)
    camera = inst.to_camera_model()

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching frames: {args.frames}", file=sys.stderr)
        return 1

    output = pathlib.Path(args.output) if args.output else \
        InstrumentModel.obscuration_sidecar_path(model_file)

    print(f"Calibrating obscuration from {len(frames)} frames against "
          f"{model_file.name}")

    cat_lib = BrightStarCatalog()

    az_all = []
    alt_all = []
    det_all = []
    vmag_all = []
    cf_all = []

    n_ok = 0
    n_low = 0
    for i, fpath in enumerate(frames):
        try:
            data, header = load_image(str(fpath))
            meta = (parse_fits_header(header)
                    if header is not None else {"obs_time": args._obs_time})
            if meta.get("obs_time") is None:
                n_low += 1
                continue
            cat = cat_lib.get_visible_stars(
                inst.site_lat, inst.site_lon, meta["obs_time"])
            det = detect_stars(data, fwhm=inst.detection_fwhm,
                               threshold_sigma=inst.detection_threshold_sigma,
                               n_brightest=1000)
            if inst.mirrored:
                data = data[:, ::-1]
                det["x"] = (data.shape[1] - 1) - np.asarray(
                    det["x"], dtype=np.float64)

            result = fast_solve(data, det, cat, camera, guided=True)
            if result.n_matched < 30:
                n_low += 1
                continue

            # Frame clear fraction
            use_det = (result.guided_det_table
                       if result.guided_det_table is not None
                       and len(result.guided_det_table) > 0 else det)
            ref_zp = inst.photometric_zeropoint or None
            az_t, alt_t, trans_t, _ = compute_transmission(
                use_det, cat, result.matched_pairs, result.camera_model,
                image=data, reference_zeropoint=ref_zp,
            )
            tmap = interpolate_transmission(az_t, alt_t, trans_t)
            clear_frac = float(np.nanmean(
                tmap.get_observability_mask(threshold=0.7)))

            # Record in-frame bright catalog stars
            cam_solved = result.camera_model
            vmag = np.asarray(cat["vmag"], dtype=np.float64)
            alt_deg = np.asarray(cat["alt_deg"], dtype=np.float64)
            az_deg = np.asarray(cat["az_deg"], dtype=np.float64)

            bright = vmag < args.vmag_max
            idx = np.where(bright & (alt_deg > 5.0))[0]
            if idx.size == 0:
                continue
            px, py = cam_solved.sky_to_pixel(
                np.radians(az_deg[idx]), np.radians(alt_deg[idx]))
            ny, nx = data.shape
            in_frame = (np.isfinite(px) & np.isfinite(py)
                        & (px >= 0) & (px < nx)
                        & (py >= 0) & (py < ny))
            matched_cat = {ci for _, ci in result.matched_pairs}

            ok = idx[in_frame]
            for ci in ok:
                az_all.append(float(az_deg[ci]))
                alt_all.append(float(alt_deg[ci]))
                vmag_all.append(float(vmag[ci]))
                det_all.append(1 if int(ci) in matched_cat else 0)
                cf_all.append(clear_frac)
            n_ok += 1

            if args.verbose or (i + 1) % 10 == 0 or (i + 1) == len(frames):
                print(f"  [{i+1}/{len(frames)}] {fpath.name}: "
                      f"n_match={result.n_matched}, clear={clear_frac:.2f}",
                      flush=True)
        except Exception as exc:
            if args.verbose:
                print(f"  [{i+1}/{len(frames)}] {fpath.name}: "
                      f"SKIP {type(exc).__name__}: {exc}", file=sys.stderr)
            n_low += 1
            continue

    if not az_all:
        print("No usable observations — all frames failed to solve.",
              file=sys.stderr)
        return 1

    print(f"\n{n_ok}/{len(frames)} frames solved, "
          f"{len(az_all):,} per-star observations collected "
          f"({n_low} frames skipped)")

    mask = build_from_observations(
        az_deg=np.array(az_all), alt_deg=np.array(alt_all),
        detected=np.array(det_all, dtype=np.int32),
        clear_fraction=np.array(cf_all),
        vmag=np.array(vmag_all),
        clear_gate=args.clear_gate,
        vmag_min=args.vmag_min, vmag_max=args.vmag_max,
        min_visits=args.min_visits,
        az_step_deg=args.az_step, alt_step_deg=args.alt_step,
        n_frames=n_ok,
    )

    n_filled = int(np.isfinite(mask.weight).sum())
    n_total = mask.weight.size
    n_obscured = int(((mask.weight < 0.3) & np.isfinite(mask.weight)).sum())
    print(f"Mask grid: {len(mask.alt_edges_deg) - 1} alt × "
          f"{len(mask.az_edges_deg) - 1} az bins")
    print(f"  Filled (≥ {args.min_visits} clear-sky visits): {n_filled:,} "
          f"({100.0 * n_filled / n_total:.1f}%)")
    print(f"  Obscured (weight < 0.3): {n_obscured:,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output)
    print(f"\nSaved: {output}")
    if output == InstrumentModel.obscuration_sidecar_path(model_file):
        print(f"  (auto-loaded by `allclear solve --model {model_file.name}`)")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
