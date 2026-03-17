"""Command-line interface for allclear."""

import argparse
import logging
import os
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
                       help="Observer latitude (deg). Falls back to SITELAT env var.")
    p_fit.add_argument("--lon", type=float, default=None,
                       help="Observer longitude (deg). Falls back to SITELONG env var.")
    p_fit.add_argument("--time", type=str, default=None,
                       help="Observation time (UTC), e.g. '2024-01-15 03:30:00'. "
                            "Required for images without FITS headers or EXIF timestamps.")
    p_fit.add_argument("--output", type=str, default="instrument_model.json",
                       help="Output model file (default: instrument_model.json)")
    p_fit.add_argument("--focal", type=float, default=None,
                       help="Initial focal length guess (pixels)")
    p_fit.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    p_fit.add_argument("--diagnostic-plot", type=str, default=None,
                       help="Save diagnostic plot to this path")

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

    # --- animate ---
    p_anim = subparsers.add_parser(
        "animate",
        help="Assemble solved frames into a timelapse animation",
    )
    p_anim.add_argument("--frames", type=str, default=None,
                        help="Image file(s) — path or glob (FITS/JPG/PNG/TIFF). "
                             "Requires --model. Omit if using --input-dir.")
    p_anim.add_argument("--model", type=str, default=None,
                        help="Instrument model JSON file (required with --frames)")
    p_anim.add_argument("--input-dir", type=str, default=None,
                        help="Directory of existing solve output PNGs "
                             "(e.g. from 'allclear solve --output-dir')")
    p_anim.add_argument("--time", type=str, default=None,
                        help="Observation time (UTC) override for non-FITS images")
    p_anim.add_argument("--output", type=str, default="timelapse.webp",
                        help="Output file — .webp (default, small), .gif, or .mp4 (requires ffmpeg)")
    p_anim.add_argument("--fps", type=float, default=4,
                        help="Frames per second (default: 4)")
    p_anim.add_argument("--max-width", type=int, default=1200,
                        help="Max output width in pixels (default: 1200)")
    p_anim.add_argument("--mode", type=str, default="transmission",
                        choices=["solved", "transmission", "extinction"],
                        help="Annotation style (default: transmission)")
    p_anim.add_argument("--threshold", type=float, default=0.7,
                        help="Clear-sky transmission threshold (default 0.7)")

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
    elif args.command == "animate":
        return cmd_animate(args)


def _resolve_site(args_lat, args_lon):
    """Resolve site lat/lon from CLI args or SITELAT/SITELONG env vars.

    Returns (lat, lon) floats, either of which may be None if unresolvable.
    Prints a warning if neither source provides a value.
    """
    lat = args_lat
    lon = args_lon
    if lat is None and "SITELAT" in os.environ:
        lat = float(os.environ["SITELAT"])
    if lon is None and "SITELONG" in os.environ:
        lon = float(os.environ["SITELONG"])
    if lat is None or lon is None:
        missing = []
        if lat is None:
            missing.append("latitude (--lat or SITELAT)")
        if lon is None:
            missing.append("longitude (--lon or SITELONG)")
        print(
            f"WARNING: Observer site {' and '.join(missing)} not set. "
            "Star catalog will be incomplete.",
            file=sys.stderr,
        )
    return lat, lon


def _resolve_frames(pattern):
    """Expand a glob pattern or single path to a list of supported image files."""
    from .utils import SUPPORTED_EXTS
    p = pathlib.Path(pattern).expanduser()
    if p.is_file():
        return [p]
    parent = p.parent
    files = sorted(parent.glob(p.name)) if parent.exists() else []
    # Also try the pattern directly if no matches
    if not files:
        files = sorted(pathlib.Path(".").glob(pattern))
    return [f for f in files if f.suffix.lower() in SUPPORTED_EXTS]


def _load_frame(path, lat, lon, initial_f=None, obs_time=None):
    """Load an image frame and prepare catalog + detections.

    Parameters
    ----------
    path : str or Path
        Image file (FITS, JPG, PNG, TIFF).
    lat, lon : float
        Observer coordinates (degrees).
    initial_f : float, optional
        Initial focal length guess (pixels).
    obs_time : astropy.time.Time, optional
        Observation time override.  Required for non-FITS images that
        lack EXIF timestamps.
    """
    from .utils import load_image, extract_obs_time, parse_fits_header
    from .catalog import BrightStarCatalog
    from .detection import detect_stars

    data, header = load_image(path)

    if header is not None:
        # FITS — full metadata available
        meta = parse_fits_header(header)
        if obs_time is not None:
            meta["obs_time"] = obs_time
        # CLI lat/lon override or fill missing header values
        if lat is not None:
            meta["lat_deg"] = lat
        elif meta["lat_deg"] is None:
            raise ValueError(
                f"No site latitude in FITS header for {path}. Provide --lat."
            )
        if lon is not None:
            meta["lon_deg"] = lon
        elif meta["lon_deg"] is None:
            raise ValueError(
                f"No site longitude in FITS header for {path}. Provide --lon."
            )
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

    if initial_f is None:
        initial_f = meta["focal_mm"] / (meta["pixel_um"] / 1000.0)

    catalog = BrightStarCatalog()
    cat = catalog.get_visible_stars(lat, lon, meta["obs_time"])

    det = detect_stars(data, fwhm=5.0, threshold_sigma=5.0, n_brightest=1000)

    return data, meta, cat, det, initial_f


# ---- instrument-fit ----

def cmd_instrument_fit(args):
    from .strategies import instrument_fit_pipeline
    from .instrument import InstrumentModel
    from datetime import datetime, timezone

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    lat, lon = _resolve_site(args.lat, args.lon)

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching image files: {args.frames}", file=sys.stderr)
        return 1

    # Use first frame for characterization (could average multiple later)
    fits_path = frames[0]
    print(f"Instrument fit using {fits_path.name}")

    try:
        data, meta, cat, det, initial_f = _load_frame(
            str(fits_path), lat, lon, args.focal,
            obs_time=args._obs_time,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.focal is not None:
        initial_f = args.focal

    ny, nx = data.shape
    print(f"  Image: {nx}x{ny}")
    print(f"  Obs time: {meta['obs_time'].iso}")
    print(f"  Catalog stars: {len(cat)}")
    print(f"  Detected sources: {len(det)}")
    print(f"  Initial f estimate: {initial_f:.0f} px")

    model, n_matched, rms, diag = instrument_fit_pipeline(
        data, det, cat, initial_f=initial_f, verbose=args.verbose,
    )

    # Quality gate — reject clearly bad fits
    MIN_MATCHES = 30
    MAX_RMS = 5.0
    fail_reasons = []
    if n_matched < MIN_MATCHES:
        fail_reasons.append(
            f"too few matches ({n_matched} < {MIN_MATCHES})")
    if rms > MAX_RMS:
        fail_reasons.append(
            f"RMS too high ({rms:.1f} > {MAX_RMS:.1f} px)")
    if fail_reasons:
        print(f"\nInstrument fit FAILED: {'; '.join(fail_reasons)}",
              file=sys.stderr)
        print("  This frame may be cloudy or have too few visible stars.",
              file=sys.stderr)
        print("  Try a clearer frame for calibration.", file=sys.stderr)
        # Still save diagnostic plot if requested — useful for debugging
        if args.diagnostic_plot:
            _save_diagnostic_plot(data, model, det, cat, n_matched, rms,
                                  diag, args.diagnostic_plot)
            print(f"  Diagnostic plot: {args.diagnostic_plot}",
                  file=sys.stderr)
        return 1

    # Build and save instrument model
    inst = InstrumentModel.from_camera_model(
        model,
        site_lat=lat,
        site_lon=lon,
        image_width=nx,
        image_height=ny,
        n_stars_matched=n_matched,
        n_stars_expected=len(cat),
        rms_residual_px=rms,
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
        frame_used=fits_path.name,
    )
    inst.save(args.output)

    print(f"\nResults:")
    print(f"  Matched:      {n_matched} stars")
    print(f"  RMS residual: {rms:.2f} pixels")
    print(f"  Projection:   {model.proj_type.value}")
    print(f"  Center:       ({model.cx:.1f}, {model.cy:.1f})")
    print(f"  Focal length: {model.f:.1f} pixels")
    print(f"  Boresight:    az={np.degrees(model.az0):.2f}\u00b0 "
          f"alt={np.degrees(model.alt0):.2f}\u00b0")
    print(f"  Roll:         {np.degrees(model.rho):.2f}\u00b0")
    print(f"  Distortion:   k1={model.k1:.2e}")
    print(f"  Model saved:  {args.output}")

    if diag.get("projection_results"):
        print("  Per-projection RMS:")
        for name, info in sorted(diag["projection_results"].items()):
            print(f"    {name}: {info['rms']:.2f} ({info['n']} matches)")

    # Diagnostic plot
    if args.diagnostic_plot:
        _save_diagnostic_plot(data, model, det, cat, n_matched, rms, diag,
                              args.diagnostic_plot)
        print(f"  Diagnostic plot: {args.diagnostic_plot}")

    # Generate guided matches for the annotated image (more complete
    # than DAOStarFinder matching, which misses most stars)
    from .solver import fast_solve
    from .transmission import compute_transmission
    fit_result = fast_solve(data, det, cat, model, refine=False, guided=True)
    fit_det = fit_result.guided_det_table
    fit_pairs = fit_result.matched_pairs

    # Compute and save photometric zeropoint from the calibration frame
    _, _, _, cal_zp = compute_transmission(
        fit_det, cat, fit_pairs, model, image=data,
    )
    inst.photometric_zeropoint = cal_zp
    inst.save(args.output)
    print(f"  Zeropoint:    {cal_zp:.3f} (clear-sky reference)")

    # Save annotated image using guided matches
    plot_path = pathlib.Path(args.output).stem + "_solved.png"
    _save_annotated_image(data, model, fit_det, cat, plot_path,
                          matched_pairs=fit_pairs,
                          obs_time=meta["obs_time"],
                          lat=lat, lon=lon)
    print(f"  Annotated image: {plot_path}")

    return 0


# ---- solve ----

def cmd_solve(args):
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission, interpolate_transmission

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

        result = fast_solve(data, det, cat, camera, guided=True)

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
            )
            tmap = interpolate_transmission(az, alt, trans)
            clear_frac = float(np.nanmean(
                tmap.get_observability_mask(threshold=args.threshold)
            ))
            status_str += f", clear={clear_frac:.0%}"

            # Warn if this frame's zeropoint is better (brighter)
            # than the calibration reference — suggests clearer sky
            if ref_zp and ref_zp != 0.0:
                _, _, _, frame_zp = compute_transmission(
                    use_det, cat, result.matched_pairs,
                    result.camera_model, image=data,
                )
                # Larger zeropoint = brighter stars = clearer sky
                if frame_zp > ref_zp + 0.15:
                    status_str += (
                        f"\n    NOTE: frame zeropoint ({frame_zp:.3f}) is "
                        f"better than model ({ref_zp:.3f}). "
                        f"Consider re-running instrument-fit with this frame."
                    )

            if not args.no_plot:
                if output_dir:
                    out_base = output_dir / fpath.stem
                else:
                    out_base = pathlib.Path(fpath.stem)

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
                )

                # Extinction (delta_mag) overlay
                ext_path = str(out_base) + "_extinction.png"
                _save_annotated_image(
                    data, result.camera_model, use_det, cat,
                    ext_path,
                    matched_pairs=result.matched_pairs,
                    transmission_data=(az, alt, trans),
                    overlay_mode="extinction",
                    obs_time=meta["obs_time"],
                    lat=inst.site_lat, lon=inst.site_lon,
                )

                # Blink GIF: cycle through solved, transmission, extinction
                blink_path = str(out_base) + "_blink.gif"
                _save_blink_gif(
                    [solved_path, trans_path, ext_path], blink_path,
                )
                status_str += (f"\n    -> {solved_path}"
                               f"\n    -> {trans_path}"
                               f"\n    -> {ext_path}"
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

    result = fast_solve(data, det, cat, camera, guided=True)

    if result.n_matched < 3:
        print(f"Insufficient matches ({result.n_matched}) — "
              "cannot determine sky conditions.")
        return 1

    use_det = result.guided_det_table if result.guided_det_table is not None and len(result.guided_det_table) > 0 else det
    ref_zp = inst.photometric_zeropoint or None
    az, alt, trans, zp = compute_transmission(
        use_det, cat, result.matched_pairs, result.camera_model,
        image=data, reference_zeropoint=ref_zp,
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
                ext_str = ""
            elif t_val >= args.threshold:
                status = "CLEAR"
                ext_mag = -2.5 * np.log10(max(t_val, 1e-4))
                ext_str = f", extinction {ext_mag:.1f} mag"
            else:
                status = "CLOUDY"
                ext_mag = -2.5 * np.log10(max(t_val, 1e-4))
                ext_str = f", extinction {ext_mag:.1f} mag"

            print(f"  {name} (alt={t_alt:.0f}\u00b0 az={t_az:.0f}\u00b0): "
                  f"transmission {t_val:.2f}{ext_str} ({status})")

    return 0


# ---- animate ----

def _animate_from_dir(args):
    """Load pre-rendered PNGs from a solve output directory."""
    from PIL import Image

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Directory not found: {args.input_dir}", file=sys.stderr)
        return None

    suffix = f"_{args.mode}.png"
    png_files = sorted(input_dir.glob(f"*{suffix}"))
    if not png_files:
        print(f"No *{suffix} files in {input_dir}", file=sys.stderr)
        print(f"  Available modes: "
              + ", ".join(sorted({p.stem.rsplit("_", 1)[-1]
                                  for p in input_dir.glob("*.png")})),
              file=sys.stderr)
        return None

    print(f"Loading {len(png_files)} {args.mode} frames from {input_dir}")
    pil_frames = []
    for p in png_files:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if w > args.max_width:
            scale = args.max_width / w
            img = img.resize((args.max_width, int(h * scale)), Image.LANCZOS)
        pil_frames.append(img)

    return pil_frames


def _animate_from_frames(args):
    """Solve frames from scratch and render to PIL images."""
    from .instrument import InstrumentModel
    from .solver import fast_solve
    from .transmission import compute_transmission
    from .plotting import plot_frame
    from .utils import load_image, extract_obs_time, parse_fits_header
    from PIL import Image
    import io
    import matplotlib.pyplot as plt

    model_file = pathlib.Path(args.model)
    if not model_file.exists():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return None

    inst = InstrumentModel.load(model_file)
    camera = inst.to_camera_model()

    frames = _resolve_frames(args.frames)
    if not frames:
        print(f"No matching image files: {args.frames}", file=sys.stderr)
        return None

    # Extract observation times for chronological sorting
    frame_times = []
    for fpath in frames:
        try:
            _, header = load_image(str(fpath))
            if header is not None:
                meta = parse_fits_header(header)
                obs_time = args._obs_time or meta["obs_time"]
            else:
                obs_time = args._obs_time or extract_obs_time(fpath)
            frame_times.append((fpath, obs_time))
        except Exception as e:
            print(f"  SKIP {fpath.name}: cannot read time ({e})")

    if not frame_times:
        print("No frames with valid observation times.", file=sys.stderr)
        return None

    # Sort chronologically (frames without times go to the end)
    frame_times.sort(key=lambda ft: ft[1].jd if ft[1] is not None else float("inf"))

    print(f"Animating {len(frame_times)} frame(s) with model {args.model}")
    if frame_times[0][1] and frame_times[-1][1]:
        print(f"  Time range: {frame_times[0][1].iso} -> {frame_times[-1][1].iso}")

    pil_frames = []
    for i, (fpath, obs_time) in enumerate(frame_times):
        try:
            data, meta, cat, det, _ = _load_frame(
                str(fpath), inst.site_lat, inst.site_lon,
                obs_time=obs_time,
            )
        except ValueError as e:
            print(f"  [{i+1}/{len(frame_times)}] SKIP {fpath.name}: {e}")
            continue

        result = fast_solve(data, det, cat, camera, guided=True)
        use_det = (result.guided_det_table
                   if result.guided_det_table is not None
                   and len(result.guided_det_table) > 0
                   else det)

        # Build transmission/extinction overlay if requested and enough matches
        transmission_data = None
        if args.mode in ("transmission", "extinction") and result.n_matched >= 3:
            ref_zp = inst.photometric_zeropoint or None
            az, alt, trans, _ = compute_transmission(
                use_det, cat, result.matched_pairs, result.camera_model,
                image=data, reference_zeropoint=ref_zp,
            )
            transmission_data = (az, alt, trans)

        # Render to in-memory image via matplotlib
        fig, ax = plot_frame(
            data, result.camera_model,
            det_table=use_det, cat_table=cat,
            matched_pairs=result.matched_pairs,
            show_grid=True,
            transmission_data=transmission_data,
            overlay_mode=args.mode if args.mode != "solved" else "transmission",
            obs_time=meta["obs_time"],
            lat_deg=inst.site_lat, lon_deg=inst.site_lon,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # Downscale for reasonable file size
        w, h = img.size
        if w > args.max_width:
            scale = args.max_width / w
            img = img.resize((args.max_width, int(h * scale)), Image.LANCZOS)

        pil_frames.append(img)

        n_match_str = f"{result.n_matched}/{result.n_expected}"
        print(f"  [{i+1}/{len(frame_times)}] {fpath.name}: {n_match_str} matches")

    return pil_frames or None


def cmd_animate(args):
    from PIL import Image

    if args.input_dir:
        pil_frames = _animate_from_dir(args)
    elif args.frames and args.model:
        pil_frames = _animate_from_frames(args)
    else:
        print("Provide either --input-dir or both --frames and --model.",
              file=sys.stderr)
        return 1

    if not pil_frames:
        print("No frames rendered successfully.", file=sys.stderr)
        return 1

    # Ensure all frames are the same size (pad/crop to first frame's size)
    target_size = pil_frames[0].size
    for i in range(1, len(pil_frames)):
        if pil_frames[i].size != target_size:
            pil_frames[i] = pil_frames[i].resize(target_size, Image.LANCZOS)

    duration_ms = int(1000 / args.fps)
    output_path = pathlib.Path(args.output)
    ext = output_path.suffix.lower()

    if ext == ".mp4":
        import subprocess
        import shutil
        if not shutil.which("ffmpeg"):
            print("Error: mp4 output requires ffmpeg. "
                  "Use .webp or .gif, or install ffmpeg.", file=sys.stderr)
            return 1
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, frame in enumerate(pil_frames):
                frame.save(pathlib.Path(tmpdir) / f"{idx:06d}.png")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(args.fps),
                "-i", str(pathlib.Path(tmpdir) / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ], check=True, capture_output=True)
    elif ext == ".webp":
        pil_frames[0].save(
            output_path, save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0, quality=80, method=4,
        )
    else:
        # GIF fallback
        quantized = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                     for f in pil_frames]
        quantized[0].save(
            output_path, save_all=True, append_images=quantized[1:],
            duration=duration_ms, loop=0, optimize=True,
        )

    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path} ({len(pil_frames)} frames, {file_mb:.1f} MB)")

    return 0


# ---- Plotting helpers ----

def _save_blink_gif(image_paths, output_path, duration_ms=1500,
                     max_width=1200):
    """Create a slow-blink animated GIF cycling through multiple PNGs."""
    from PIL import Image
    if len(image_paths) < 2:
        return

    first = Image.open(image_paths[0]).convert("RGB")
    size = first.size

    # Downscale for reasonable GIF size
    w, h = size
    if w > max_width:
        scale = max_width / w
        size = (max_width, int(h * scale))
        first = first.resize(size, Image.LANCZOS)

    imgs = [first]
    for p in image_paths[1:]:
        imgs.append(Image.open(p).convert("RGB").resize(size, Image.LANCZOS))

    # Quantize to shared 256-color palette for smaller file
    quantized = [img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                 for img in imgs]
    quantized[0].save(
        output_path, save_all=True, append_images=quantized[1:],
        duration=duration_ms, loop=0, optimize=True,
    )


def _save_annotated_image(data, model, det, cat, output_path,
                          matched_pairs=None, transmission_data=None,
                          overlay_mode="transmission",
                          obs_time=None, lat=None, lon=None):
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
        overlay_mode=overlay_mode,
        obs_time=obs_time,
        lat_deg=lat,
        lon_deg=lon,
        output_path=output_path,
    )


def _save_diagnostic_plot(data, model, det, cat, n_matched, rms, diag, path):
    """Save a multi-panel diagnostic plot."""
    from .plotting import plot_residuals
    plot_residuals(det, cat, [], model, output_path=path)


if __name__ == "__main__":
    sys.exit(main() or 0)
