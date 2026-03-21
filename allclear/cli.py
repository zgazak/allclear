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
    p_solve.add_argument("--refit-rotation", action="store_true",
                         help="Allow wide rotation search if initial solve "
                              "fails. Use for cameras on rotating mounts or "
                              "platforms that have been physically moved.")

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

    det = detect_stars(data, fwhm=5.0, threshold_sigma=5.0, n_brightest=1000)

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
        print(f"  Image: {nx}x{ny}")
        print(f"  Obs time: {meta['obs_time'].iso}")
        print(f"  Catalog stars: {len(cat)}")
        print(f"  Detected sources: {len(det)}")
        print(f"  Initial f estimate: {initial_f:.0f} px")

        model, n_matched, rms, diag = instrument_fit_pipeline(
            data, det, cat, initial_f=initial_f, verbose=args.verbose,
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

    # ---- Select best zeropoint (most negative = clearest sky) ----
    best_zp_result = min(results, key=lambda r: r["zeropoint"])
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
        _save_diagnostic_plot(
            best["data"], model, best["det"], best["cat"],
            n_matched, rms, best["diag"], args.diagnostic_plot,
            meta=best["meta"])
        print(f"  Diagnostic plot: {args.diagnostic_plot}")

    # Save annotated image using guided matches from best frame
    plot_path = pathlib.Path(args.output).stem + "_solved.png"
    _save_annotated_image(
        best["data"], model, best["fit_det"], best["cat"], plot_path,
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

        result = fast_solve(data, det, cat, camera, guided=True,
                            refit_rotation=args.refit_rotation)

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
            elif t_val >= args.threshold:
                status = "CLEAR"
            else:
                status = "CLOUDY"

            print(f"  {name} (alt={t_alt:.0f}\u00b0 az={t_az:.0f}\u00b0): "
                  f"transmission {t_val:.2f} ({status})")

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
        obs_time=obs_time,
        lat_deg=lat,
        lon_deg=lon,
        output_path=output_path,
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


if __name__ == "__main__":
    sys.exit(main() or 0)
