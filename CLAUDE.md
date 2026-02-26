# CLAUDE.md — AllClear Project Guide

## What this project does
All-sky camera cloud detection. Takes FITS images from a fisheye camera pointing at the sky, identifies stars by matching against a catalog, and produces cloud transmission maps.

## Build & run

```bash
uv sync                     # install dependencies
uv run allclear --help      # CLI help

# One-time camera calibration
uv run allclear instrument-fit \
    --frames example_images/2023_11_19__00_00_11.fits \
    --lat 20.7458 --lon -156.4317 \
    --output instrument_model.json

# Process frames
uv run allclear solve \
    --frames "example_images/*.fits" \
    --model instrument_model.json
```

## Test

```bash
uv run pytest allclear/tests/ -v
```

32 tests. Two test files (test_solver.py, test_synthetic_roundtrip.py) have import errors from API drift — they reference `AllSkySolver` which no longer exists.

## Key architecture

### Two-phase workflow
1. **`instrument-fit`** (slow, one-time): Blind-solves camera geometry from scratch. Pattern-match hypothesis → guided refinement → distortion fitting. Saves model JSON with photometric zeropoint.
2. **`solve`** (fast, per-frame): Uses known model for guided matching + small pointing refinement. Produces transmission maps.

### Guided matching (core technique)
For each catalog star, project to pixel position, find nearest bright peak in a search box, centroid for sub-pixel accuracy. This avoids DAOStarFinder detections which are dominated by dome/obstruction artifacts.

### Transmission measurement
- **Global zeropoint**: Calibrated during instrument-fit from clear-sky frame. Stored in model JSON. All solve frames measure absolute transmission against this reference.
- **Local-background flux**: Star flux = peak - local_median (box edge pixels). Handles bright obstructions (dome glow shows high peak but low contrast).
- **Unmatched probing**: Bright in-frame catalog stars with no point source detected → zero transmission.

### Camera model
Fisheye projection: equidistant (r = f*theta) with radial distortion (k1, k2). Parameters: center (cx, cy), boresight (az0, alt0), roll (rho), focal length (f). Near-zenith pointing means az0 and rho are partially degenerate.

## Important conventions
- Angles in code: radians (except display/CLI which use degrees)
- Pixel coordinates: origin="lower" (FITS convention, y increases upward)
- Camera model stored in `projection.py:CameraModel` dataclass
- Instrument model stored in `instrument.py:InstrumentModel` dataclass (JSON persistence)
- k1 must be <= 0 (barrel distortion for fisheye)
- Rotation matrix: R = Rz(-rho) * Rx(-tilt) * Rz(-az0)

## File map
- `strategies.py` — Blind solve pipeline (largest file, ~1300 lines)
- `solver.py` — Fast solve with known model
- `projection.py` — CameraModel, sky_to_pixel / pixel_to_sky
- `instrument.py` — InstrumentModel JSON save/load
- `transmission.py` — compute_transmission, interpolate_transmission
- `plotting.py` — Annotated frame rendering, grid overlay, planets
- `catalog.py` — BrightStarCatalog class
- `detection.py` — DAOStarFinder wrapper
- `cli.py` — CLI entry point (instrument-fit, solve, check)

## Known constraints
- Camera: ZWO ASI178MM (3096x2080, 16-bit). Dead columns at x < 200 and x > 2800.
- Stars brighter than vmag ~1.5 are saturated at 16-bit — transmission artificially low for those.
- Observation site: Haleakala (lat=20.7458, lon=-156.4317).
- Jupiter/bright planets are not in the star catalog but appear in images. Moon/planets are labeled in plots.
