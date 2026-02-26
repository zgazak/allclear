# allclear

![Tests](https://raw.githubusercontent.com/zgazak/allclear/main/tests.svg) ![Coverage](https://raw.githubusercontent.com/zgazak/allclear/main/coverage.svg)

All-sky camera cloud detection via stellar transmission mapping.

`allclear` takes FITS images from all-sky cameras, blind-solves the camera
geometry by matching detected stars against a catalog, and produces
full-hemisphere cloud transmission maps with annotated PNG output.

|             Clear sky              |               Obscured sky               |
| :--------------------------------: | :--------------------------------------: |
| ![Clear sky](https://raw.githubusercontent.com/zgazak/allclear/main/docs/clear_blink.gif) | ![Obscured sky](https://raw.githubusercontent.com/zgazak/allclear/main/docs/obscured_blink.gif) |

*Blink comparison: star identification (red circles = catalog, green crosshairs = matched) and transmission overlay (green = clear, red = cloudy).*

## How it works

1. **Detection** -- Background-subtracted star detection using DAOStarFinder
2. **Catalog** -- Hipparcos/BSC5 bright star catalog (~9000 stars, V <= 6.5) projected to observer alt/az with atmospheric refraction and extinction
3. **Blind solve** -- Hypothesis-and-verify pattern matching: for each bright (detection, catalog star) pair, hypothesize camera parameters, count matches. Best hypothesis seeds iterative refinement.
4. **Camera model** -- Least-squares refinement of a fisheye camera model (center, focal length, boresight, roll, radial distortion) with automatic projection type selection
5. **Guided matching** -- For each catalog star, project to pixel position and find the nearest bright peak. Sub-pixel centroiding. Avoids DAOStarFinder artifacts from domes/obstructions.
6. **Transmission** -- Per-star photometric transmission from local-background-subtracted flux vs catalog magnitude, calibrated against a global clear-sky zeropoint. Unmatched catalog stars contribute zero transmission. Interpolated via RBF.

## Installation

```
pip install allclear
```

Or with [uv](https://docs.astral.sh/uv/):

```
uv add allclear
uv sync
```

Requires Python >= 3.10. Dependencies: numpy, scipy, astropy, photutils, matplotlib.

## Quick start

### 1. Characterize camera geometry (one-time)

```
allclear instrument-fit \
    --frames your_sky_image.fits \
    --lat 20.7458 --lon -156.4317 \
    --output instrument_model.json
```

This blind-solves the camera model and saves it as JSON, including a
photometric zeropoint calibrated from the (presumably clear) input frame.
Typically matches 400-550 stars at 2-3 pixel RMS.

Supports FITS, JPG, PNG, and TIFF. For non-FITS images, provide the
observation time with `--time`:

```
allclear instrument-fit \
    --frames sky_photo.jpg \
    --lat 20.7458 --lon -156.4317 \
    --time "2024-01-15 03:30:00-10:00" \
    --output instrument_model.json
```

### 2. Process frames with known model (fast)

```
allclear solve \
    --frames "sky_images/*.fits" \
    --model instrument_model.json
```

For each frame, produces:
- `<stem>_solved.png` -- Annotated image with star matches (green crosshairs = matched, red circles = catalog)
- `<stem>_transmission.png` -- Same image with RdYlGn transmission overlay (green = clear, red = cloudy/obstructed)
- `<stem>_blink.gif` -- Animated blink between solved and transmission views

Use `--no-plot` to suppress image output. Use `--output-dir <dir>` to write to a specific directory.

### 3. Quick sky check

```
allclear check \
    --frame sky_image.fits \
    --model instrument_model.json \
    --target-ra 83.63 --target-dec -5.39 --target-name "Orion Nebula"
```

Reports overall clear fraction and transmission at a specific sky position.

## Camera model

The fisheye projection model supports four lens types:

| Type          | Formula               |
| ------------- | --------------------- |
| Equidistant   | r = f * theta         |
| Equisolid     | r = 2f * sin(theta/2) |
| Stereographic | r = 2f * tan(theta/2) |
| Orthographic  | r = f * sin(theta)    |

Parameters: optical center (cx, cy), boresight direction (az0, alt0),
roll angle (rho), focal length (f), and radial distortion coefficients (k1, k2).

The instrument model is saved as a JSON file with camera geometry,
site coordinates, detection settings, photometric zeropoint, and fit quality
metadata.

## Transmission mapping

Transmission is measured on an **absolute scale**: the photometric zeropoint
from `instrument-fit` (clear sky) defines transmission = 1.0. Subsequent
`solve` frames measure dimming relative to that reference.

- Stars behind obstructions (domes, telescopes) are detected via
  local-background contrast -- bright diffuse glow is distinguished from
  point-source starlight
- Unmatched catalog stars (bright, in-frame, but no detectable point source)
  contribute zero transmission to the interpolation
- If a solve frame has a *better* zeropoint than the model, a warning
  suggests re-running `instrument-fit` with the clearer frame

## Project structure

```
allclear/
    __init__.py          # Package init
    cli.py               # Command-line interface (instrument-fit, solve, check)
    strategies.py        # Blind solve pipeline, pattern matching, guided refinement
    solver.py            # Fast solve with known model (guided matching + pointing refinement)
    instrument.py        # InstrumentModel dataclass + JSON persistence
    projection.py        # Fisheye camera model (4 projection types, sky<->pixel)
    catalog.py           # BrightStarCatalog (Hipparcos/BSC5, refraction, extinction)
    detection.py         # Star detection (Background2D + DAOStarFinder)
    matching.py          # KDTree 1-to-1 matching, triangle-hash blind matching
    transmission.py      # Per-star photometry, RBF interpolation, TransmissionMap
    plotting.py          # Annotated frame rendering, grid overlay, planets
    synthetic.py         # Synthetic frame generation for testing
    utils.py             # FITS I/O, coordinate math, extinction
    data/                # Cached star catalog (ECSV)
    tests/               # Test suite (32 tests)
```

## Development

```bash
git clone https://github.com/zgazak/allclear.git
cd allclear
uv sync                          # install with dev dependencies
uv run pytest allclear/tests/ -v # run tests (32 tests)
```

## License

See [LICENSE](LICENSE).
