# allclear

![Tests](https://raw.githubusercontent.com/zgazak/allclear/main/tests.svg) ![Coverage](https://raw.githubusercontent.com/zgazak/allclear/main/coverage.svg)

All-sky camera cloud detection via stellar transmission mapping.

`allclear` takes images from all-sky cameras, blind-solves the camera
geometry by matching detected stars against a catalog, and produces
full-hemisphere cloud transmission maps with annotated PNG output.
Works across diverse camera hardware — tested on four cameras at four
observatory sites with sensors from 1392×1040 to 4096×4096, fixed and
tracking mounts, and mirror-reversed optical trains.

|             Clear sky              |               Obscured sky               |
| :--------------------------------: | :--------------------------------------: |
| ![Clear sky](https://raw.githubusercontent.com/zgazak/allclear/main/docs/clear_blink.gif) | ![Obscured sky](https://raw.githubusercontent.com/zgazak/allclear/main/docs/obscured_blink.gif) |

*Blink comparison: star identification (red circles = catalog, green crosshairs = matched) and transmission overlay (green = clear, red = cloudy).*

## How it works

1. **Detection** -- Background-subtracted star detection using DAOStarFinder
2. **Catalog** -- Hipparcos/BSC5 bright star catalog (~9000 stars, V <= 6.5) projected to observer alt/az with atmospheric refraction and extinction
3. **Blind solve** -- Hypothesis-and-verify pattern matching: for each bright (detection, catalog star) pair, hypothesize camera parameters, count matches. Tests both normal and mirror-reversed orientations. Best hypothesis seeds iterative refinement.
4. **Camera model** -- Least-squares refinement of a fisheye camera model (center, focal length, boresight, roll, radial distortion). Horizon-radius constraint breaks the f/k1 degeneracy and seeds automatic projection type selection among four fisheye models.
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
Typically matches 300-2300 stars at 0.5-2.3 pixel RMS depending on camera.

Supports FITS, JPG, PNG, and TIFF. For non-FITS images, provide the
observation time with `--time`:

```
allclear instrument-fit \
    --frames sky_photo.jpg \
    --lat 20.7458 --lon -156.4317 \
    --time "2024-01-15 03:30:00-10:00" \
    --output instrument_model.json
```

If a fully blind solve struggles on an unusual camera, bootstrap interactively
with `allclear manual-fit --frames your_sky_image.fits` — click a few known
bright stars or planets to seed the geometry, then let the solver refine.

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

### 4. (Optional) Mask fixed obstructions

Most installations have domes, telescopes, or trees permanently blocking part of
the sky. Build an obscuration mask from a set of clear-sky frames so those
regions are excluded from matching and transmission instead of being read as
cloud:

```
allclear calibrate obscuration \
    --model instrument_model.json \
    --frames "clear_nights/*.fits"
```

This writes a `<model>_obscuration.json` sidecar that `solve` and `check` load
automatically when it sits next to the model file.

### 5. (Optional) Build a timelapse animation

Assemble many frames into a timelapse — handy for reviewing a whole night at a
glance. Either re-render from raw frames with a known model:

```
allclear animate \
    --frames "sky_images/*.fits" \
    --model instrument_model.json \
    --mode transmission \
    --output night.webp
```

or stitch the PNGs a previous `solve --output-dir` run already produced
(fast, no re-solving):

```
allclear animate \
    --input-dir solve_output/ \
    --mode transmission \
    --output night.webp
```

`--mode` selects the annotation style: `solved`, `transmission`, or
`extinction` (extinction in magnitudes). The output format follows the file
extension — `.webp` (default, compact), `.gif`, or `.mp4` (requires `ffmpeg`).
Control playback with `--fps` and frame size with `--max-width`.

## Camera model

The fisheye projection model supports four lens types, selected automatically
during `instrument-fit`:

| Type          | Formula               |
| ------------- | --------------------- |
| Equidistant   | r = f * theta         |
| Equisolid     | r = 2f * sin(theta/2) |
| Stereographic | r = 2f * tan(theta/2) |
| Orthographic  | r = f * sin(theta)    |

Parameters: optical center (cx, cy), boresight direction (az0, alt0),
roll angle (rho), focal length (f), and radial distortion coefficients (k1, k2).

Mirror-reversed optical trains (e.g. cameras with diagonal mirrors or prisms)
are detected and handled automatically — the solver tests both orientations
and picks the one that produces more star matches.

The instrument model is saved as a JSON file with camera geometry,
site coordinates, detection settings, photometric zeropoint, mirror flag,
and fit quality metadata.

### Conventions

- **Angles** are radians internally; the CLI and printed output use degrees.
- **Pixel coordinates** follow the FITS convention: origin at lower-left, with
  y increasing upward.
- **Distortion**: `k1 <= 0` (barrel distortion) for physical fisheye lenses.
- **Rotation** decomposes as `R = Rz(-rho) * Rx(-tilt) * Rz(-az0)`. Near the
  zenith, `az0` and `rho` are partially degenerate (they produce nearly
  identical image rotation), so the solver fixes `az0 = 0` early and splits the
  rotation out later.

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

## Multi-camera benchmark

Validated on four cameras spanning diverse hardware and sites:

| Camera | Site | Sensor | Resolution | Matches | RMS (px) |
| ------ | ---- | ------ | ---------- | ------- | -------- |
| ESO APICAM | Paranal, Chile | KAF-16803 CCD | 4096×4096 | 2331 | 1.7 |
| Haleakala | Haleakala, HI | ZWO ASI178MM | 3096×2080 | 862 | 2.0 |
| Cloudynight | Flagstaff, AZ | SX Oculus | 1392×1040 | 344 | 0.5 |
| Liverpool SkyCam | La Palma, Spain | Sony ICX267AL | 1392×1040 | 649 | 1.2 |

All results from fully blind single-frame solves — no initial geometry or
camera metadata provided. APICAM required automatic mirror detection (E-W
flipped optical train). See [`benchmark/README.md`](benchmark/README.md) for
dataset download scripts, ground-truth labels, and reproducible experiments
(yearly blind-fit robustness + nightly tracking).

## Limitations

- **Saturated bright stars** (V ≲ 1.5 on a 16-bit sensor) clip at the pixel
  ceiling, so their measured transmission reads artificially low — a known
  artifact, not real cloud.
- **Planets and the Moon** are not in the stellar catalog. Bright planets
  (e.g. Jupiter) appear as unmatched sources, and while the Moon is labeled in
  plots, strong moonlight raises the local sky background near it.
- **Sensor edge and dead columns** can produce spurious detections; the
  obscuration mask and magnitude gating mitigate this.
- **Rotating mounts without field derotation** (roll that changes from frame to
  frame) are not yet supported. Fixed mounts and derotated tracking mounts
  (stable roll) are handled.

## Project structure

```
allclear/
    __init__.py          # Package init
    cli.py               # Command-line interface (instrument-fit, solve, check, calibrate, manual-fit, animate)
    api.py               # Python API (SkyTransmissionResult, satellite link queries)
    strategies.py        # Blind solve pipeline, pattern matching, guided refinement
    solver.py            # Fast solve with known model (guided matching + pointing refinement)
    monitor.py           # Operational wrapper (model auto-update, history tracking)
    instrument.py        # InstrumentModel dataclass + JSON persistence
    projection.py        # Fisheye camera model (4 projection types, sky<->pixel)
    catalog.py           # BrightStarCatalog (Hipparcos/BSC5, refraction, extinction)
    detection.py         # Star detection (Background2D + DAOStarFinder)
    matching.py          # KDTree 1-to-1 matching, triangle-hash blind matching
    transmission.py      # Per-star photometry, RBF interpolation, TransmissionMap
    obscuration.py       # Sky-coordinate obstruction mask (domes, trees, telescopes)
    plotting.py          # Annotated frame rendering, grid overlay, planets
    manual_fit.py        # Interactive click-to-bootstrap camera model
    synthetic.py         # Synthetic frame generation for testing
    utils.py             # FITS I/O, coordinate math, extinction
    data/                # Cached star catalog (ECSV)
    tests/               # Test suite
benchmark/
    data/                # FITS frames from 4 cameras (downloaded via scripts)
    labels/              # Human-annotated ground truth (JSON per frame)
    scripts/             # Reproducible download scripts for all cameras
    solutions/           # AllClear astrometric solutions (output)
```

## Development

```bash
git clone https://github.com/zgazak/allclear.git
cd allclear
uv sync                          # install with dev dependencies
uv run pytest allclear/tests/ -v # run tests (88 tests)
```

## License

See [LICENSE](LICENSE).
