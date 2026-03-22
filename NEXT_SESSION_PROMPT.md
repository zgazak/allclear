# Next Session: Fix Liverpool SkyCam, polish pipeline

## Big win this session: Mirror detection
APICAM was mirror-flipped (E-W swapped). Adding automatic mirror testing to the pattern match (test both `det_x` and `2*cx - det_x`) was the breakthrough — APICAM went from 129 to **2305 matches at RMS=2.27px**. The LMC/SMC labels land perfectly on the fuzzy patches.

## Remaining issues

### Solve mode refinement for APICAM
The instrument-fit model (2306 matches, RMS=2.27) is excellent, but `solve` mode on other frames gets only 192 matches with systematic offsets. The `--refit-rotation` flag makes it worse (127 matches). Issues:
- Solver uses vmag < 5.0 bright_mask (only 464 stars) vs instrument-fit's full catalog
- The `_refine_pointing` bounds may be too tight for the sky rotation between epochs
- The solver's `_find_rotation_offset` function may be finding wrong offsets in dense fields

### Liverpool SkyCam
Liverpool is also mirrored (auto-detected), but the solve is still mediocre: 338 matches at RMS=4.26 with stereographic projection. The equidistant Phases B-E all collapse (0 matches) because k1 fitting diverges — the lens is genuinely stereographic and can't be modeled with equidistant+k1. Only Phase F (stereographic) saves it, but it's finding a local minimum.

### What's wrong with Liverpool
- Saturn label is at wrong position in diagnostic plot (should be az=119°, alt=31°)
- Red catalog circles don't align well with visible stars
- The stereographic Phase F gets 338 matches but might have wrong rotation
- The manual-fit tool shows it IS mirrored, but even after mirroring the solve isn't great

### Ideas to fix Liverpool
1. Run the pattern match directly with stereographic projection (currently only tests equidistant in Step 2)
2. The pattern match's `implied_f = dr[i] / bcat_za[j]` assumes equidistant — for stereographic it should be `implied_f = dr[i] / (2 * tan(bcat_za[j]/2))`
3. Try using the manual-fit tool to bootstrap a correct model, then use that as reference

## Test commands
```bash
# APICAM — should get 2000+ matches (SOLVED)
uv run allclear instrument-fit --frames "benchmark/data/eso_apicam/APICAM.2019-05-03T01:08:57.000.fits" --lat-key 'ESO TEL GEOLAT' --lon-key 'ESO TEL GEOLON' --output temp/test_apicam.json --diagnostic-plot temp/test_apicam.png --verbose

# Liverpool — needs work (338 matches, mediocre)
uv run allclear instrument-fit --frames benchmark/data/liverpool_skycam/a_e_20240809_248_1_1_1.fits --lat 28.762 --lon -17.879 --output temp/test_liverpool.json --diagnostic-plot temp/test_liverpool.png --verbose

# Haleakala (regression — should get 766 matches)
uv run allclear instrument-fit --frames example_images/2023_11_19__00_00_11.fits --lat 20.7458 --lon -156.4317 --output temp/test_haleakala.json --diagnostic-plot temp/test_haleakala.png --verbose

# Cloudynight (regression — should get 702 matches)
uv run allclear instrument-fit --frames benchmark/data/cloudynight/005.fits --lat 35.097 --lon -111.535 --output temp/test_cloudynight.json --diagnostic-plot temp/test_cloudynight.png --verbose

# Manual-fit tool (interactive)
uv run allclear manual-fit --frames "benchmark/data/liverpool_skycam/a_e_20240809_248_1_1_1.fits" --lat 28.762 --lon -17.879 --model temp/test_liverpool.json
```

## Key changes this session (all kept)
- **Mirror auto-detection** in Step 2 — tests both normal and flipped x-coordinates
- `_adaptive_min_peak_offset()` — noise-aware threshold
- Nearest-peak matching in `_guided_match` — finds closest local max, not brightest
- Always sigma-clip in `guided_refine`
- LMC/SMC labels in `plotting.py`
- Projection type search (Step 5b + Phase F) — Liverpool is stereographic
- `manual-fit` CLI subcommand — interactive GUI for correcting/bootstrapping models
- Density rotation estimate (Step 1b) — infrastructure exists
- Relaxed k1 bound, scaled n_brightest/anchors, `_is_better` improvements
- Wider center search grid for large images

## Files modified
- `allclear/strategies.py` — mirror detection, density seeding, matching improvements
- `allclear/manual_fit.py` — NEW: interactive GUI tool
- `allclear/plotting.py` — LMC/SMC labels, conditional backend
- `allclear/cli.py` — manual-fit subcommand, mirror-aware diagnostic plots, scaled detections
- `allclear/projection.py` — unchanged
