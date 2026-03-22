# Next Session: Fix star confusion in dense fields (Milky Way)

## The Real Problem
In dense star fields (especially the Milky Way), the guided matching picks the WRONG star. With >1 star per 3 pixels in the MW, the "nearest local maximum" at a predicted position is almost always a different star than the target. This causes:

1. **Wrong transmission measurements**: The matched "star" has wrong flux → wrong transmission
2. **MW-shaped clear region**: Matches cluster in the MW (high density = always finds SOMETHING) while sparser regions show fewer matches
3. **All frames look "cloudy" except near zenith**: Because the edges (lower altitude, longer path through the MW) have the densest star confusion

## What's working
- **instrument-fit**: APICAM gets 2306 matches at RMS=2.27px on its calibration frame (MAY). Model is correct.
- **Mirror detection**: Fully automatic, works perfectly.
- **Model geometry**: Verified correct — LMC, SMC, planets all in right positions on all frames.
- **Solver on calibration frame**: 799 matches, RMS=1.72, 92% clear on MAY. Perfect.

## What's broken
- **Solver on other frames**: 657 matches but many are WRONG stars in dense regions. Transmission map shows MW-shaped clear region instead of uniform clear sky. 39% clear on a perfectly clear night.

## Root Cause
`_guided_match` finds the nearest local maximum above threshold within the search box. In the MW where stars are separated by 2-3 pixels, the nearest maximum is usually a DIFFERENT star. The model prediction is accurate to ~1-2px, but there are 3-5 stars within 3px of any point in the MW.

## Ideas to Fix

### 1. Flux-based match verification
For each guided match, compare the expected flux (from catalog vmag + model zeropoint) with the measured flux. If they differ by >1 mag, reject the match. This would eliminate most wrong-star matches because the wrong star is usually a different magnitude.

```python
expected_flux = 10**(-0.4 * (vmag - zeropoint))
measured_flux = peak - local_background
if abs(-2.5*log10(measured_flux) - (-2.5*log10(expected_flux))) > 1.0:
    reject  # wrong star
```

### 2. PSF-based centroiding
Instead of finding local maxima, fit a PSF model at the predicted position. If there's a star AT the predicted position (even if it's not the brightest nearby), the PSF fit will find it. This naturally handles blended stars.

### 3. Differential matching
Compare the image to a model-predicted image (synthetic sky from catalog). The difference image would show only unmatched features (clouds, artifacts). Stars that match would cancel out.

### 4. Local neighborhood pattern matching (BEST APPROACH)
Instead of matching individual stars, match each star using its **local neighborhood pattern** — the 2-3 nearest bright catalog neighbors form a triangle with a unique shape. Find that triangle in the image near the predicted position. This:
- Works even in the densest MW regions (the triangle shape is unique)
- Self-verifies the match (if you find the triangle, you've confirmed the star)
- Gives a local offset correction for free
- Naturally handles clouds: missing stars break the triangle, marking the area as cloudy
- `matching.py` already has `match_triangles` infrastructure

The approach per star:
1. Get the 2-3 nearest bright catalog neighbors
2. Compute the triangle's edge ratios (scale-invariant signature)
3. In the image, search for the same triangle near the predicted position
4. If found, the offset gives the local correction and the flux gives transmission
5. If not found, the star is behind a cloud (or too faint)

This avoids the "nearest local max = wrong star" problem entirely.

### 5. Isolation-based weighting
Weight matches by local star density. In the MW, reduce the weight of each match. In sparse regions, increase weight. This prevents the MW from dominating the transmission map.

## Test commands
```bash
# Calibration frame (should be ~92% clear)
uv run allclear solve --frames "benchmark/data/eso_apicam/APICAM.2019-05-03T01:08:57.000.fits" --model temp/test_apicam.json --output-dir temp/apicam_solve/ --format .png

# Clear frame that currently shows 39% clear (should be ~95%+)
uv run allclear solve --frames "benchmark/data/eso_apicam/APICAM.2019-02-10T01:52:35.000.fits" --model temp/test_apicam.json --output-dir temp/apicam_solve/ --format .png

# All frames
uv run allclear solve --frames "benchmark/data/eso_apicam/APICAM.2019-*.fits" --model temp/test_apicam.json --output-dir temp/apicam_solve/ --format .png
```

## Key insight from this session
The solver's detection threshold was 15-sigma (too high for dim frames). Lowered to 3-sigma with tight search radius (r=3-4). This finds more stars, but many are wrong matches in dense fields. The bottleneck is now match ACCURACY, not match COUNT.
