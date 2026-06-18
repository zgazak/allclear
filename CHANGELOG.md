# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-18

### Added
- **Obscuration mask** (Tier 2 calibration): sky-coordinate (az, alt) mask of
  domes, trees, and other fixed horizon intrusions, built from many clear-sky
  frames via `allclear calibrate obscuration`. Auto-loaded from a
  `<model>_obscuration.json` sidecar and wired through solve, transmission, and
  plotting so occluded sky is excluded rather than mismeasured.
- **`fix_center` solve option**: pins the optical center and frees `az0`
  per frame, curing per-frame center wander that previously false-reddened the
  zenith. Improves both match count and RMS.
- **Outer-envelope horizon detection**: scans inward and outward with
  outward-biased rejection so domes/trees (which only intrude inward) no longer
  pull the fitted horizon radius inward.
- **3-state blind-solve quality gate** (pass / marginal / fail), persisted in
  the instrument model JSON.
- **Satellite link-obscuration queries** on `SkyTransmissionResult` (optional
  `sgp4` dependency, installable via the `satellite` extra).
- **`manual-fit` GUI**: interactive bootstrap by clicking known objects.
- **`animate` subcommand**: assemble many frames into a timelapse, either by
  re-solving raw frames (`--frames` + `--model`) or stitching a previous
  `solve --output-dir` run (`--input-dir`). Outputs `.webp`/`.gif`/`.mp4`
  (mp4 needs `ffmpeg`), with `--mode solved|transmission|extinction`.
- **Extinction overlay mode** and an inset colorbar on annotated frames; frame
  timestamp and clip-margin-aware grid labels.

### Changed
- **Transmission now always normalizes to the model reference zeropoint** when
  one is provided (per-frame zeropoint is used only as a fallback). The previous
  per-frame upgrade could let a cloudy frame redefine "clear" and render an
  overcast sky all-green.
- Catalog download fallback now uses VizieR via `astroquery` (lazy import); the
  bright-star catalog ships with the package, so this path is only needed when
  building a custom catalog.

### Removed
- **`pyvo` dependency** — it was declared but never used.
- **`plotting_inkblot` module** and its `inkblot` dependency — an experimental
  rendering backend, superseded by the matplotlib renderer.

### Packaging
- Slimmer source distribution: `benchmark/`, `paper/`, `docs/`, `temp/`, the
  test suite, and project working notes are no longer included in the sdist or
  wheel.

## [0.2.2] - earlier

Initial public releases on PyPI (0.2.0, 0.2.1, 0.2.2): blind camera solve,
fast per-frame solve with a known model, and per-star transmission mapping.

[0.3.0]: https://github.com/zgazak/allclear/releases/tag/v0.3.0
