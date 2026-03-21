# AllClear Benchmark Dataset

Multi-camera, multi-site benchmark for all-sky camera astrometric calibration
and cloud transmission mapping.

## Dataset structure

```
benchmark/
  data/
    haleakala/          # ZWO ASI178MM, 3096x2080, 1.8mm fisheye (fixed mount)
    eso_apicam/         # KAF-16803 CCD, 4096x4096, Canon 12mm fisheye (tracking mount)
    liverpool_skycam/   # Sony ICX267AL CCD, 1392x1040, 1.55mm f/2.0 (fixed mount)
    cloudynight/        # Starlight Xpress Oculus, 1392x1040, 1.55mm f/1.2 (fixed mount)
  labels/               # Human-annotated ground truth (JSON per frame)
  scripts/              # Reproducible download scripts
  solutions/            # AllClear astrometric solutions (output)
```

## Cameras

| ID | Sensor | Resolution | Pixel (um) | Lens | Exposure | Mount | Site | Lat | Lon |
|----|--------|-----------|------------|------|----------|-------|------|-----|-----|
| haleakala | ZWO ASI178MM (IMX178) | 3096x2080 | 2.4 | 1.8mm fisheye | variable | fixed | Haleakala, HI | 20.746 | -156.432 |
| eso_apicam | KAF-16803 CCD | 4096x4096 | 9.0 | Canon 12mm | 120s | tracking | Paranal, Chile | -24.627 | -70.405 |
| liverpool_skycam | Sony ICX267AL CCD | 1392x1040 | 4.65 | 1.55mm f/2.0 | 30s | fixed | La Palma, Spain | 28.762 | -17.879 |
| cloudynight | Starlight Xpress Oculus | 1392x1040 | 4.65 | 1.55mm f/1.2 | 60s | fixed | Flagstaff, AZ | 35.097 | -111.535 |

## Obtaining the data

Raw FITS frames are not redistributed. Use the provided scripts to download
from original sources:

```bash
# ESO APICAM (requires: pip install pyvo requests)
python benchmark/scripts/fetch_eso_apicam.py

# Liverpool SkyCam-A (requires: pip install requests beautifulsoup4)
python benchmark/scripts/fetch_liverpool_skycam.py

# cloudynight / Mommert 2020 (requires: pip install requests)
python benchmark/scripts/fetch_cloudynight.py

# Haleakala frames: copy from your own archive into benchmark/data/haleakala/
```

## Labels

Each frame has a JSON label in `labels/` following the schema in
`labels/label_schema.json`.  Labels include:

- **Global condition**: clear / thin_cloud / partial_cloud / mostly_cloudy / overcast / twilight
- **Per-quadrant condition**: NE / NW / SE / SW as projected on sky
- **Moon/planet presence**
- **Dome obstruction estimate**
- **Free-form notes**

## Data sources and licenses

- **ESO APICAM**: ESO Science Archive, public data.
  https://archive.eso.org (instrument=APICAM)
- **Liverpool SkyCam**: Liverpool Telescope archive, immediately public.
  https://telescope.livjm.ac.uk/SkyCam/
- **cloudynight**: Mommert (2020, AJ 159, 178), BSD-3-Clause.
  https://github.com/mommermi/cloudynight
- **Haleakala**: Original data collected by the authors.

## Using with AllClear

```bash
# Blind-solve a benchmark frame (instrument-fit)
uv run allclear instrument-fit \
    --frames benchmark/data/eso_apicam/APICAM.2019-06-15T*.fits \
    --lat -24.627 --lon -70.405 \
    --output benchmark/solutions/eso_apicam_model.json

# Fast-solve with known model
uv run allclear solve \
    --frames "benchmark/data/eso_apicam/*.fits" \
    --model benchmark/solutions/eso_apicam_model.json
```
