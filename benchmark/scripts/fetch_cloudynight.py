#!/usr/bin/env python3
"""Download the cloudynight example dataset (Mommert 2020).

20 bzip2-compressed FITS frames from the Lowell Discovery Telescope all-sky
camera, plus the cloud labels and obstruction mask.

Camera: Starlight Xpress Oculus (1392x1040 CCD, 1.55mm f/1.2 fisheye),
60s exposures.  Flagstaff, AZ (35.0969, -111.5350, ~2360m).

Source: https://github.com/mommermi/cloudynight
License: BSD-3-Clause (per repository)

Usage:
    python fetch_cloudynight.py [--dest benchmark/data/cloudynight]
"""

import argparse
import bz2
import shutil
from pathlib import Path

import requests

GITHUB_RAW = "https://raw.githubusercontent.com/mommermi/cloudynight/master"

# 20 example FITS images + mask + labels
FITS_FILES = [f"{i:03d}.fits.bz2" for i in range(20)]
EXTRA_FILES = ["mask.fits", "y_train.dat"]


def download_file(url, dest, timeout=120):
    """Download a single file."""
    if dest.exists():
        print(f"  Already have {dest.name}")
        return dest

    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        print(f"{size_mb:.1f} MB")
        return dest
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return None


def decompress_bz2(src, dst):
    """Decompress a .bz2 file."""
    with bz2.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    src.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch cloudynight benchmark dataset (Mommert 2020)"
    )
    parser.add_argument("--dest", default="benchmark/data/cloudynight",
                        help="Destination directory")
    parser.add_argument("--keep-bz2", action="store_true",
                        help="Keep bz2 files instead of decompressing")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    # FITS images
    for fname in FITS_FILES:
        url = f"{GITHUB_RAW}/example_data/images/{fname}"
        out_path = dest / fname
        result = download_file(url, out_path)
        if result and not args.keep_bz2:
            fits_path = dest / fname.replace(".bz2", "")
            if not fits_path.exists():
                print(f"  Decompressing {fname} ...")
                decompress_bz2(out_path, fits_path)
        if result:
            downloaded += 1

    # Mask and labels
    for fname in EXTRA_FILES:
        url = f"{GITHUB_RAW}/example_data/images/{fname}"
        result = download_file(url, dest / fname)
        if result:
            downloaded += 1

    # Also grab the config to know the subregion layout
    url = f"{GITHUB_RAW}/cloudynight/conf.py"
    download_file(url, dest / "cloudynight_conf.py")

    print(f"\nDone: {downloaded}/{len(FITS_FILES) + len(EXTRA_FILES)} files "
          f"downloaded to {dest}")

    # Write camera metadata
    meta = dest / "camera_info.txt"
    with open(meta, "w") as f:
        f.write("# Cloudynight camera metadata (Mommert 2020)\n")
        f.write("site_name = Lowell Discovery Telescope\n")
        f.write("latitude = 35.0969\n")
        f.write("longitude = -111.5350\n")
        f.write("altitude_m = 2360\n")
        f.write("sensor = Starlight Xpress Oculus (Sony ICX267AL CCD)\n")
        f.write("resolution = 1392x1040\n")
        f.write("pixel_size_um = 4.65\n")
        f.write("lens_focal_mm = 1.55\n")
        f.write("lens_aperture = f/1.2\n")
        f.write("fov_deg = 180\n")
        f.write("exposure_s = 60\n")
        f.write("bit_depth = 16\n")
        f.write("mount = fixed\n")

    print(f"Camera metadata written to {meta}")


if __name__ == "__main__":
    main()
