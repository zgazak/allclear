#!/usr/bin/env python3
"""Download a curated sample of Liverpool Telescope SkyCam-A all-sky frames.

Grabs gzipped FITS files from the publicly browsable LT archive.
Picks frames spread across multiple nights and seasons for condition variety.

Camera: Starlight Xpress Oculus (Sony ICX267AL CCD, 1392x1040, 4.65um pixels),
1.55mm f/2.0 fisheye, 30s exposures, dark/flat corrected.
La Palma, Spain (28.762, -17.879).

Requirements: pip install requests beautifulsoup4
No authentication needed — SkyCam-A data is immediately public.

Usage:
    python fetch_liverpool_skycam.py [--dest benchmark/data/liverpool_skycam] [--n-frames 30]
"""

import argparse
import gzip
import re
import shutil
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

LT_ARCHIVE_BASE = "https://telescope.livjm.ac.uk/data/archive/data/lt/Skycam"

# Sample nights spread across seasons and years.
# We'll scrape the directory listing for each night and pick a few SkyCam-A frames.
SAMPLE_NIGHTS = [
    # Northern winter (Dec-Feb) — more weather variety
    "20241215",
    "20250115",
    "20250210",
    # Northern summer (Jun-Aug) — generally drier
    "20240620",
    "20240715",
    "20240810",
    # Transition months
    "20240415",
    "20241015",
    "20250315",
    # Additional spread
    "20240520",
    "20241120",
    "20250105",
]

FRAMES_PER_NIGHT = 3


def list_skycam_a_frames(night_date):
    """Scrape the archive directory for SkyCam-A FITS files on a given night."""
    year = night_date[:4]
    url = f"{LT_ARCHIVE_BASE}/{year}/{night_date}/"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: could not list {night_date}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    fits_files = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # SkyCam-A files match a_e_YYYYMMDD_NN_1_1_1.fits.gz
        if re.match(r"a_e_\d{8}_\d+_1_1_1\.fits\.gz$", href):
            fits_files.append(href)

    return sorted(fits_files)


def pick_spread(files, n=3):
    """Pick n files spread evenly through the list."""
    if not files:
        return []
    indices = [int(i * (len(files) - 1) / max(n - 1, 1)) for i in range(n)]
    indices = sorted(set(indices))
    return [files[i] for i in indices]


def download_frame(night_date, filename, dest_dir, decompress=True, timeout=120):
    """Download and optionally decompress a single SkyCam-A frame."""
    year = night_date[:4]
    url = f"{LT_ARCHIVE_BASE}/{year}/{night_date}/{filename}"

    if decompress and filename.endswith(".gz"):
        out_name = filename[:-3]  # strip .gz
    else:
        out_name = filename

    dest = dest_dir / out_name
    if dest.exists():
        print(f"  Already have {out_name}")
        return dest

    gz_dest = dest_dir / filename

    print(f"  Downloading {filename} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(gz_dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
    except Exception as e:
        print(f"FAILED: {e}")
        if gz_dest.exists():
            gz_dest.unlink()
        return None

    if decompress and filename.endswith(".gz"):
        try:
            with gzip.open(gz_dest, "rb") as f_in:
                with open(dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_dest.unlink()
        except Exception as e:
            print(f"decompress failed: {e}")
            if dest.exists():
                dest.unlink()
            return None

    size_mb = dest.stat().st_size / 1e6
    print(f"{size_mb:.1f} MB")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Liverpool SkyCam-A benchmark frames"
    )
    parser.add_argument("--dest", default="benchmark/data/liverpool_skycam",
                        help="Destination directory")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Target number of frames (approx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List available frames, don't download")
    parser.add_argument("--keep-gz", action="store_true",
                        help="Keep gzipped files instead of decompressing")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    all_picks = []  # list of (night_date, filename)

    for night in SAMPLE_NIGHTS:
        print(f"\nListing SkyCam-A frames for {night} ...")
        frames = list_skycam_a_frames(night)
        if frames:
            picks = pick_spread(frames, n=FRAMES_PER_NIGHT)
            print(f"  Found {len(frames)} frames, picking {len(picks)}")
            all_picks.extend((night, f) for f in picks)
        else:
            print("  No frames found")
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Selected {len(all_picks)} frames total")

    if args.dry_run:
        for night, fname in all_picks:
            print(f"  {night}/{fname}")
        return

    # Write manifest
    manifest = dest / "manifest.txt"
    with open(manifest, "w") as f:
        f.write("# Liverpool SkyCam-A benchmark frames\n")
        f.write("# night_date  filename\n")
        for night, fname in all_picks:
            f.write(f"{night}  {fname}\n")

    # Download
    downloaded = 0
    for night, fname in all_picks:
        result = download_frame(night, fname, dest,
                                decompress=not args.keep_gz)
        if result is not None:
            downloaded += 1
        time.sleep(0.5)

    print(f"\nDone: {downloaded}/{len(all_picks)} frames downloaded to {dest}")


if __name__ == "__main__":
    main()
