#!/usr/bin/env python3
"""Download a curated sample of ESO APICAM all-sky frames from Paranal.

Queries the ESO TAP archive for APICAM frames and downloads a stratified
sample spanning different sky conditions (clear, thin cloud, partial,
overcast).  Because we can't pre-filter by cloud cover, we grab frames
spread across many nights and let the human labeler sort them.

Camera: KAF-16803 CCD (4096x4096, 9um pixels), Canon 12mm fisheye,
120s tracked exposures.  Paranal, Chile (-24.6272, -70.4048).

Requirements: pip install pyvo requests
No ESO authentication needed — APICAM data is fully public.

Usage:
    python fetch_eso_apicam.py [--dest benchmark/data/eso_apicam] [--n-frames 30]
"""

import argparse
import os
import time
from pathlib import Path

import pyvo
import requests

ESO_TAP_URL = "http://archive.eso.org/tap_obs"

# We pick nights spread across seasons and moon phases to get variety.
# These are date ranges that should each yield several nighttime frames.
# We'll grab 2-3 frames per night from different times to catch varying
# conditions.
SAMPLE_NIGHTS = [
    # Southern winter (June-Aug) — dry season, generally clearer
    ("2019-06-15", "2019-06-16"),
    ("2019-07-20", "2019-07-21"),
    ("2019-08-10", "2019-08-11"),
    # Southern summer (Dec-Feb) — wetter, more clouds
    ("2019-12-20", "2019-12-21"),
    ("2020-01-15", "2020-01-16"),
    ("2019-02-10", "2019-02-11"),
    # Transition months
    ("2019-04-05", "2019-04-06"),
    ("2019-10-12", "2019-10-13"),
    # Additional spread
    ("2018-09-15", "2018-09-16"),
    ("2019-03-22", "2019-03-23"),
    ("2019-11-08", "2019-11-09"),
    ("2020-05-01", "2020-05-02"),
]

FRAMES_PER_NIGHT = 3  # early, mid, late in the night


def query_night(tap, date_start, date_end, max_rows=50):
    """Query APICAM frames for one night."""
    query = f"""
    SELECT dp_id, date_obs, exposure, origfile
    FROM dbo.raw
    WHERE instrument = 'APICAM'
      AND date_obs BETWEEN '{date_start}' AND '{date_end}'
    ORDER BY date_obs
    """
    try:
        results = tap.search(query, maxrec=max_rows)
        return results.to_table()
    except Exception as e:
        print(f"  Warning: query failed for {date_start}: {e}")
        return None


def pick_spread(table, n=3):
    """Pick n frames spread evenly through the night."""
    if table is None or len(table) == 0:
        return []
    indices = [int(i * (len(table) - 1) / max(n - 1, 1)) for i in range(n)]
    # deduplicate
    indices = sorted(set(indices))
    return [table[i] for i in indices]


def download_frame(dp_id, dest_dir, timeout=120):
    """Download a single frame from ESO data portal.

    ESO returns Unix-compressed (.Z) files; we decompress automatically.
    """
    import subprocess

    url = f"https://dataportal.eso.org/dataPortal/file/{dp_id}"
    fname = f"{dp_id}.fits"
    dest = dest_dir / fname

    if dest.exists():
        print(f"  Already have {fname}")
        return dest

    print(f"  Downloading {dp_id} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        # ESO portal returns .Z compressed files — detect and decompress
        with open(dest, "rb") as f:
            magic = f.read(2)
        if magic == b"\x1f\x9d":  # Unix compress magic number
            z_path = dest.with_suffix(".fits.Z")
            dest.rename(z_path)
            subprocess.run(["uncompress", str(z_path)], check=True)

        size_mb = dest.stat().st_size / 1e6
        print(f"{size_mb:.1f} MB")
        return dest
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch ESO APICAM benchmark frames")
    parser.add_argument("--dest", default="benchmark/data/eso_apicam",
                        help="Destination directory")
    parser.add_argument("--n-frames", type=int, default=30,
                        help="Target number of frames (approx)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Query only, don't download")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ESO TAP at {ESO_TAP_URL} ...")
    tap = pyvo.dal.TAPService(ESO_TAP_URL)

    all_frames = []
    for date_start, date_end in SAMPLE_NIGHTS:
        print(f"\nQuerying {date_start} ...")
        table = query_night(tap, date_start, date_end)
        if table is not None and len(table) > 0:
            picks = pick_spread(table, n=FRAMES_PER_NIGHT)
            print(f"  Found {len(table)} frames, picking {len(picks)}")
            all_frames.extend(picks)
        else:
            print("  No frames found")
        time.sleep(1)  # be polite to ESO

    print(f"\n{'='*60}")
    print(f"Selected {len(all_frames)} frames total")

    if args.dry_run:
        for row in all_frames:
            print(f"  {row['dp_id']}  {row['date_obs']}")
        return

    # Write manifest
    manifest = dest / "manifest.txt"
    with open(manifest, "w") as f:
        f.write("# ESO APICAM benchmark frames\n")
        f.write("# dp_id  date_obs\n")
        for row in all_frames:
            f.write(f"{row['dp_id']}  {row['date_obs']}\n")

    # Download
    downloaded = 0
    for row in all_frames:
        result = download_frame(row["dp_id"], dest)
        if result is not None:
            downloaded += 1
        time.sleep(0.5)  # rate limiting

    print(f"\nDone: {downloaded}/{len(all_frames)} frames downloaded to {dest}")


if __name__ == "__main__":
    main()
