#!/usr/bin/env python3
"""Download APICAM frames for a pointing-drift study.

Two modes:
  --mode seasonal  One frame per night for ~1 year, same hour each night.
                   Shows long-term (seasonal) thermal flexure trends.
  --mode nightly   Every frame from a single clear night.
                   Shows the within-night thermal cooling curve.

The resulting frames are processed by analyze_pointing_drift.py to
extract per-frame pointing deltas and plot time-series.

Requirements: pip install pyvo requests
No ESO authentication needed — APICAM data is fully public.

Usage:
    python fetch_apicam_drift_study.py --mode seasonal [--dest DIR] [--dry-run]
    python fetch_apicam_drift_study.py --mode nightly  [--dest DIR] [--night 2019-06-15]
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import pyvo
import requests

ESO_TAP_URL = "http://archive.eso.org/tap_obs"

# APICAM archive spans 2018-03-01 to 2020-06-25 (~118K frames).
# For seasonal sampling, we pick one frame per night at a consistent
# hour (~01:00 UT = ~22:00 local Chile) to minimize sky-rotation variation.
SEASONAL_DATE_RANGE = ("2019-01-01", "2019-12-31")
SEASONAL_TARGET_HOUR = 1  # UT hour to sample (middle of Chilean night)

# Default night for nightly mode — 2019-06-15 is a clear winter night
# with 244 frames spanning 00:02 to 23:57 UT.
DEFAULT_NIGHT = "2019-06-15"


def query_seasonal_frames(tap, date_start, date_end, target_hour=1):
    """Query one frame per night at a consistent hour across a date range.

    Returns a list of (dp_id, date_obs) tuples.
    """
    # Get all frames within ±30 min of target hour
    query = f"""
    SELECT dp_id, date_obs
    FROM dbo.raw
    WHERE instrument = 'APICAM'
      AND date_obs BETWEEN '{date_start}' AND '{date_end}'
    ORDER BY date_obs
    """
    print(f"Querying APICAM frames {date_start} to {date_end} ...")
    results = tap.search(query, maxrec=200000)
    table = results.to_table()
    print(f"  Total frames in range: {len(table)}")

    # Group by night and pick the frame closest to target_hour UT
    from collections import defaultdict
    from astropy.time import Time

    nights = defaultdict(list)
    for row in table:
        t = Time(row["date_obs"], scale="utc")
        # Night = date of evening (frames before 12 UT belong to previous night)
        night_date = t.datetime.strftime("%Y-%m-%d")
        hour_ut = t.datetime.hour + t.datetime.minute / 60.0
        nights[night_date].append((row["dp_id"], row["date_obs"], hour_ut))

    # Pick one frame per night closest to target hour
    selected = []
    for night_date in sorted(nights.keys()):
        frames = nights[night_date]
        # Filter to nighttime frames (before 10 UT = before ~7am local)
        nighttime = [f for f in frames if f[2] < 10]
        if not nighttime:
            continue
        best = min(nighttime, key=lambda f: abs(f[2] - target_hour))
        selected.append((best[0], best[1]))

    print(f"  Selected {len(selected)} frames (1 per night at ~{target_hour:02d}:00 UT)")
    return selected


def query_nightly_frames(tap, night_date):
    """Query all frames from a single night.

    Returns a list of (dp_id, date_obs) tuples.
    """
    next_day = f"{night_date[:8]}{int(night_date[8:10])+1:02d}"
    query = f"""
    SELECT dp_id, date_obs
    FROM dbo.raw
    WHERE instrument = 'APICAM'
      AND date_obs BETWEEN '{night_date}' AND '{next_day}'
    ORDER BY date_obs
    """
    print(f"Querying all APICAM frames on {night_date} ...")
    results = tap.search(query, maxrec=1000)
    table = results.to_table()

    # Filter to nighttime only (UT 0-10, roughly)
    from astropy.time import Time
    selected = []
    for row in table:
        t = Time(row["date_obs"], scale="utc")
        hour_ut = t.datetime.hour + t.datetime.minute / 60.0
        if hour_ut < 10:
            selected.append((row["dp_id"], row["date_obs"]))

    print(f"  Found {len(table)} total frames, {len(selected)} nighttime frames")
    return selected


def download_frame(dp_id, dest_dir, timeout=120):
    """Download a single frame from ESO data portal.

    ESO returns Unix-compressed (.Z) files; we decompress automatically.
    """
    url = f"https://dataportal.eso.org/dataPortal/file/{dp_id}"
    fname = f"{dp_id}.fits"
    dest = dest_dir / fname

    if dest.exists():
        return dest

    print(f"  {dp_id} ...", end=" ", flush=True)
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        # ESO portal returns .Z compressed files — detect and decompress
        with open(dest, "rb") as f:
            magic = f.read(2)
        if magic == b"\x1f\x9d":
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
    parser = argparse.ArgumentParser(
        description="Download APICAM frames for pointing-drift study"
    )
    parser.add_argument(
        "--mode",
        choices=["seasonal", "nightly"],
        required=True,
        help="seasonal: 1 frame/night for ~1 year; nightly: all frames from one night",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: benchmark/data/apicam_drift_{mode})",
    )
    parser.add_argument(
        "--night",
        default=DEFAULT_NIGHT,
        help=f"Night date for nightly mode (default: {DEFAULT_NIGHT})",
    )
    parser.add_argument(
        "--date-start",
        default=SEASONAL_DATE_RANGE[0],
        help=f"Start date for seasonal mode (default: {SEASONAL_DATE_RANGE[0]})",
    )
    parser.add_argument(
        "--date-end",
        default=SEASONAL_DATE_RANGE[1],
        help=f"End date for seasonal mode (default: {SEASONAL_DATE_RANGE[1]})",
    )
    parser.add_argument(
        "--target-hour",
        type=float,
        default=SEASONAL_TARGET_HOUR,
        help=f"Target UT hour for seasonal sampling (default: {SEASONAL_TARGET_HOUR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Query only, don't download")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else Path(f"benchmark/data/apicam_drift_{args.mode}")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ESO TAP at {ESO_TAP_URL} ...")
    tap = pyvo.dal.TAPService(ESO_TAP_URL)

    if args.mode == "seasonal":
        frames = query_seasonal_frames(
            tap, args.date_start, args.date_end, args.target_hour
        )
    else:
        frames = query_nightly_frames(tap, args.night)

    if args.max_frames and len(frames) > args.max_frames:
        # Subsample evenly
        step = len(frames) / args.max_frames
        frames = [frames[int(i * step)] for i in range(args.max_frames)]
        print(f"  Subsampled to {len(frames)} frames")

    print(f"\n{'='*60}")
    print(f"Selected {len(frames)} frames → {dest}")

    # Write manifest
    manifest = dest / "manifest.txt"
    with open(manifest, "w") as f:
        f.write(f"# APICAM drift study ({args.mode})\n")
        f.write("# dp_id  date_obs\n")
        for dp_id, date_obs in frames:
            f.write(f"{dp_id}  {date_obs}\n")

    if args.dry_run:
        for dp_id, date_obs in frames:
            print(f"  {dp_id}  {date_obs}")
        print(f"\nManifest written to {manifest}")
        return

    # Download
    downloaded = 0
    for i, (dp_id, date_obs) in enumerate(frames):
        print(f"[{i+1}/{len(frames)}]", end=" ")
        result = download_frame(dp_id, dest)
        if result is not None:
            downloaded += 1
        time.sleep(0.5)

    print(f"\nDone: {downloaded}/{len(frames)} frames downloaded to {dest}")
    print(f"Manifest: {manifest}")
    print(f"\nNext step: run analyze_pointing_drift.py on these frames")


if __name__ == "__main__":
    main()
