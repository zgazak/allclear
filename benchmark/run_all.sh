#!/bin/bash
# Run instrument-fit on all benchmark cameras + Liverpool variants.
# All output goes to temp/<name>_log.txt, images to temp/.
set -e
mkdir -p temp

echo "Starting all instrument-fit runs..."

# Liverpool — moon
uv run allclear instrument-fit \
    --frames benchmark/data/liverpool_skycam/a_e_20250105_9_1_1_1.fits \
    --lat 28.762 --lon -17.879 \
    --output temp/liverpool_moon.json --verbose \
    --diagnostic temp/liverpool_moon-diag.png \
    > temp/liverpool_moon_log.txt 2>&1 &
echo "  liverpool_moon started (pid $!)"

# Liverpool — no moon
uv run allclear instrument-fit \
    --frames benchmark/data/liverpool_skycam/a_e_20240519_530_1_1_1.fits \
    --lat 28.762 --lon -17.879 \
    --output temp/liverpool_nomoon.json --verbose \
    --diagnostic temp/liverpool_nomoon-diag.png \
    > temp/liverpool_nomoon_log.txt 2>&1 &
echo "  liverpool_nomoon started (pid $!)"

# Liverpool — low moon
uv run allclear instrument-fit \
    --frames benchmark/data/liverpool_skycam/a_e_20240414_578_1_1_1.fits \
    --lat 28.762 --lon -17.879 \
    --output temp/liverpool_lowmoon.json --verbose \
    --diagnostic temp/liverpool_lowmoon-diag.png \
    > temp/liverpool_lowmoon_log.txt 2>&1 &
echo "  liverpool_lowmoon started (pid $!)"

# Haleakala
uv run allclear instrument-fit \
    --frames benchmark/data/haleakala/2023_11_19__00_01_19.fits \
    --lat 20.7458 --lon -156.4317 \
    --output temp/haleakala.json --verbose \
    --diagnostic temp/haleakala-diag.png \
    > temp/haleakala_log.txt 2>&1 &
echo "  haleakala started (pid $!)"

# Haleakala alt frame
uv run allclear instrument-fit \
    --frames benchmark/data/haleakala/2023_11_19__00_34_14.fits \
    --lat 20.7458 --lon -156.4317 \
    --output temp/haleakala_alt.json --verbose \
    --diagnostic temp/haleakala_alt-diag.png \
    > temp/haleakala_alt_log.txt 2>&1 &
echo "  haleakala_alt started (pid $!)"

# Cloudynight
uv run allclear instrument-fit \
    --frames benchmark/data/cloudynight/005.fits \
    --lat 35.1983 --lon -111.6513 \
    --output temp/cloudynight.json --verbose \
    --diagnostic temp/cloudynight-diag.png \
    > temp/cloudynight_log.txt 2>&1 &
echo "  cloudynight started (pid $!)"

# APICAM
uv run allclear instrument-fit \
    --frames 'benchmark/data/eso_apicam/APICAM.2019-02-10T01:52:35.000.fits' \
    --lat -24.6272 --lon -70.4048 \
    --output temp/apicam.json --verbose \
    --diagnostic temp/apicam-diag.png \
    > temp/apicam_log.txt 2>&1 &
echo "  apicam started (pid $!)"

echo ""
echo "All jobs launched. Waiting..."
wait
echo ""
echo "=== RESULTS ==="
for f in temp/*_log.txt; do
    name=$(basename "$f" _log.txt)
    final=$(grep "^Final:" "$f" 2>/dev/null || echo "NO RESULT")
    valid=$(grep "Validation:" "$f" 2>/dev/null || echo "")
    status=$(grep -E "^  (OK|FAILED):" "$f" 2>/dev/null | head -1 || echo "")
    printf "  %-20s %s\n" "$name" "$final"
    [ -n "$valid" ] && printf "  %-20s %s\n" "" "$valid"
    [ -n "$status" ] && printf "  %-20s %s\n" "" "$status"
done
