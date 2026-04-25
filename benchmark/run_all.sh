#!/bin/bash
# Run instrument-fit on all benchmark cameras + Liverpool variants.
# All output goes to benchmark/solutions/<name>_log.txt, images to benchmark/solutions/.
set -e
mkdir -p benchmark/solutions

echo "Starting all instrument-fit runs..."

# Liverpool — no moon
uv run allclear instrument-fit \
    --frames benchmark/data/liverpool_skycam/a_e_20240519_530_1_1_1.fits \
    --lat 28.762 --lon -17.879 \
    --output benchmark/solutions/liverpool.json --verbose \
    --diagnostic benchmark/solutions/liverpool-diag.png \
    > benchmark/solutions/liverpool_log.txt 2>&1 &
echo "  liverpool started (pid $!)"

# Haleakala
uv run allclear instrument-fit \
    --frames benchmark/data/haleakala/2023_11_19__00_01_19.fits \
    --lat 20.7458 --lon -156.4317 \
    --output benchmark/solutions/haleakala.json --verbose \
    --diagnostic benchmark/solutions/haleakala-diag.png \
    > benchmark/solutions/haleakala_log.txt 2>&1 &
echo "  haleakala started (pid $!)"

# Cloudynight
uv run allclear instrument-fit \
    --frames benchmark/data/cloudynight/005.fits \
    --lat 35.1983 --lon -111.6513 \
    --output benchmark/solutions/cloudynight.json --verbose \
    --diagnostic benchmark/solutions/cloudynight-diag.png \
    > benchmark/solutions/cloudynight_log.txt 2>&1 &
echo "  cloudynight started (pid $!)"

# APICAM
uv run allclear instrument-fit \
    --frames 'benchmark/data/eso_apicam/APICAM.2019-02-10T01:52:35.000.fits' \
    --lat -24.6272 --lon -70.4048 \
    --output benchmark/solutions/apicam.json --verbose \
    --diagnostic benchmark/solutions/apicam-diag.png \
    > benchmark/solutions/apicam_log.txt 2>&1 &
echo "  apicam started (pid $!)"

echo ""
echo "All jobs launched. Waiting..."
wait
echo ""
echo "=== RESULTS ==="
for f in benchmark/solutions/*_log.txt; do
    name=$(basename "$f" _log.txt)
    final=$(grep "^Final:" "$f" 2>/dev/null || echo "NO RESULT")
    valid=$(grep "Validation:" "$f" 2>/dev/null || echo "")
    status=$(grep -E "^  (OK|FAILED):" "$f" 2>/dev/null | head -1 || echo "")
    printf "  %-20s %s\n" "$name" "$final"
    [ -n "$valid" ] && printf "  %-20s %s\n" "" "$valid"
    [ -n "$status" ] && printf "  %-20s %s\n" "" "$status"
done
