#!/bin/bash
# Refresh paper/figures/ from the latest generated figures under
# benchmark/results/.  The figure generators write into
# benchmark/results/**; this copies each figure already present in
# paper/figures/ from its newest same-named source, so re-running a
# generator + this script keeps the paper's figures current.
#
# Usage:  bash benchmark/scripts/sync_paper_figures.sh
set -euo pipefail
cd "$(dirname "$0")/../.."            # repo root
n=0
for f in paper/figures/*.png paper/figures/*.pdf; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    src=$(find benchmark/results -name "$base" -type f -print -quit 2>/dev/null || true)
    if [ -n "$src" ] && [ "$src" -nt "$f" ]; then
        cp "$src" "$f"
        echo "synced  $base  <-  $src"
        n=$((n + 1))
    fi
done
echo "Done. $n figure(s) refreshed."
