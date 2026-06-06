#!/usr/bin/env python
"""Compare two seasonal blind-solve runs (blind_solve_results.csv).

Separates true verdict flips (both runs actually gated the frame) from
first-time verdicts (frames that were 'cached' in the old run and never
gate-evaluated). Also reports f-distribution shift and the
horizon-broken tally.

Usage:
    python compare_seasonal_runs.py OLD_DIR NEW_DIR
"""
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(d):
    rows = {}
    with open(Path(d) / "blind_solve_results.csv") as fh:
        for row in csv.DictReader(fh):
            rows[row["filename"]] = row
    return rows


def cls(status):
    if status in ("ok", "cached"):
        return "pass"
    if status.startswith("failed") or status.startswith("error") \
            or status.startswith("load_error"):
        return "fail"
    return status  # marginal


def main(old_dir, new_dir):
    old, new = load(old_dir), load(new_dir)

    print(f"old: {len(old)} frames   new: {len(new)} frames\n")
    for name, rows in (("old", old), ("new", new)):
        c = Counter(cls(r["status"]) for r in rows.values())
        hb = sum(1 for r in rows.values() if "horizon-broken" in r["status"])
        cached = sum(1 for r in rows.values() if r["status"] == "cached")
        note = f" (incl. {cached} cached, never gated)" if cached else ""
        print(f"{name}: {dict(c)}{note}, horizon-broken fails: {hb}")

    flips, first_time = [], []
    dn, df = [], []
    for fn, nr in new.items():
        orow = old.get(fn)
        if orow is None:
            continue
        ncls, ocls = cls(nr["status"]), cls(orow["status"])
        if nr["status"] not in ("cached",) and orow["status"] != "cached":
            if ncls != ocls:
                flips.append((fn, orow["status"], nr["status"]))
        elif orow["status"] == "cached":
            first_time.append((fn, ncls))
        try:
            if ncls != "fail" and cls(orow["status"]) != "fail":
                dn.append(int(nr["n_matched"]) - int(orow["n_matched"]))
                df.append(float(nr["f"]) - float(orow["f"]))
        except (ValueError, KeyError):
            pass

    print(f"\nTRUE FLIPS (both runs gated): {len(flips)}")
    for fn, a, b in sorted(flips):
        print(f"  {fn}\n      {a}\n   -> {b}")

    ft = Counter(c for _, c in first_time)
    print(f"\nFirst-time verdicts (old=cached): {dict(ft)}")

    if dn:
        print(f"\nSolved-frame deltas (n={len(dn)}):")
        print(f"  dn_matches: median {statistics.median(dn):+.0f}, "
              f"mean {statistics.mean(dn):+.1f}")
        print(f"  df:         median {statistics.median(df):+.2f}px, "
              f"mean {statistics.mean(df):+.2f}px, "
              f"max |df| {max(abs(x) for x in df):.2f}px")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
