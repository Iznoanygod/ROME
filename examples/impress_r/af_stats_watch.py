#!/usr/bin/env python3
"""Read a live IMPRESS campaign's score CSVs and report the distribution.

Every confidence threshold in IMPRESS-R is predictor-specific — AlphaFold2 and
Boltz do not share a scale — so the thresholds have to come from the campaign
you are actually running. This reads whatever ``af_stats_*.csv`` exist so far
and tells you what a filter would admit.

    python examples/impress_r/af_stats_watch.py /path/to/campaign            # once
    python examples/impress_r/af_stats_watch.py /path/to/campaign --follow   # every 60s

Read-only: it never writes into the campaign. Safe to run against a directory
being written by a live job.

Output is the distribution, then a table of candidate thresholds with the
fraction each admits. Pick the row admitting roughly a third and pass it to
``impress_corpus_filter(...)`` — or skip the whole exercise and use
``percentile_sampler(0.33)``, which does this continuously and needs no numbers.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from collections import defaultdict

#: The schema `plddt_extract_pipeline.py` writes, and its direction of goodness.
COLUMNS = {"avg_plddt": "high", "ptm": "high", "avg_pae": "low"}

PASS_RE = re.compile(r"af_stats_(?P<pipeline>.+)_pass_(?P<pass>\d+)\.csv$")


def read_records(root: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(root, "**", "af_stats_*.csv"),
                                 recursive=True)):
        match = PASS_RE.search(os.path.basename(path))
        if not match:
            continue
        try:
            with open(path) as fd:
                for row in csv.DictReader(fd):
                    row["_pipeline"] = match.group("pipeline")
                    row["_pass"] = int(match.group("pass"))
                    records.append(row)
        except OSError:          # being written right now; catch it next tick
            continue
    return records


def numeric(records: list[dict], column: str) -> list[float]:
    out = []
    for row in records:
        try:
            out.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    values = sorted(values)

    def q(p: float) -> float:
        return values[min(len(values) - 1, int(p * len(values)))]

    return {"min": values[0], "p10": q(.10), "p25": q(.25), "median": q(.50),
            "p75": q(.75), "p90": q(.90), "max": values[-1]}


def report(root: str) -> int:
    records = read_records(root)
    if not records:
        print(f"no af_stats_*.csv under {root} yet")
        return 1

    passes = defaultdict(int)
    for row in records:
        passes[row["_pass"]] += 1
    pipelines = {row["_pipeline"] for row in records}

    print(f"campaign : {root}")
    print(f"records  : {len(records)} from {len(pipelines)} pipelines, "
          f"passes {min(passes)}-{max(passes)}")
    print("per pass : " + "  ".join(f"p{p}:{passes[p]}" for p in sorted(passes)))

    print(f"\n{'column':<12}{'n':>5}{'min':>9}{'p25':>9}{'median':>9}"
          f"{'p75':>9}{'p90':>9}{'max':>9}")
    stats = {}
    for column in COLUMNS:
        values = numeric(records, column)
        if not values:
            continue
        stats[column] = quantiles(values)
        q = stats[column]
        print(f"{column:<12}{len(values):>5}{q['min']:>9.3f}{q['p25']:>9.3f}"
              f"{q['median']:>9.3f}{q['p75']:>9.3f}{q['p90']:>9.3f}{q['max']:>9.3f}")

    if not stats:
        print("\nno recognised score columns; expected: " + ", ".join(COLUMNS))
        return 1

    # What a fixed filter would admit, swept across the observed distribution.
    print(f"\n{'thresholds (plddt / ptm / pae)':<34}{'admits':>16}")
    total = len(records)
    for keep in (0.75, 0.50, 0.33, 0.25, 0.10):
        # Set each clause at the percentile that alone would keep `keep`, then
        # measure the joint effect — the clauses correlate, so it is always
        # stricter than `keep`.
        cuts = {}
        for column, direction in COLUMNS.items():
            values = sorted(numeric(records, column))
            if not values:
                continue
            index = int((1 - keep) * len(values)) if direction == "high" \
                else int(keep * len(values))
            cuts[column] = values[min(len(values) - 1, index)]
        if len(cuts) < len(COLUMNS):
            continue
        admitted = sum(
            1 for row in records
            if float(row["avg_plddt"]) >= cuts["avg_plddt"]
            and float(row["ptm"]) >= cuts["ptm"]
            and float(row["avg_pae"]) <= cuts["avg_pae"]
        )
        label = (f"{cuts['avg_plddt']:.1f} / {cuts['ptm']:.3f} / "
                 f"{cuts['avg_pae']:.2f}")
        print(f"{label:<34}{admitted:>6}/{total:<4} ({100*admitted/total:3.0f}%)")

    print("\nNote each row sets its three clauses at the percentile that would\n"
          "admit that fraction *alone*. The joint result is always stricter,\n"
          "often far stricter, because the three scores correlate — which is\n"
          "why picking three numbers by hand rarely lands where you intended.")

    print("\nimpress_corpus_filter defaults (80 / 0.80 / 5.0) would admit: ", end="")
    stock = sum(
        1 for row in records
        if float(row["avg_plddt"]) >= 80.0 and float(row["ptm"]) >= 0.80
        and float(row["avg_pae"]) <= 5.0
    )
    print(f"{stock}/{total} ({100*stock/total:.0f}%)")
    if stock > 0.6 * total:
        print("  -> too permissive to select anything. Use a row above, or\n"
              "     percentile_sampler(0.33), which needs no thresholds.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", help="campaign directory (searched recursively)")
    ap.add_argument("--follow", action="store_true", help="re-read on an interval")
    ap.add_argument("--interval", type=int, default=60, help="seconds (default 60)")
    args = ap.parse_args()

    if not args.follow:
        return report(args.campaign)

    while True:
        print("\033[2J\033[H", end="")          # clear, so the table stays put
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report(args.campaign)
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
