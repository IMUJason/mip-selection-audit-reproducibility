#!/usr/bin/env python
"""Regime diagnostics for the Study-2 development and hold-out splits.

Reproduces the regime-conditioned mean log-gap deltas quoted in Section 5
from the released Study-2 logs. Regime labels come from the portfolio-probe
record (mode ``raise_portfolio_ud`` carries the probe fields); gaps come from
the ``raise_cut``, ``default``, and ``scip_dynamic`` records of the same
instance, restricted to finite positive gaps (the restricted log10 metric).

Inputs (released):
  results/study2/d120_dev_baselines/results_merged.jsonl
  results/study2/d120_dev_portfolio_ud/results_merged.jsonl
  results/study2/h120_holdout/results_merged.jsonl

Output: results/analysis/regime_diagnostics.csv
  one row per (split, pairing, regime): n finite pairs, mean delta,
  where delta = log10_gap(left) - log10_gap(right); negative favors the
  left-named method.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "results" / "study2"
OUT = ROOT / "results" / "analysis" / "regime_diagnostics.csv"

PAIRINGS = [("raise_cut", "default"), ("raise_cut", "scip_dynamic")]


def log_gap(rec):
    p, d = rec.get("primalbound"), rec.get("dualbound")
    if p is None or d is None:
        return None
    p, d = float(p), float(d)
    if p == 0:
        return None
    gap = (p - d) / abs(p)
    if not math.isfinite(gap) or gap <= 0:
        return None
    return math.log10(gap)


def load(split):
    by_inst = defaultdict(dict)
    for line in open(S2 / split / "results_merged.jsonl"):
        r = json.loads(line)
        by_inst[r["instance_id"]][r["mode"]] = r
    return by_inst


def main():
    rows = []
    for split, gap_split in (("d120_dev", "d120_dev_baselines"),
                             ("h120_holdout", "h120_holdout")):
        by_inst = load(gap_split)
        if split == "d120_dev":                      # probe lives in UD split
            probe = {i: m["raise_portfolio_ud"]
                     for i, m in load("d120_dev_portfolio_ud").items()
                     if m.get("raise_portfolio_ud", {}).get("portfolio_probe_regime")}
        else:
            probe = {i: m["raise_portfolio_ud"]
                     for i, m in by_inst.items()
                     if m.get("raise_portfolio_ud", {}).get("portfolio_probe_regime")}
        for left, right in PAIRINGS:
            agg = defaultdict(list)
            for iid, pr in probe.items():
                m = by_inst.get(iid, {})
                a, b = m.get(left), m.get(right)
                if a is None or b is None:
                    continue
                ga, gb = log_gap(a), log_gap(b)
                if ga is None or gb is None:
                    continue
                agg[pr["portfolio_probe_regime"]].append(ga - gb)
            for regime in sorted(agg):
                v = agg[regime]
                rows.append([split, f"{left}_vs_{right}", regime,
                             len(v), f"{sum(v) / len(v):+.4f}"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "pairing", "regime", "n_pairs", "mean_log_gap_delta"])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
