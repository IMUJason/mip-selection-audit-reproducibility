#!/usr/bin/env python
"""Paired statistics complementing the descriptive comparisons (Section 5).

Computes, from the released artifacts only:
  S1-repl   Study-1 replication cohort, 300 s face: per-instance bound-driven
            vs exploration class-mean deltas over both-feasible cells;
            Wilcoxon signed-rank on the paired cell deltas.
  S1-orig   Study-1 original benchmark: per-cell class-mean deltas
            (both-feasible cells); Wilcoxon.
  LV2-hold  Learned-v2 vs best-bound (the SBS) per cell on the replication
            hold-out (both-feasible); Wilcoxon.
  Study-2   raise_cut vs default / scip_dynamic paired per-instance log-gap
            deltas on the c140 and h120 splits, finite-pair and restricted
            metrics; Wilcoxon plus paired bootstrap 95% CIs (10k resamples).

Output: results/analysis/paired_stats.csv (+ stdout summary).
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_study2_external import C140, H120, load, paired, restricted  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "paired_stats.csv"

BD = ["best_bound", "depth_first", "hybrid_best_bound_depth"]
EX = ["best_estimate", "boltzmann_adaptive", "random_uniform"]


def pct_gap(incumbent, optimum):
    return 100.0 * (incumbent - optimum) / abs(optimum)


def s1_ext_cells():
    ledger = ROOT / "results" / "grid_ext" / "ext_ledger.jsonl"
    manifest = json.loads((ROOT / "data" / "extension_manifest.json").read_text())
    items = manifest["instances"] if isinstance(manifest, dict) else manifest
    optima = {i["instance_name"]: i["optimum"] for i in items}
    pat = re.compile(r"^p45x_(.+?)_([a-z_]+?)_(b\d+)_(s\d+)$")
    gaps = defaultdict(lambda: defaultdict(list))
    for line in ledger.read_text().splitlines():
        r = json.loads(line)
        m = pat.match(r["run_id"])
        if not m or m.group(3) != "b300" or r.get("incumbent") is None:
            continue
        inst, pol = m.group(1), m.group(2)
        if inst in optima:
            gaps[inst][pol].append(pct_gap(r["incumbent"], optima[inst]))
    # learned-v2 replication runs live as raw summaries, not ledger rows
    raw = ROOT / "results" / "grid_ext" / "results" / "raw"
    for f in raw.glob("p45x_*_learned_v2_b300_s*_summary.json"):
        d = json.loads(f.read_text())
        m = re.match(r"^p45x_(.+?)_learned_v2_b300_s\d+(?:_\d+)?$", d["run_id"])
        inc, opt = (d.get("incumbent_objective"),
                    optima.get(m.group(1)) if m else None)
        if m and inc is not None and opt is not None:
            gaps[m.group(1)]["learned_v2"].append(pct_gap(inc, opt))
    cells, lv2 = [], []
    for inst, by_pol in gaps.items():
        bd = [g for p in BD for g in by_pol.get(p, [])]
        ex = [g for p in EX for g in by_pol.get(p, [])]
        if bd and ex:
            delta = statistics.fmean(bd) - statistics.fmean(ex)
            cells.append((inst, delta))
            lv = by_pol.get("learned_v2", [])
            bb = by_pol.get("best_bound", [])
            if lv and bb:
                lv2.append((inst, statistics.fmean(lv) - statistics.fmean(bb)))
    return cells, lv2


def s1_orig_cells():
    rows = list(csv.DictReader(open(ROOT / "results" / "analysis" / "grid_runs.csv")))
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("policy") in BD + EX and r.get("gap_vs_opt") not in ("", None):
            cells[(r["instance"], int(float(r["budget"])))][r["policy"]].append(
                100 * float(r["gap_vs_opt"]))
    out = []
    for (inst, b), by_pol in cells.items():
        bd = [g for p in BD for g in by_pol.get(p, [])]
        ex = [g for p in EX for g in by_pol.get(p, [])]
        if bd and ex:
            out.append(((inst, b), statistics.fmean(bd) - statistics.fmean(ex)))
    return out


def log_gap(p, d):
    p, d = float(p), float(d)
    if p == 0:
        return None
    g = (p - d) / abs(p)
    return math.log10(g) if (math.isfinite(g) and g > 0) else None


def study2_pairs(split_path, left, right):
    """Official metric implementations from verify_study2_external.py."""
    d = load(split_path)
    fin = paired(d, left, right)
    res = restricted(d, right)
    fin_deltas, res_deltas = [], []
    # rebuild raw delta lists for Wilcoxon/bootstrap
    A, B = d[left], d[right]
    for i in sorted(set(A) & set(B)):
        a, b = A[i], B[i]
        if all(a.get("gap") is not None and b.get("gap") is not None
               and a["gap"] < 1e19 and b["gap"] < 1e19 for _ in (0,)):
            fin_deltas.append(math.log1p(a["gap"]) - math.log1p(b["gap"]))

    def lg(it):
        p, q = it.get("primalbound", float("inf")), it.get("dualbound", 0)
        if p is None or q is None or p <= 0 or q <= 0:
            return None
        if p <= q:
            return 0.0
        g = (p - q) / abs(p)
        return 0.0 if g <= 0 else math.log10(g)

    for i in sorted(set(A) & set(B)):
        ga, gb = lg(A[i]), lg(B[i])
        if ga is not None and gb is not None:
            res_deltas.append(ga - gb)
    return fin_deltas, res_deltas


def boot_ci(deltas, n=10_000, seed=0):
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(statistics.fmean(rng.choices(deltas, k=len(deltas))))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def wilco(deltas):
    if len(set(deltas)) == 1:
        return float("nan")
    return wilcoxon(deltas).pvalue


def main():
    rows = []
    cells, lv2 = s1_ext_cells()
    rows.append(["S1-repl", "BD vs EX class means, 22 cells", len(cells),
                 f"{statistics.fmean(d for _, d in cells):+.2f}",
                 f"{wilco([d for _, d in cells]):.3f}", ""])
    rows.append(["LV2-hold", "learned-v2 vs best-bound", len(lv2),
                 f"{statistics.fmean(d for _, d in lv2):+.2f}",
                 f"{wilco([d for _, d in lv2]):.3f}", ""])
    orig = s1_orig_cells()
    rows.append(["S1-orig", "BD vs EX class means", len(orig),
                 f"{statistics.fmean(d for _, d in orig):+.2f}",
                 f"{wilco([d for _, d in orig]):.3f}", ""])
    for split_path, left, right in [(C140, "raise_cut", "default"),
                                    (H120, "raise_cut", "default"),
                                    (H120, "raise_cut", "scip_dynamic")]:
        for metric, deltas in zip(
                ("finite", "restricted"), study2_pairs(split_path, left, right)):
            lo, hi = boot_ci(deltas)
            rows.append([f"S2-{str(split_path)[-20:-13]}",
                         f"{left} vs {right} ({metric})",
                         len(deltas),
                         f"{statistics.fmean(deltas):+.4f}",
                         f"{wilco(deltas):.3f}",
                         f"[{lo:+.4f}, {hi:+.4f}]"])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "detail", "n", "mean_delta",
                    "wilcoxon_p", "boot95_ci"])
        w.writerows(rows)
    for r in rows:
        print(" | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
