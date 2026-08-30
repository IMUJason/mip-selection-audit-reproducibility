#!/usr/bin/env python3
"""Verify EVERY row of paper/tables/study2.tex against the released merged logs.

Chain: merged result JSONL -> this script -> study2.tex (all 10 rows + the
restricted-metric sensitivity note). Run from the plan4and5 directory:
    python scripts/verify_study2_external.py

Data layout (relative to the package root, identical in the released
repository and the paper workspace):
  results/study2/h140_benchmark/       benchmark hold-out (140 inst, 5 modes)
  results/study2/c140_confirmatory/    external confirmatory (140 inst)
  results/study2/d120_dev_baselines/   external development split, baselines
  results/study2/d120_dev_portfolio_ud/  development split, UD route
                                       (d120 modes are split across the two)
  results/study2/h120_holdout/         untouched external hold-out (120 inst)

Metrics
-------
paired (this table's metric): gap_eval = run's own `gap` field, dropped when
  >= 1e19; delta = log1p(gap_left) - log1p(gap_right) on finite pairs; W/L/T on
  raw gap delta at 1e-9; two-sided exact sign test.
restricted (sensitivity): log10((primal - dual)/|primal|) from
  primalbound/dualbound, REQUIRES primal > 0 AND dual > 0 (drops instances with
  no proven positive dual bound); W/L/T on log-delta at 1e-6.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT / "results" / "study2"
H140 = S2 / "h140_benchmark" / "results_merged.jsonl"
C140 = S2 / "c140_confirmatory" / "results_merged.jsonl"
D120A = S2 / "d120_dev_baselines" / "results_merged.jsonl"
D120B = S2 / "d120_dev_portfolio_ud" / "results_merged.jsonl"
H120 = S2 / "h120_holdout" / "results_merged.jsonl"


def sign_p(w, l):
    n = w + l
    if n == 0:
        return None
    tail = min(w, l)
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n)


def load(path, into=None):
    d = into if into is not None else {}
    for line in open(path):
        r = json.loads(line)
        d.setdefault(r["mode"], {})[r["instance_id"]] = r
    return d


def paired(d, a, b):
    """left=a, right=b; delta oriented toward a (negative favors a)."""
    A, B = d[a], d[b]
    insts = sorted(set(A) & set(B))
    fin = [(A[i], B[i]) for i in insts
           if A[i].get("gap") is not None and B[i].get("gap") is not None
           and A[i]["gap"] < 1e19 and B[i]["gap"] < 1e19]
    dl = [x["gap"] - y["gap"] for x, y in fin]
    ld = [math.log1p(x["gap"]) - math.log1p(y["gap"]) for x, y in fin]
    w = sum(1 for v in dl if v < -1e-9)
    l = sum(1 for v in dl if v > 1e-9)
    return dict(pairs=len(insts), finite=len(fin), w=w, l=l, t=len(fin) - w - l,
                mld=sum(ld) / len(ld), p=sign_p(w, l))


def restricted(d, base):
    def lg(it):
        p, q = it.get("primalbound", float("inf")), it.get("dualbound", 0)
        if p is None or q is None or p <= 0 or q <= 0:
            return None
        if p <= q:
            return 0.0
        g = (p - q) / abs(p)
        return 0.0 if g <= 0 else math.log10(g)
    a, b = d["raise_cut"], d[base]
    retained = {i for i in set(a) & set(b) if lg(a[i]) is not None and lg(b[i]) is not None}
    dl = [lg(a[i]) - lg(b[i]) for i in sorted(retained)]
    w = sum(1 for v in dl if v < -1e-6)
    l = sum(1 for v in dl if v > 1e-6)
    return dict(retained=len(retained), w=w, l=l, t=len(dl) - w - l,
                mld=sum(dl) / len(dl), p=sign_p(w, l))


def close(x, y, tol=5e-4):
    return abs(x - y) <= tol


def main():
    h140 = load(H140)
    c140 = load(C140)
    d120 = load(D120B, into=load(D120A))
    h120 = load(H120)

    # (label, got, expected) -- expected values are exactly what study2.tex prints
    checks = [
        # --- the ten table rows (finite-pair metric) ---
        ("h140 RC vs adaptive ", paired(h140, "raise_cut", "plan5_adaptive"),
         dict(w=27, l=23, t=19, mld=-0.0359)),
        ("h140 RC vs efficacy ", paired(h140, "raise_cut", "efficacy"),
         dict(w=9, l=8, t=53, mld=-0.0231)),
        ("h140 RC vs default  ", paired(h140, "raise_cut", "default"),
         dict(w=28, l=27, t=14, mld=+0.0317)),
        ("c140 Portfolio vs RC", paired(c140, "raise_portfolio", "raise_cut"),
         dict(w=1, l=2, t=73, mld=+0.0198)),
        ("c140 Port-RC vs RC  ", paired(c140, "raise_portfolio_rc", "raise_cut"),
         dict(w=1, l=5, t=71, mld=+0.0241)),
        ("d120 UD vs RC       ", paired(d120, "raise_portfolio_ud", "raise_cut"),
         dict(w=3, l=0, t=63, mld=-0.0251)),
        ("h120 UD vs RC       ", paired(h120, "raise_portfolio_ud", "raise_cut"),
         dict(w=4, l=2, t=54, mld=+0.0002)),
        ("c140 RC vs default  ", paired(c140, "raise_cut", "default"),
         dict(w=29, l=24, t=21, mld=-0.0114)),
        ("h120 RC vs default  ", paired(h120, "raise_cut", "default"),
         dict(w=16, l=25, t=19, mld=+0.0018)),
        ("h120 RC vs dynamic  ", paired(h120, "raise_cut", "scip_dynamic"),
         dict(w=15, l=23, t=20, mld=+0.1043)),
        # --- restricted-metric sensitivity (caption note) ---
        ("restr. c140 vs default", restricted(c140, "default"),
         dict(retained=75, w=26, l=23, t=26, mld=-0.0496)),
        ("restr. h120 vs default", restricted(h120, "default"),
         dict(retained=64, w=12, l=18, t=34, mld=-0.0383)),
        ("restr. h120 vs dynamic", restricted(h120, "scip_dynamic"),
         dict(retained=64, w=13, l=15, t=36, mld=-0.0429)),
    ]
    ok = True
    for name, got, exp in checks:
        line_ok = all(close(got[k], v) if k == "mld" else got[k] == v
                      for k, v in exp.items())
        ok &= line_ok
        print(f"{'OK ' if line_ok else 'FAIL'} {name}: "
              f"W={got['w']} L={got['l']} T={got['t']} mld={got['mld']:+.4f} p={got['p']:.2f}")
    print("ALL CHECKS PASSED" if ok else "MISMATCH FOUND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
