#!/usr/bin/env python
"""Analyze the CPU-time vs wall-clock robustness subset (finding 3).

Reads results/cpu_time/cpu_time_ledger.jsonl and reports, per instance x
policy cell: the cross-seed spread of incumbents (max-min) and gap under
each time basis, plus the count of value-level seed-variance cells under
each basis. Wall twins give same-machine contrast.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "cpu_time" / "cpu_time_ledger.jsonl"


def main() -> None:
    cells = defaultdict(lambda: defaultdict(list))
    for line in open(LEDGER):
        r = json.loads(line)
        if r.get("status") in ("failed",):
            continue
        key = (r["instance"], r["policy"])
        cells[key][r["basis"]].append(r)

    print(f"{'instance':16s} {'policy':24s} {'wall spread':>12s} "
          f"{'cpu spread':>11s}  incumbents (wall | cpu)")
    var_wall = var_cpu = n = 0
    for (inst, pol), by_basis in sorted(cells.items()):
        w = by_basis.get("wall", [])
        c = by_basis.get("cpu", [])
        if not w or not c:
            continue
        n += 1
        w_inc = [x["incumbent"] for x in w if x["incumbent"] is not None]
        c_inc = [x["incumbent"] for x in c if x["incumbent"] is not None]
        w_spread = (max(w_inc) - min(w_inc)) if w_inc else None
        c_spread = (max(c_inc) - min(c_inc)) if c_inc else None
        if w_spread:
            var_wall += 1
        if c_spread:
            var_cpu += 1
        print(f"{inst:16s} {pol:24s} "
              f"{w_spread if w_spread is not None else '-':>12} "
              f"{c_spread if c_spread is not None else '-':>11}  "
              f"{sorted(w_inc)} | {sorted(c_inc)}")
    print(f"\ncells compared: {n}; value-level seed-variance cells: "
          f"wall={var_wall}, cpu={var_cpu}")


if __name__ == "__main__":
    main()
