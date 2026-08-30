"""Anchor-quality mechanism analysis from existing run traces.

Mechanistic claim (R2 of the paper): the incumbent a time-limited primal
heuristic returns is proportional to the bound quality of the node it is
anchored at. Every trace already records, per incumbent-improving event, the
selected node's LP bound and the resulting incumbent - so the mechanism is
measurable directly, across all policies/instances/budgets, with zero new
solver runs.

Outputs (results/analysis/):
  anchor_events.csv    one row per incumbent-improving event
  anchor_summary.csv   per policy: mean anchor gap, mean incumbent gap, corr
  anchor_split.csv     per instance x budget: bound-driven vs exploration split
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "results" / "grid" / "logs" / "runs"
ANALYSIS = ROOT / "results" / "analysis"
SOLU = Path("/Users/cjx-ms/Documents/我的坚果云/科研/论文/MIPLIB 2017 Dataset/miplib2017-v36.solu")

BOUND_DRIVEN = {"best_bound", "depth_first", "hybrid_best_bound_depth"}
NEAR_ZERO = {"academictimetablesmall"}


def load_optima() -> dict[str, float]:
    optima = {}
    pat = re.compile(r"^=(\w+)=\s+(\S+)\s+(\S+)")
    for line in SOLU.read_text().splitlines():
        m = pat.match(line.strip())
        if m and m.group(1) in {"opt", "knwn"}:
            try:
                optima[m.group(2)] = float(m.group(3))
            except ValueError:
                pass
    return optima


def gap(value: float, opt: float) -> float:
    return (value - opt) / (abs(opt) if abs(opt) > 1e-9 else 1.0)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else None


def main() -> None:
    optima = load_optima()
    events = []
    for path in sorted(TRACES.glob("p45_*_trace.jsonl")):
        m = re.match(r"p45_(.+?)_(.+)_b(\d+)_s(\d+)_\d+_trace", path.name)
        if not m:
            continue
        instance, policy, budget, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        opt = optima.get(instance)
        if opt is None or instance in NEAR_ZERO or abs(opt) < 1.0:
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            inc_b, inc_a = e.get("incumbent_before"), e.get("incumbent_after")
            lp = e.get("node_lp_objective")
            if inc_a is None or lp is None:
                continue
            if inc_b is not None and inc_a >= inc_b - 1e-9:
                continue  # only improving events
            events.append(
                {
                    "instance": instance,
                    "policy": policy,
                    "budget": budget,
                    "seed": seed,
                    "depth": e.get("depth"),
                    "backend": e.get("heuristic_backend"),
                    "anchor_gap": gap(lp, opt),
                    "incumbent_gap": gap(inc_a, opt),
                }
            )

    import csv

    with open(ANALYSIS / "anchor_events.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(events[0].keys()))
        w.writeheader()
        w.writerows(events)
    print(f"anchor events: {len(events)}")

    by_policy = defaultdict(list)
    for e in events:
        by_policy[e["policy"]].append(e)
    rows = []
    for policy, evs in sorted(by_policy.items()):
        ag = [e["anchor_gap"] for e in evs]
        ig = [e["incumbent_gap"] for e in evs]
        rows.append(
            {
                "policy": policy,
                "n_events": len(evs),
                "mean_anchor_gap": statistics.fmean(ag),
                "mean_incumbent_gap": statistics.fmean(ig),
                "median_anchor_gap": statistics.median(ag),
                "spearman": spearman(ag, ig),
            }
        )
    with open(ANALYSIS / "anchor_summary.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        sp = f"{r['spearman']:.3f}" if r["spearman"] is not None else "--"
        print(f"  {r['policy']:>24}: n={r['n_events']:>5} anchor={100*r['mean_anchor_gap']:>8.2f}% inc={100*r['mean_incumbent_gap']:>8.2f}% rho={sp}")

    # per-instance budget-300 bound-driven vs exploration split (replication table)
    inst_cell = defaultdict(lambda: defaultdict(list))
    for e in events:
        inst_cell[(e["instance"], e["budget"])][e["policy"]].append(e["incumbent_gap"])
    split_rows = []
    for (inst, b), pols in sorted(inst_cell.items()):
        bd, ex = [], []
        for p, gaps in pols.items():
            mean = statistics.fmean(gaps)
            (bd if p in BOUND_DRIVEN else ex).append((p, mean))
        if not bd or not ex:
            continue
        bd_mean = statistics.fmean([g for _, g in bd])
        ex_mean = statistics.fmean([g for _, g in ex])
        split_rows.append(
            {
                "instance": inst,
                "budget": b,
                "bound_driven_mean_gap": bd_mean,
                "exploration_mean_gap": ex_mean,
                "ratio": ex_mean / bd_mean if bd_mean > 1e-12 else None,
                "n_events_bd": sum(1 for p, g in bd),
                "n_events_ex": sum(1 for p, g in ex),
            }
        )
    with open(ANALYSIS / "anchor_split.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(split_rows[0].keys()))
        w.writeheader()
        w.writerows(split_rows)
    print("\nbudget-300 splits (exploration/bound-driven incumbent gap ratio):")
    for r in split_rows:
        if r["budget"] == 300:
            ratio = f"{r['ratio']:.1f}x" if r["ratio"] else "--"
            print(f"  {r['instance']:>22}: bd={100*r['bound_driven_mean_gap']:>7.2f}% ex={100*r['exploration_mean_gap']:>7.2f}% ratio={ratio}")


if __name__ == "__main__":
    main()
