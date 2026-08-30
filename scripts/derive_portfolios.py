"""Derive the three wrapper-portfolio policies from phase-1 summaries.

Rationale (audit note): budgeted / instance_aware / multi_feature portfolios
resolve DETERMINISTICALLY to one of the fixed phase-1 policies before any node
selection happens - budgeted via the budget threshold, the other two via
budget plus root features that phase-1 summaries already log. A wrapper run
with the same seed therefore reproduces the resolved policy's run exactly, so
re-running them would burn ~6 CPU-hours producing bit-identical results.

Resolution rules replicated from BranchAndBoundConfig defaults:
  budgeted_portfolio:      budget <= 180 -> boltzmann_adaptive, else best_estimate
  instance_aware_portfolio: budget > 180 -> best_estimate;
        else root_fractional_count >= 5000 -> safeguarded_hybrid, else boltzmann
  multi_feature_portfolio:  budget > 180 -> best_estimate;
        else votes {frac_count>=5000, frac_ratio>=0.08, int_vars>=100000,
        root_solve_seconds>=1.0}; >=2 votes -> safeguarded_hybrid else boltzmann

Output: results/analysis/derived_portfolios.csv with one row per
(instance, budget, seed, wrapper) joined to its source phase-1 run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "grid" / "results" / "raw"
OUT = ROOT / "results" / "analysis" / "derived_portfolios.csv"

SHORT_SMALL = "boltzmann_adaptive"
SHORT_LARGE = "safeguarded_hybrid"
LONG = "best_estimate"


def collect() -> dict[tuple, dict]:
    runs = {}
    for path in RAW.glob("p45_*_summary.json"):
        d = json.loads(path.read_text())
        m = re.match(r"p45_(.+?)_(.+)_b(\d+)_s(\d+)_\d+$", d.get("run_id", ""))
        if not m:
            continue
        instance, policy, budget, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if policy not in {"boltzmann_adaptive", "best_estimate", "safeguarded_hybrid"}:
            continue
        key = (instance, policy, budget, seed)
        if key not in runs or d.get("elapsed_seconds", 0) >= runs[key].get("elapsed_seconds", 0):
            runs[key] = d
    return runs


def main() -> None:
    runs = collect()
    print(f"source runs: {len(runs)}")
    instances = sorted({k[0] for k in runs})
    rows = []
    for instance in instances:
        for budget in [10, 60, 300, 900]:
            for seed in [0, 1, 2]:
                # pick the source summary whose root features the wrapper would see;
                # prefer boltzmann (the short-budget default anchor) when present
                anchor = None
                for pol in ["boltzmann_adaptive", "best_estimate", "safeguarded_hybrid"]:
                    anchor = runs.get((instance, pol, budget, seed))
                    if anchor is not None:
                        break
                if anchor is None:
                    continue
                frac_count = anchor.get("root_fractional_count") or 0
                frac_ratio = anchor.get("root_fractional_ratio") or 0.0
                int_vars = anchor.get("root_integer_variable_count") or 0
                solve_s = anchor.get("root_solve_seconds") or 0.0

                wrapper_map = {}
                wrapper_map["budgeted_portfolio"] = (
                    (SHORT_SMALL, "budget<=180") if budget <= 180 else (LONG, "budget>180")
                )
                if budget > 180:
                    wrapper_map["instance_aware_portfolio"] = (LONG, "budget>180")
                    wrapper_map["multi_feature_portfolio"] = (LONG, "budget>180")
                else:
                    wrapper_map["instance_aware_portfolio"] = (
                        (SHORT_LARGE, f"frac_count={frac_count}>=5000")
                        if frac_count >= 5000
                        else (SHORT_SMALL, f"frac_count={frac_count}<5000")
                    )
                    votes = sum(
                        [
                            frac_count >= 5000,
                            frac_ratio >= 0.08,
                            int_vars >= 100000,
                            solve_s >= 1.0,
                        ]
                    )
                    wrapper_map["multi_feature_portfolio"] = (
                        (SHORT_LARGE, f"votes={votes}/4") if votes >= 2 else (SHORT_SMALL, f"votes={votes}/4")
                    )

                for wrapper, (resolved, reason) in wrapper_map.items():
                    source = runs.get((instance, resolved, budget, seed))
                    if source is None:
                        continue
                    rows.append(
                        {
                            "instance": instance,
                            "policy": wrapper,
                            "budget": budget,
                            "seed": seed,
                            "resolved_policy": resolved,
                            "resolution_reason": reason,
                            "source_run_id": source["run_id"],
                            "status": source["status"],
                            "incumbent": source["incumbent_objective"],
                            "ttff": source.get("time_to_first_feasible"),
                            "elapsed": source.get("elapsed_seconds"),
                            "gap_vs_opt": None,  # joined by analyze-side optima lookup
                        }
                    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    reasons = {}
    for r in rows:
        reasons.setdefault(r["policy"], {}).setdefault(r["resolution_reason"].split("=")[0].split("/")[0], 0)
        key = r["resolution_reason"].split("=")[0].split("/")[0]
        reasons[r["policy"]][key] += 1
    print(f"derived rows: {len(rows)} -> {OUT}")
    for w, d in reasons.items():
        print(f"  {w}: {d}")


if __name__ == "__main__":
    main()
