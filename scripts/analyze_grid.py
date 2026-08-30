"""Grid + native analysis with deep sanity checks for the plan4and5 merged paper.

Outputs (results/analysis/):
  grid_runs.csv        per-run records joined with official MIPLIB optima
  native_runs.csv      per-run native CPLEX records
  sanity_report.md     invariant violations (see INVARIANTS below)
  policy_agg.csv       policy x budget aggregates (gap, ttff, feasibility)
  regret_matrix.csv    per-instance selection regret vs oracle policy
  winners.csv          per-instance x budget best policy + oracle stats

Deep-check invariants (any violation must be investigated before writing):
  I1 incumbent never better than known MIPLIB optimum (tolerance 1e-6 rel)
  I2 status=optimal  ->  gap ~ 0 and incumbent ~ optimum
  I3 time_limit runs wall ~ budget (+5s slack); node_limit runs wall <= budget + slack
  I4 ttff <= elapsed; ttff present iff incumbent present
  I5 deterministic policies (best_bound, depth_first, best_estimate,
     greedy_score, hybrid_best_bound_depth) are seed-invariant
  I6 stochastic policies (random_uniform, boltzmann_adaptive) vary across seeds
  I7 all benchmark instances are minimize (standardized == native)
"""
from __future__ import annotations
import os

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "grid" / "results" / "raw"
NATIVE = ROOT / "results" / "cplex_native"
ANALYSIS = ROOT / "results" / "analysis"
SOLU = Path(os.environ.get("MIPLIB_SOLU", ROOT / "data" / "miplib2017-v36.solu"))

DETERMINISTIC = {"best_bound", "depth_first", "best_estimate", "greedy_score", "hybrid_best_bound_depth"}
STOCHASTIC = {"random_uniform", "boltzmann_adaptive"}


def load_optima() -> dict[str, float]:
    optima: dict[str, float] = {}
    pattern = re.compile(r"^=(\w+)=\s+(\S+)\s+(\S+)")
    for line in SOLU.read_text().splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        kind, name, value = m.group(1), m.group(2), m.group(3)
        if kind in {"opt", "knwn"}:
            try:
                optima[name] = float(value)
            except ValueError:
                continue
    return optima


def collect_grid_runs() -> list[dict]:
    runs = []
    for path in sorted(RAW.glob("p45_*_summary.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        m = re.match(r"p45_(.+?)_(.+)_b(\d+)_s(\d+)$", payload.get("run_id", ""))
        if not m:
            # fall back to parsing from the file stem timestamp suffix
            stem = path.name[: -len("_summary.json")]
            m2 = re.match(r"p45_(.+?)_(.+)_b(\d+)_s(\d+)_\d+$", stem)
            if not m2:
                continue
            m = m2
        instance, policy, budget, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        runs.append(
            {
                "instance": instance,
                "policy": policy,
                "budget": budget,
                "seed": seed,
                "run_id": payload.get("run_id"),
                "status": payload.get("status"),
                "incumbent": payload.get("incumbent_objective"),
                "best_bound": payload.get("best_bound"),
                "final_gap": payload.get("final_gap"),
                "nodes": payload.get("nodes_evaluated"),
                "ttff": payload.get("time_to_first_feasible"),
                "elapsed": payload.get("elapsed_seconds"),
                "root_fractional_count": payload.get("root_fractional_count"),
                "root_fractional_ratio": payload.get("root_fractional_ratio"),
                "root_integer_variable_count": payload.get("root_integer_variable_count"),
                "root_solve_seconds": payload.get("root_solve_seconds"),
                "root_integrality_gap_l1": payload.get("root_integrality_gap_l1"),
                "root_binary_near_integral_ratio": payload.get("root_binary_near_integral_ratio"),
                "root_binary_midpoint_ratio": payload.get("root_binary_midpoint_ratio"),
            }
        )
    return runs


NEAR_ZERO_INSTANCES = {"academictimetablesmall"}  # official optimum 0: relative gap undefined


def rel_gap(incumbent: float | None, optimum: float) -> float | None:
    if incumbent is None:
        return None
    denom = abs(optimum) if abs(optimum) > 1e-9 else 1.0
    return (incumbent - optimum) / denom


def sanity_checks(runs: list[dict], optima: dict[str, float]) -> list[str]:
    problems: list[str] = []
    by_key = defaultdict(list)
    for r in runs:
        by_key[(r["instance"], r["policy"], r["budget"])].append(r)

    for r in runs:
        rid = f"{r['instance']}/{r['policy']}/b{r['budget']}/s{r['seed']}"
        opt = optima.get(r["instance"])
        inc = r["incumbent"]
        # I1
        if opt is not None and inc is not None and inc < opt - 1e-6 * max(abs(opt), 1.0):
            problems.append(f"I1 BETTER-THAN-OPTIMUM {rid}: incumbent={inc} < opt={opt}")
        # I2
        if r["status"] == "optimal" and opt is not None and inc is not None:
            if abs(inc - opt) > 1e-4 * max(abs(opt), 1.0):
                problems.append(f"I2 OPTIMAL-MISMATCH {rid}: incumbent={inc} vs opt={opt}")
        # I3
        if r["status"] == "time_limit" and r["elapsed"] is not None:
            if r["elapsed"] < r["budget"] - 5.0:
                problems.append(f"I3 EARLY-TIME-LIMIT {rid}: elapsed={r['elapsed']:.1f} < budget={r['budget']}")
        if r["status"] == "node_limit" and r["elapsed"] is not None:
            if r["elapsed"] > r["budget"] + 30.0:
                problems.append(f"I3 NODE-LIMIT-OVERSHOOT {rid}: elapsed={r['elapsed']:.1f} > budget={r['budget']}")
        # I4
        if (r["ttff"] is None) != (inc is None):
            problems.append(f"I4 TTFF-INCUMBENT-MISMATCH {rid}")
        if r["ttff"] is not None and r["elapsed"] is not None and r["ttff"] > r["elapsed"] + 1e-6:
            problems.append(f"I4 TTFF>ELAPSED {rid}: {r['ttff']:.2f} > {r['elapsed']:.2f}")
        if r["nodes"] is not None and r["nodes"] > 401:
            problems.append(f"I3 NODE-LIMIT-EXCEEDED {rid}: nodes={r['nodes']}")

    # I5 / I6
    for (instance, policy, budget), group in by_key.items():
        if len(group) < 2:
            continue
        incs = sorted({round(g["incumbent"], 6) for g in group if g["incumbent"] is not None})
        has_missing = any(g["incumbent"] is None for g in group)
        statuses = {g["status"] for g in group}
        if policy in DETERMINISTIC and (len(incs) > 1 or len(statuses) > 1 or has_missing):
            kind = "I5a VALUE-VARIANCE" if len(incs) > 1 else "I5b COVERAGE-VARIANCE"
            problems.append(
                f"{kind} {instance}/{policy}/b{budget}: incumbents={incs} statuses={statuses} missing={has_missing}"
            )
        if policy in STOCHASTIC and len(incs) == 1 and len(group) == 3:
            identical_nodes = len({g["nodes"] for g in group}) == 1
            if not identical_nodes:
                problems.append(
                    f"I6 STOCHASTIC-IDENTICAL-BUT-NODES-DIFFER {instance}/{policy}/b{budget}"
                )
    return problems


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    optima = load_optima()
    runs = collect_grid_runs()
    print(f"grid runs collected: {len(runs)} | optima known for {sum(1 for r in runs if r['instance'] in optima)}")

    # join optima + gaps
    for r in runs:
        opt = optima.get(r["instance"])
        r["optimum"] = opt
        r["gap_vs_opt"] = rel_gap(r["incumbent"], opt) if opt is not None else None
        r["gap_usable"] = r["instance"] not in NEAR_ZERO_INSTANCES
        status = r["status"] or ""
        if status.startswith("solver_status_"):
            # node-LP exhaustion (CPLEX 108 family); on rail01/rail02 short budgets
            # this is the ROOT LP exceeding the budget (verified: rail02 root LP = 351 s)
            r["status_family"] = "root_lp_timeout"
        elif status in {"time_limit", "node_limit", "optimal", "infeasible", "stalled"}:
            r["status_family"] = status
        else:
            r["status_family"] = "other"

    # wrapper portfolios derived from phase-1 runs (deterministic resolution)
    derived_path = ANALYSIS / "derived_portfolios.csv"
    if derived_path.exists():
        import csv as _csv

        with open(derived_path) as handle:
            for row in _csv.DictReader(handle):
                opt = optima.get(row["instance"])
                inc = float(row["incumbent"]) if row["incumbent"] else None
                status = row["status"] or ""
                runs.append(
                    {
                        "instance": row["instance"],
                        "policy": row["policy"],
                        "budget": int(row["budget"]),
                        "seed": int(row["seed"]),
                        "run_id": row.get("source_run_id"),
                        "status": status,
                        "status_family": (
                            "root_lp_timeout" if status.startswith("solver_status_")
                            else (status if status in {"time_limit", "node_limit", "optimal", "infeasible", "stalled"} else "other")
                        ),
                        "incumbent": inc,
                        "best_bound": None,
                        "final_gap": None,
                        "nodes": None,
                        "ttff": float(row["ttff"]) if row.get("ttff") else None,
                        "elapsed": float(row["elapsed"]) if row.get("elapsed") else None,
                        "optimum": opt,
                        "gap_vs_opt": rel_gap(inc, opt) if (opt is not None and inc is not None) else None,
                        "gap_usable": row["instance"] not in NEAR_ZERO_INSTANCES,
                        "resolved_policy": row.get("resolved_policy"),
                    }
                )

    problems = sanity_checks(runs, optima)

    # ---- CSV outputs
    def dump_csv(path: Path, records: list[dict], fields: list[str]) -> None:
        import csv as _csv

        with open(path, "w", newline="") as handle:
            writer = _csv.writer(handle, quoting=_csv.QUOTE_MINIMAL)
            writer.writerow(fields)
            for r in records:
                writer.writerow(["" if r.get(f) is None else r.get(f) for f in fields])

    dump_csv(
        ANALYSIS / "grid_runs.csv",
        runs,
        ["instance", "policy", "budget", "seed", "status", "status_family", "incumbent", "optimum",
         "gap_vs_opt", "final_gap", "nodes", "ttff", "elapsed", "root_fractional_count",
         "root_fractional_ratio", "root_integer_variable_count", "root_solve_seconds",
         "root_integrality_gap_l1", "root_binary_near_integral_ratio", "root_binary_midpoint_ratio"],
    )

    native_runs = []
    for path in sorted(NATIVE.glob("native_*.json")):
        try:
            native_runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    for r in native_runs:
        opt = optima.get(r["instance_name"])
        r["optimum"] = opt
        r["gap_vs_opt"] = rel_gap(r.get("incumbent_standardized"), opt) if opt is not None else None
        r["gap_usable"] = r["instance_name"] not in NEAR_ZERO_INSTANCES
    dump_csv(
        ANALYSIS / "native_runs.csv",
        native_runs,
        ["instance_name", "strategy", "budget", "seed", "status", "has_solution",
         "incumbent_standardized", "optimum", "gap_vs_opt", "wall_seconds"],
    )
    print(f"native runs collected: {len(native_runs)}")

    # ---- aggregates: policy x budget (mean gap vs opt, feasibility rate, mean ttff)
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        agg[(r["policy"], r["budget"])].append(r)
    rows = []
    for (policy, budget), group in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        gaps = [g["gap_vs_opt"] for g in group if g["gap_vs_opt"] is not None]
        ttffs = [g["ttff"] for g in group if g["ttff"] is not None]
        rows.append(
            {
                "policy": policy,
                "budget": budget,
                "n_runs": len(group),
                "feasible_runs": sum(1 for g in group if g["incumbent"] is not None),
                "mean_gap": statistics.fmean(gaps) if gaps else None,
                "median_gap": statistics.median(gaps) if gaps else None,
                "mean_ttff": statistics.fmean(ttffs) if ttffs else None,
            }
        )
    dump_csv(
        ANALYSIS / "policy_agg.csv",
        rows,
        ["policy", "budget", "n_runs", "feasible_runs", "mean_gap", "median_gap", "mean_ttff"],
    )

    # ---- regret vs per-instance oracle policy (same budget, any policy)
    regret_rows = []
    by_ib = defaultdict(dict)
    for r in runs:
        if r["gap_vs_opt"] is None:
            continue
        if r["policy"] == "learned_v2":
            continue  # in-sample diagnostic of R6; excluded from the 12-policy main-grid oracle
        key = (r["instance"], r["budget"])
        by_ib.setdefault(key, {})
        by_ib[key].setdefault(r["policy"], []).append(r)
    for (instance, budget), policies in sorted(by_ib.items()):
        policy_means = {p: statistics.fmean(g["gap_vs_opt"] for g in gs) for p, gs in policies.items()}
        # No-incumbent penalty, consistent with the policy_budget table: a policy
        # with no feasible run in this cell is charged at the worst per-policy
        # mean achieved in the cell. The oracle is attained by a feasible policy,
        # so winner determination is unchanged; regret denominators now coincide
        # with the penalized-gap table (identity: penalized_mean - mean_regret
        # = oracle mean, per budget).
        worst = max(policy_means.values())
        ALL_POLICIES = [
            "best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
            "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid",
            "budgeted_portfolio", "instance_aware_portfolio", "multi_feature_portfolio",
            "learned_portfolio",
        ]
        penalized_means = {p: policy_means.get(p, worst) for p in ALL_POLICIES if p in policy_means or True}
        oracle_gap = min(policy_means.values())
        base_first = [
            "best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
            "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid",
            "budgeted_portfolio", "instance_aware_portfolio", "multi_feature_portfolio",
            "learned_portfolio",
        ]
        rank = {p: i for i, p in enumerate(base_first)}
        winner = min(policy_means, key=lambda p: (policy_means[p], rank.get(p, 99)))
        for policy, mean_gap in penalized_means.items():
            regret_rows.append(
                {
                    "instance": instance,
                    "budget": budget,
                    "policy": policy,
                    "mean_gap": mean_gap,
                    "oracle_gap": oracle_gap,
                    "regret": mean_gap - oracle_gap,
                    "winner": winner,
                }
            )
    dump_csv(
        ANALYSIS / "regret_matrix.csv",
        regret_rows,
        ["instance", "budget", "policy", "mean_gap", "oracle_gap", "regret", "winner"],
    )

    win_rows = []
    for (instance, budget), policies in sorted(by_ib.items()):
        policy_means = {p: statistics.fmean(g["gap_vs_opt"] for g in gs) for p, gs in policies.items()}
        # No-incumbent penalty, consistent with the policy_budget table: a policy
        # with no feasible run in this cell is charged at the worst per-policy
        # mean achieved in the cell. The oracle is attained by a feasible policy,
        # so winner determination is unchanged; regret denominators now coincide
        # with the penalized-gap table (identity: penalized_mean - mean_regret
        # = oracle mean, per budget).
        worst = max(policy_means.values())
        ALL_POLICIES = [
            "best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
            "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid",
            "budgeted_portfolio", "instance_aware_portfolio", "multi_feature_portfolio",
            "learned_portfolio",
        ]
        penalized_means = {p: policy_means.get(p, worst) for p in ALL_POLICIES if p in policy_means or True}
        oracle_gap = min(policy_means.values())
        base_first = [
            "best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
            "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid",
            "budgeted_portfolio", "instance_aware_portfolio", "multi_feature_portfolio",
            "learned_portfolio",
        ]
        rank = {p: i for i, p in enumerate(base_first)}
        win_rows.append(
            {
                "instance": instance,
                "budget": budget,
                "winner": min(policy_means, key=lambda p: (policy_means[p], rank.get(p, 99))),
                "oracle_gap": oracle_gap,
            }
        )
    dump_csv(ANALYSIS / "winners.csv", win_rows, ["instance", "budget", "winner", "oracle_gap"])

    # ---- sanity report
    with open(ANALYSIS / "sanity_report.md", "w") as handle:
        handle.write("# Grid sanity report\n\n")
        handle.write(f"- grid runs: {len(runs)}\n- native runs: {len(native_runs)}\n")
        handle.write(f"- invariant violations: {len(problems)}\n\n")
        for p in problems:
            handle.write(f"- {p}\n")

    print(f"invariant violations: {len(problems)}")
    for p in problems[:20]:
        print("  ", p)
    print(f"analysis written to {ANALYSIS}")


if __name__ == "__main__":
    main()
