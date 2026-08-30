"""Phase-1 full grid driver for the plan4and5 merged paper (node-selection side).

Grid: 16 MIPLIB transport instances x 8 fixed policies x 4 budgets x 3 seeds,
under the CPLEX backend (env310). Common settings replicate the original
Plan 4 round-20 protocol (node_limit=400, shortlist=16, heuristic 0.08s/step,
local branching radius 8 / 48 binaries) so cross-paper comparisons stay honest.

Usage:
  /opt/anaconda3/envs/env310/bin/python scripts/run_grid.py --workers 6
  ... --dry-run     print the task table only
  ... --limit N     run only the first N tasks (pilot extension)
Resume: a task is skipped when its summary JSON already exists and parses.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine  # noqa: E402

DATA = ROOT / "data" / "instances"
OUTPUT_ROOT = ROOT / "results" / "grid"

POLICIES = [
    "best_bound",
    "depth_first",
    "hybrid_best_bound_depth",
    "random_uniform",
    "greedy_score",
    "best_estimate",
    "boltzmann_adaptive",
    "safeguarded_hybrid",
]
BUDGETS = [10.0, 60.0, 300.0, 900.0]
SEEDS = [0, 1, 2]

COMMON = dict(
    node_limit=400,
    batch_size=1,
    shortlist_size=16,
    heuristic_enabled=True,
    heuristic_time_limit_seconds=0.08,
    heuristic_every_n_selected=1,
    heuristic_repair_max_free_integer_vars=10,
    heuristic_local_branching_enabled=True,
    heuristic_local_branching_radius=8,
    heuristic_local_branching_max_binary_vars=48,
    heuristic_local_branching_max_fractional_vars=24,
    heuristic_local_branching_max_gap=0.20,
    threads=1,
    solver_backend="cplex",
    # Row-level rounding-violation features cost up to 815 s on gfd-schedulen
    # (458k constraints, pure-Python scan) and are audit-only: no fixed policy
    # consumes them. Keep the cheap variable-level features, drop the row scan.
    # Metrics are computed post-solve and are read-only, so search trajectories
    # are unaffected (verified: root LP and node sequence identical).
    root_lp_proxy_mode="variable_only",
)


def load_instances() -> list[dict]:
    manifest = json.loads((ROOT / "data" / "dataset_manifest_round1.json").read_text())
    return manifest["items"]


def task_list(instances: list[dict]) -> list[dict]:
    tasks = []
    for item in instances:
        for budget in BUDGETS:
            for seed in SEEDS:
                for policy in POLICIES:
                    run_id = f"p45_{item['instance_name']}_{policy}_b{int(budget)}_s{seed}"
                    tasks.append(
                        {
                            "run_id": run_id,
                            "instance_name": item["instance_name"],
                            "instance_file": item["filename"],
                            "policy": policy,
                            "budget": budget,
                            "seed": seed,
                        }
                    )
    return tasks


def run_task(task: dict) -> dict:
    run_id = task["run_id"]
    # engine summaries are named f"{run_tag}_{unix_ts}_summary.json"
    for existing in sorted((OUTPUT_ROOT / "results" / "raw").glob(f"{run_id}_*_summary.json")):
        try:
            payload = json.loads(existing.read_text())
            if str(payload.get("run_id", "")).startswith(run_id + "_"):
                return {"run_id": run_id, "status": "skipped_existing"}
        except (json.JSONDecodeError, OSError):
            continue

    config = BranchAndBoundConfig(
        instance_path=str(DATA / task["instance_file"]),
        policy=task["policy"],
        time_limit_seconds=task["budget"],
        random_seed=task["seed"],
        run_tag=run_id,
        output_root=str(OUTPUT_ROOT),
        **COMMON,
    )
    started = time.monotonic()
    try:
        summary = BranchAndBoundEngine(config).run()
        return {
            "run_id": run_id,
            "status": summary.status,
            "incumbent": summary.incumbent_objective,
            "gap": summary.final_gap,
            "nodes": summary.nodes_evaluated,
            "ttff": summary.time_to_first_feasible,
            "wall_seconds": round(time.monotonic() - started, 2),
        }
    except Exception:  # noqa: BLE001 - record and continue with the grid
        return {
            "run_id": run_id,
            "status": "failed",
            "error": traceback.format_exc(limit=8),
            "wall_seconds": round(time.monotonic() - started, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=float, default=None)
    args = parser.parse_args()

    instances = load_instances()
    global POLICIES, BUDGETS
    if args.policies:
        POLICIES = args.policies
    if args.budgets:
        BUDGETS = args.budgets

    tasks = task_list(instances)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.dry_run:
        for task in tasks:
            print(task["run_id"])
        print(f"total tasks: {len(tasks)}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = OUTPUT_ROOT / "grid_ledger.jsonl"
    print(f"tasks: {len(tasks)} | workers: {args.workers} | ledger: {ledger}", flush=True)

    done = 0
    with mp.Pool(args.workers) as pool, open(ledger, "a") as handle:
        for result in pool.imap_unordered(run_task, tasks):
            done += 1
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            flag = "OK " if result["status"] not in {"failed"} else "FAIL"
            if result["status"] != "skipped_existing":
                print(
                    f"[{done}/{len(tasks)}] {flag} {result['run_id']} "
                    f"status={result['status']} wall={result.get('wall_seconds')}s",
                    flush=True,
                )
            if result["status"] == "failed":
                print(result.get("error", ""), flush=True)
    print("GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
