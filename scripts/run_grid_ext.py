"""Extension grid: anchoring replication + learned-selector hold-out set.

24 additional MIPLIB instances (locked in data/extension_manifest.json with
verified optima, 1k-150k integer variables, 30KB-2MB). Protocol:
6 policies x {60, 300} s x 2 seeds, run tag prefix p45x_, separate output root
so the original 16-instance tables stay untouched.

Usage:
  /opt/anaconda3/envs/env310/bin/python scripts/run_grid_ext.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine  # noqa: E402

DATA = ROOT / "data" / "instances"
OUTPUT_ROOT = ROOT / "results" / "grid_ext"

POLICIES = [
    "best_bound",
    "depth_first",
    "hybrid_best_bound_depth",
    "best_estimate",
    "boltzmann_adaptive",
    "random_uniform",
]
BUDGETS = [60.0, 300.0]
SEEDS = [0, 1]

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
    root_lp_proxy_mode="variable_only",
)


def run_task(task: dict) -> dict:
    run_id = task["run_id"]
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
    except Exception:  # noqa: BLE001
        return {"run_id": run_id, "status": "failed", "error": traceback.format_exc(limit=8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items = json.loads((ROOT / "data" / "extension_manifest.json").read_text())
    tasks = [
        {
            "run_id": f"p45x_{item['instance_name']}_{policy}_b{int(b)}_s{seed}",
            "instance_file": item["filename"],
            "policy": policy,
            "budget": b,
            "seed": seed,
        }
        for item in items
        for b in BUDGETS
        for seed in SEEDS
        for policy in POLICIES
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.dry_run:
        print(f"tasks: {len(tasks)}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"ext tasks: {len(tasks)} | workers: {args.workers}", flush=True)
    ledger = OUTPUT_ROOT / "ext_ledger.jsonl"
    done = 0
    with mp.Pool(args.workers) as pool, open(ledger, "a") as handle:
        for result in pool.imap_unordered(run_task, tasks):
            done += 1
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if result["status"] != "skipped_existing":
                print(f"[{done}/{len(tasks)}] {result['run_id']} status={result['status']} wall={result.get('wall_seconds')}s", flush=True)
            if result["status"] == "failed":
                print(result.get("error", ""), flush=True)
    print("EXT GRID COMPLETE", flush=True)


if __name__ == "__main__":
    main()
