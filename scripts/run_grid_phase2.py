"""Phase 2: learned_portfolio runs under the phase-1 protocol (CPLEX backend).

192 fresh runs: 16 instances x 4 budgets x 3 seeds. Model:
selector_model_round27_merged_cv_selected.json (variable-level features only,
matching the phase-1 proxy protocol; auto mode would also resolve to
variable_only - set explicitly for audit clarity).

The three wrapper portfolios (budgeted / instance_aware / multi_feature) are
NOT re-run: they resolve deterministically to fixed phase-1 policies via
config thresholds and root features already logged in every phase-1 summary;
derive_portfolios.py reconstructs them exactly, with a run-level audit trail.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
import sys

sys.path.insert(0, sys_path)

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine  # noqa: E402

DATA = ROOT / "data" / "instances"
OUTPUT_ROOT = ROOT / "results" / "grid"
MODEL = ROOT / "data" / "models" / "selector_model_round27_merged_cv_selected.json"

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
    root_lp_proxy_mode="variable_only",
    policy="learned_portfolio",
    learned_policy_model_path=str(MODEL),
    learned_policy_default="boltzmann_adaptive",
)


def load_instances() -> list[dict]:
    return json.loads((ROOT / "data" / "dataset_manifest_round1.json").read_text())["items"]


def run_task(task: dict) -> dict:
    run_id = f"p45_{task['instance_name']}_learned_portfolio_b{int(task['budget'])}_s{task['seed']}"
    for existing in sorted((OUTPUT_ROOT / "results" / "raw").glob(f"{run_id}_*_summary.json")):
        try:
            payload = json.loads(existing.read_text())
            if str(payload.get("run_id", "")).startswith(run_id + "_"):
                return {"run_id": run_id, "status": "skipped_existing"}
        except (json.JSONDecodeError, OSError):
            continue

    config = BranchAndBoundConfig(
        instance_path=str(DATA / task["instance_file"]),
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
            "resolved_policy": summary.resolved_policy,
            "resolution_reason": summary.policy_resolution_reason,
            "incumbent": summary.incumbent_objective,
            "gap": summary.final_gap,
            "nodes": summary.nodes_evaluated,
            "ttff": summary.time_to_first_feasible,
            "wall_seconds": round(time.monotonic() - started, 2),
        }
    except Exception:  # noqa: BLE001
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
    args = parser.parse_args()

    tasks = [
        {"instance_name": item["instance_name"], "instance_file": item["filename"], "budget": b, "seed": s}
        for item in load_instances()
        for b in BUDGETS
        for s in SEEDS
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    ledger = OUTPUT_ROOT / "phase2_ledger.jsonl"
    print(f"learned tasks: {len(tasks)} | workers: {args.workers}", flush=True)
    done = 0
    with mp.Pool(args.workers) as pool, open(ledger, "a") as handle:
        for result in pool.imap_unordered(run_task, tasks):
            done += 1
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            if result["status"] != "skipped_existing":
                print(
                    f"[{done}/{len(tasks)}] {result['run_id']} status={result['status']} "
                    f"resolved={result.get('resolved_policy')} wall={result.get('wall_seconds')}s",
                    flush=True,
                )
            if result["status"] == "failed":
                print(result.get("error", ""), flush=True)
    print("PHASE2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
