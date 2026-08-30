"""Run learned_v2 (full-action-space tree) fresh under the standard protocols.

Original 16 instances: 4 budgets x 3 seeds (192 runs, tag p45_learned_v2_*)
matching the round-27 comparison protocol.
Extension 24 instances (TRUE hold-out, never seen in training/depth choice):
{60, 300} s x 3 seeds (144 runs, tag p45x_learned_v2_*).
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
MODEL = ROOT / "data" / "models" / "selector_model_learned_v2.json"

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
    learned_policy_default="best_bound",
)


def run_task(task: dict) -> dict:
    run_id = task["run_id"]
    out_root = ROOT / ("results/grid" if task["cohort"] == "orig" else "results/grid_ext")
    for existing in sorted((out_root / "results" / "raw").glob(f"{run_id}_*_summary.json")):
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
        output_root=str(out_root),
        **COMMON,
    )
    started = time.monotonic()
    try:
        summary = BranchAndBoundEngine(config).run()
        return {
            "run_id": run_id,
            "status": summary.status,
            "resolved_policy": summary.resolved_policy,
            "incumbent": summary.incumbent_objective,
            "gap": summary.final_gap,
            "ttff": summary.time_to_first_feasible,
            "wall_seconds": round(time.monotonic() - started, 2),
        }
    except Exception:  # noqa: BLE001
        return {"run_id": run_id, "status": "failed", "error": traceback.format_exc(limit=8)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = []
    for item in json.loads((ROOT / "data" / "dataset_manifest_round1.json").read_text())["items"]:
        for b in [10.0, 60.0, 300.0, 900.0]:
            for s in [0, 1, 2]:
                tasks.append(
                    {
                        "run_id": f"p45_{item['instance_name']}_learned_v2_b{int(b)}_s{s}",
                        "instance_file": item["filename"],
                        "budget": b,
                        "seed": s,
                        "cohort": "orig",
                    }
                )
    for item in json.loads((ROOT / "data" / "extension_manifest.json").read_text()):
        for b in [60.0, 300.0]:
            for s in [0, 1, 2]:
                tasks.append(
                    {
                        "run_id": f"p45x_{item['instance_name']}_learned_v2_b{int(b)}_s{s}",
                        "instance_file": item["filename"],
                        "budget": b,
                        "seed": s,
                        "cohort": "ext",
                    }
                )
    if args.limit:
        tasks = tasks[: args.limit]

    ledger = ROOT / "results" / "grid" / "learned_v2_ledger.jsonl"
    print(f"learned_v2 tasks: {len(tasks)} | workers: {args.workers}", flush=True)
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
    print("LEARNED_V2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
