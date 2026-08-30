"""Solver-native CPLEX node-selection reference baselines (R2#4 response).

Runs stock CPLEX branch-and-bound with each native node-selection strategy
(mip.strategy.nodeselect: 0=auto, 1=depth-first, 2=best-bound, 3=best-estimate,
4=best-estimate-impact) on the 16-instance benchmark, recording incumbent,
gap, nodes and wall time per budget. These are REFERENCE lines for the merged
paper: they quantify how far the audited custom harness sits from the
solver-native strategies under identical budgets.

Usage:
  /opt/anaconda3/envs/env310/bin/python scripts/run_cplex_native.py [--workers 6]
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import cplex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "instances"
OUT = ROOT / "results" / "cplex_native"

NODESELECT = {0: "auto", 1: "depth_first", 2: "best_bound", 3: "best_estimate"}  # code 4 invalid in CPLEX 22.1
BUDGETS = [10.0, 60.0, 300.0, 900.0]
SEEDS = [0, 1, 2]


def readable_mps(path: Path) -> Path:
    if path.suffix == ".gz":
        cache = path.parent / "_decompressed_cache" / path.name[:-3]
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists():
            with gzip.open(path, "rb") as zf, open(cache, "wb") as out:
                out.write(zf.read())
        return cache
    return path


def load_instances() -> list[dict]:
    manifest = json.loads((ROOT / "data" / "dataset_manifest_round1.json").read_text())
    return manifest["items"]


def run_one(task: dict) -> dict:
    out_path = OUT / f"native_{task['instance_name']}_{task['strategy']}_b{int(task['budget'])}_s{task['seed']}.json"
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    started = time.monotonic()
    record = dict(task)
    try:
        model = cplex.Cplex()
        model.set_results_stream(None)
        model.set_log_stream(None)
        model.set_warning_stream(None)
        model.read(str(readable_mps(DATA / task["instance_file"])))
        model.parameters.threads.set(1)
        try:
            model.parameters.randomseed.set(task["seed"])
        except cplex.exceptions.CplexError:
            pass
        model.parameters.timelimit.set(task["budget"])
        model.parameters.mip.strategy.nodeselect.set(task["nodeselect"])

        model.solve()
        sol_type = model.solution.get_solution_type()
        has_solution = sol_type != model.solution.type.none
        model_sense = 1 if model.objective.get_sense() == model.objective.sense.minimize else -1
        native_obj = model.solution.get_objective_value() if has_solution else None
        record.update(
            status=model.solution.get_status_string(),
            has_solution=has_solution,
            incumbent_native=native_obj,
            incumbent_standardized=(model_sense * native_obj) if has_solution else None,
            wall_seconds=round(time.monotonic() - started, 2),
        )
    except Exception:  # noqa: BLE001 - never leak solver exceptions into the pool
        record.update(
            status="error",
            error=traceback.format_exc(limit=6),
            wall_seconds=round(time.monotonic() - started, 2),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = []
    for item in load_instances():
        for budget in BUDGETS:
            for seed in SEEDS:
                for code, name in NODESELECT.items():
                    tasks.append(
                        {
                            "instance_name": item["instance_name"],
                            "instance_file": item["filename"],
                            "budget": budget,
                            "seed": seed,
                            "nodeselect": code,
                            "strategy": name,
                        }
                    )

    OUT.mkdir(parents=True, exist_ok=True)
    ledger = OUT / "native_ledger.jsonl"
    print(f"native tasks: {len(tasks)} | workers: {args.workers}", flush=True)
    done = 0
    with mp.Pool(args.workers) as pool, open(ledger, "a") as handle:
        for record in pool.imap_unordered(run_one, tasks):
            done += 1
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if done % 40 == 0:
                print(f"[{done}/{len(tasks)}] last={record['instance_name']} {record['strategy']}", flush=True)
    print("NATIVE RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
