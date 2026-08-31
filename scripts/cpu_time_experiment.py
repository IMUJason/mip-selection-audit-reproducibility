#!/usr/bin/env python
"""CPU-time vs wall-clock robustness subset (cross-component finding 3).

Re-runs the value-level seed-variance instances identified by the grid's
invariant monitor (rail507, drayage-25-23, csched008, air05) with the three
deterministic bound-driven policies at the 300 s budget, three seeds, under
BOTH time bases (wall and cpu), interleaved on the same machine. The wall
twins give a same-conditions contrast to the historical grid runs.

Ledger: results/cpu_time/cpu_time_ledger.jsonl (one row per finished run).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine

DATA = ROOT / "data" / "instances"
OUT = ROOT / "results" / "cpu_time"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = OUT / "cpu_time_ledger.jsonl"

INSTANCES = ["rail507", "drayage-25-23", "csched008", "air05"]
POLICIES = ["best_bound", "depth_first", "hybrid_best_bound_depth"]
BUDGET = 300.0
SEEDS = [0, 1, 2]

COMMON = dict(
    node_limit=400, batch_size=1, shortlist_size=16, heuristic_enabled=True,
    heuristic_time_limit_seconds=0.08, heuristic_every_n_selected=1,
    heuristic_repair_max_free_integer_vars=10,
    heuristic_local_branching_enabled=True,
    heuristic_local_branching_radius=8,
    heuristic_local_branching_max_binary_vars=48,
    heuristic_local_branching_max_fractional_vars=24,
    heuristic_local_branching_max_gap=0.20,
    threads=1, solver_backend="cplex", root_lp_proxy_mode="variable_only",
)


def main() -> None:
    done = set()
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                done.add(json.loads(line)["cell"])
            except (json.JSONDecodeError, KeyError):
                continue
    tasks = [
        (inst, pol, seed, basis)
        for inst in INSTANCES
        for pol in POLICIES
        for seed in SEEDS
        for basis in ("wall", "cpu")   # interleaved per cell
        if f"{inst}|{pol}|s{seed}|{basis}" not in done
    ]
    print(f"{len(tasks)} runs to go", flush=True)
    with LEDGER.open("a", encoding="utf-8") as ledger:
        for k, (inst, pol, seed, basis) in enumerate(tasks):
            cell = f"{inst}|{pol}|s{seed}|{basis}"
            run_tag = f"cpu45_{inst}_{pol}_b300_s{seed}_{basis}"
            cfg = BranchAndBoundConfig(
                instance_path=str(DATA / f"{inst}.mps.gz"),
                policy=pol, time_limit_seconds=BUDGET, random_seed=seed,
                run_tag=run_tag, output_root=str(OUT), time_basis=basis, **COMMON,
            )
            t0 = time.monotonic()
            try:
                s = BranchAndBoundEngine(cfg).run()
                row = dict(cell=cell, basis=basis, instance=inst, policy=pol,
                           seed=seed, status=s.status,
                           incumbent=s.incumbent_objective, gap=s.final_gap,
                           nodes=s.nodes_evaluated,
                           wall_seconds=round(time.monotonic() - t0, 1))
            except Exception:
                row = dict(cell=cell, basis=basis, instance=inst, policy=pol,
                           seed=seed, status="failed",
                           error=traceback.format_exc(limit=6),
                           wall_seconds=round(time.monotonic() - t0, 1))
            ledger.write(json.dumps(row) + "\n")
            ledger.flush()
            print(f"[{k+1}/{len(tasks)}] {cell} -> {row['status']} "
                  f"inc={row.get('incumbent')} ({row['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
