"""Deep verification of the two flagged anomalies.

V1 (determinism): run the SAME config twice in one process (fresh adapter each
    time) - if results are identical per seed, our code is deterministic and the
    cross-seed variance comes from CPLEX's randomseed tie-breaking.
V2 (stochasticity): inspect trace files - do stochastic policies actually make
    different node selections across seeds on atlanta-ip b300?
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine

DATA = ROOT / "data" / "instances"


def run_once(instance: str, policy: str, budget: float, seed: int, tag: str):
    config = BranchAndBoundConfig(
        instance_path=str(DATA / f"{instance}.mps.gz"),
        policy=policy,
        node_limit=400,
        batch_size=1,
        shortlist_size=16,
        time_limit_seconds=budget,
        random_seed=seed,
        threads=1,
        run_tag=tag,
        output_root=str(ROOT / "results" / "verify"),
        solver_backend="cplex",
        heuristic_enabled=True,
        heuristic_time_limit_seconds=0.08,
        heuristic_repair_max_free_integer_vars=10,
        heuristic_local_branching_enabled=True,
        heuristic_local_branching_radius=8,
        heuristic_local_branching_max_binary_vars=48,
    )
    t0 = time.monotonic()
    s = BranchAndBoundEngine(config).run()
    return {
        "incumbent": s.incumbent_objective,
        "nodes": s.nodes_evaluated,
        "status": s.status,
        "gap": s.final_gap,
        "wall": round(time.monotonic() - t0, 1),
    }


def selection_sequence(trace_path: Path, max_events: int = 40) -> list[str]:
    seq = []
    if not trace_path.exists():
        return seq
    for line in trace_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "selected_node_id" in event:
            seq.append(event["selected_node_id"])
        if len(seq) >= max_events:
            break
    return seq


def main() -> None:
    print("=== V1a: same-seed repeat determinism (air05, greedy_score, b60, seed 0) ===", flush=True)
    a = run_once("air05", "greedy_score", 60.0, 0, "v1a_first")
    b = run_once("air05", "greedy_score", 60.0, 0, "v1a_second")
    print("run1:", a)
    print("run2:", b)
    print("deterministic within seed:", a["incumbent"] == b["incumbent"] and a["nodes"] == b["nodes"])

    print("\n=== V1b: cross-seed (air05, greedy_score, b300) ===", flush=True)
    for seed in [0, 1, 2]:
        r = run_once("air05", "greedy_score", 300.0, seed, f"v1b_s{seed}")
        print(f"seed {seed}:", r)

    print("\n=== V2: selection sequences across seeds (atlanta-ip, boltzmann_adaptive, b300) ===", flush=True)
    logs = ROOT / "results" / "grid" / "logs" / "runs"
    for seed in [0, 1, 2]:
        pattern = f"p45_atlanta-ip_boltzmann_adaptive_b300_s{seed}_*"
        matches = sorted(logs.glob(pattern))
        if not matches:
            print(f"seed {seed}: no trace found")
            continue
        seq = selection_sequence(matches[-1])
        print(f"seed {seed}: first-8 selections {seq[:8]}")

    print("\n=== V2b: LP degeneracy probe - root LP values under two seeds ===", flush=True)
    from plan4.cplex_adapter import CplexRelaxationAdapter
    from plan4.models import NodeState

    for seed in [0, 1]:
        adapter = CplexRelaxationAdapter(DATA / "air05.mps.gz", threads=1, seed=seed)
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=60.0)
        frac = root.fractional_variables
        first = sorted(frac.items(), key=lambda kv: kv[0])[:3]
        print(f"seed {seed}: lp={root.lp_objective!r} n_frac={len(frac)} sample={first}")


if __name__ == "__main__":
    main()
