"""Unit tests for the CPLEX backend adapter (plan4and5 migration).

Run with:  /opt/anaconda3/envs/env310/bin/python -m pytest tests/test_cplex_adapter.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import cplex
import pytest

from plan4.branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine
from plan4.cplex_adapter import CplexRelaxationAdapter
from plan4.models import NodeState

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TOY = DATA_DIR / "toy_cli_instance.mps"


def make_adapter(path: Path = TOY, threads: int = 1, seed: int = 0) -> CplexRelaxationAdapter:
    return CplexRelaxationAdapter(path, threads, seed)


@pytest.fixture(scope="module")
def adapter() -> CplexRelaxationAdapter:
    return make_adapter()


def write_max_model(tmp_path: Path) -> Path:
    """max x1 + 2 x2  s.t. x1 + x2 <= 1.5, x binary  ->  LP 2.5, MIP 2.0."""
    m = cplex.Cplex()
    m.set_results_stream(None)
    m.variables.add(names=["x1", "x2"], types=[m.variables.type.binary] * 2)
    m.objective.set_sense(m.objective.sense.maximize)
    m.objective.set_linear([("x1", 1.0), ("x2", 2.0)])
    m.linear_constraints.add(lin_expr=[[["x1", "x2"], [1.0, 1.0]]], senses=["L"], rhs=[1.5])
    target = tmp_path / "maxtoy.lp"  # .lp preserves the maximize sense; classic MPS does not
    m.write(str(target))
    return target


class TestAdapterBasics:
    def test_load_toy(self, adapter: CplexRelaxationAdapter) -> None:
        assert adapter.integer_var_names, "toy must have integer variables"
        assert set(adapter.binary_var_names) == set(adapter.integer_var_names)
        assert adapter.model_sense == 1  # minimize
        for name in adapter.integer_var_names:
            assert name in adapter.variable_bounds

    def test_root_solve_fractional(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        assert root.status in {"fractional", "integral"}
        assert root.lp_objective is not None
        assert root.reported_lp_objective == pytest.approx(root.lp_objective, abs=1e-9)

    def test_bound_restore_between_nodes(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="r", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        base_obj = root.lp_objective

        branch = adapter.select_branch_variable(root)
        assert branch is not None
        var_name, value = branch
        children = adapter.branch(root, created_step=1, child_index_start=1)
        assert len(children) == 2
        solved = []
        for child in children:
            adapter.solve_node(child, time_limit_seconds=10.0)
            solved.append(child)
        # both children exist; at least one must have a valid LP bound >= root bound
        valid_bounds = [c.lp_objective for c in solved if c.lp_objective is not None]
        assert valid_bounds and min(valid_bounds) >= base_obj - 1e-6

        # solve a fresh root-equivalent node: bounds must have been restored
        fresh = NodeState(node_id="r2", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(fresh, time_limit_seconds=10.0)
        assert fresh.lp_objective == pytest.approx(base_obj, abs=1e-9)

    def test_invalid_branch_detection(self, adapter: CplexRelaxationAdapter) -> None:
        name = adapter.integer_var_names[0]
        bounds = adapter.variable_bounds[name]
        node = NodeState(
            node_id="bad",
            parent_id=None,
            depth=0,
            created_step=0,
            fixings={name: {"lb": bounds["ub"] + 5.0, "ub": bounds["ub"] + 10.0}},
        )
        adapter.solve_node(node, time_limit_seconds=5.0)
        assert node.status == "invalid_branch"

    def test_infeasible_detection(self, tmp_path: Path) -> None:
        m = cplex.Cplex()
        m.set_results_stream(None)
        m.variables.add(names=["x"], types=[m.variables.type.binary])
        m.linear_constraints.add(lin_expr=[[["x"], [1.0]]], senses=["G"], rhs=[2.0])
        path = tmp_path / "infeasible.mps"
        m.write(str(path))
        adapter = CplexRelaxationAdapter(path, 1, 0)
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=5.0)
        assert root.status == "infeasible"


class TestHeuristics:
    def test_plain_heuristic_finds_optimum(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        obj, solution, backend = adapter.heuristic_incumbent(
            root,
            time_limit_seconds=5.0,
            max_free_integer_vars=12,
            plain_enabled=True,
        )
        assert obj is not None
        assert backend is not None
        for value in solution.values():
            assert abs(value - round(value)) < 1e-6

    def test_repair_only_mode(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        obj, solution, backend = adapter.heuristic_incumbent(
            root,
            time_limit_seconds=5.0,
            max_free_integer_vars=12,
            plain_enabled=False,
        )
        if obj is not None:
            assert backend == "repair"
            for value in solution.values():
                assert abs(value - round(value)) < 1e-6

    def test_local_branching_hamming_semantics(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        # seed an incumbent via plain heuristic
        incumbent_obj, incumbent_solution, _ = adapter.heuristic_incumbent(
            root, time_limit_seconds=5.0, max_free_integer_vars=12, plain_enabled=True
        )
        assert incumbent_solution is not None
        obj, solution = adapter._solve_local_branching_heuristic(
            root, 3.0, incumbent_solution, radius=2, max_binary_vars=8, incumbent_objective=incumbent_obj
        )
        if solution is not None:
            centers = {
                name: adapter._rounded_value(name, incumbent_solution.get(name, 0.0))
                for name in adapter.binary_var_names
            }
            distance = sum(
                1 for name, center in centers.items() if abs(solution.get(name, center) - center) > 1e-6
            )
            assert distance <= 2 + 1e-9, "returned solution violates the local-branching radius"

    def test_heuristics_do_not_mutate_shared_state(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        base_lp = root.lp_objective
        adapter.heuristic_incumbent(root, time_limit_seconds=2.0, max_free_integer_vars=12, plain_enabled=True)
        fresh = NodeState(node_id="r2", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(fresh, time_limit_seconds=10.0)
        assert fresh.lp_objective == pytest.approx(base_lp, abs=1e-9)


class TestLpProxyMetrics:
    def test_full_mode_ranges(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0, lp_proxy_mode="full")
        metrics = root.lp_proxy_metrics
        assert metrics["root_lp_proxy_mode"] == "full"
        assert metrics["root_lp_proxy_total_constraints"] > 0
        for key in (
            "root_integrality_gap_l1",
            "root_binary_near_integral_ratio",
            "root_binary_midpoint_ratio",
            "root_lp_rounding_violated_row_ratio",
        ):
            value = metrics[key]
            assert value is None or 0.0 <= value <= 1.0 + 1e-9

    def test_sampled_mode(self, adapter: CplexRelaxationAdapter) -> None:
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0, lp_proxy_mode="sampled", lp_proxy_constraint_sample_limit=1)
        metrics = root.lp_proxy_metrics
        assert metrics["root_lp_proxy_evaluated_constraints"] == min(
            1, metrics["root_lp_proxy_total_constraints"]
        )


class TestMaximizeSense:
    def test_max_model_end_to_end(self, tmp_path: Path) -> None:
        model_path = write_max_model(tmp_path)
        adapter = CplexRelaxationAdapter(model_path, 1, 0)
        assert adapter.model_sense == -1
        root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
        adapter.solve_node(root, time_limit_seconds=10.0)
        assert root.status == "fractional"
        assert root.lp_objective == pytest.approx(-2.5, abs=1e-6)  # standardized min form

        obj, solution, _ = adapter.heuristic_incumbent(
            root, time_limit_seconds=5.0, max_free_integer_vars=12, plain_enabled=True
        )
        assert obj is not None
        assert obj == pytest.approx(-2.0, abs=1e-6)

    def test_max_engine_summary(self, tmp_path: Path) -> None:
        model_path = write_max_model(tmp_path)
        config = BranchAndBoundConfig(
            instance_path=str(model_path),
            policy="best_bound",
            node_limit=100,
            time_limit_seconds=10.0,
            random_seed=0,
            run_tag="max_engine_test",
            output_root=str(tmp_path / "out"),
            solver_backend="cplex",
        )
        engine = BranchAndBoundEngine(config)
        summary = engine.run()
        assert summary.status == "optimal"
        assert summary.incumbent_objective == pytest.approx(-2.0, abs=1e-6)
        assert summary.time_to_first_feasible is not None


class TestEngineEndToEnd:
    @pytest.mark.parametrize("policy", ["best_bound", "depth_first", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid"])
    def test_toy_runs(self, policy: str, tmp_path: Path) -> None:
        config = BranchAndBoundConfig(
            instance_path=str(TOY),
            policy=policy,
            node_limit=100,
            time_limit_seconds=10.0,
            random_seed=0,
            run_tag=f"toy_{policy}",
            output_root=str(tmp_path / "out"),
            solver_backend="cplex",
        )
        engine = BranchAndBoundEngine(config)
        summary = engine.run()
        assert summary.status in {"optimal", "time_limit", "node_limit", "infeasible"}
        if summary.incumbent_objective is not None:
            assert summary.best_bound is not None
            assert summary.incumbent_objective >= summary.best_bound - 1e-6
            assert summary.time_to_first_feasible is not None
            assert summary.final_gap is not None and summary.final_gap >= -1e-9
        assert Path(summary.summary_path).exists()
        assert Path(summary.trace_path).exists()

    def test_solver_backend_validation(self, tmp_path: Path) -> None:
        config = BranchAndBoundConfig(
            instance_path=str(TOY),
            policy="best_bound",
            solver_backend=" nonexistent ",
            output_root=str(tmp_path / "out"),
        )
        with pytest.raises(ValueError):
            BranchAndBoundEngine(config)
