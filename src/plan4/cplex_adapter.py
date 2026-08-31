"""CPLEX backend adapter for the plan4 transport node-selection harness.

Ported 1:1 from ``GurobiRelaxationAdapter`` in ``plan4/branch_and_bound.py``
(source lines 112-1192, snapshot 2026-08-29). All pure-Python helper logic
(rounding, anchoring, candidate scoring, pseudocost bookkeeping, heuristic
dispatch and budget splitting) is copied verbatim; only the solver-touching
internals are re-implemented against the ``cplex`` Python API (CPLEX 22.1.1,
conda env310).

Semantic mapping decisions (documented for audit):
- ``gp.read``            -> ``Cplex.read`` with .gz instances decompressed to a
                            cache directory (CPLEX cannot read gzipped MPS).
- ``model.copy()``       -> ``cplex.Cplex(other)`` (fresh model, no shared
                            search state; mirrors Gurobi copy semantics).
- ``model.relax()``      -> persistent LP object with all types continuous;
                            node bounds are applied/restored in place instead
                            of copying per node (LP optima are unaffected).
- Params: OutputFlag=0   -> per-object stream suppression (copies too);
          Threads        -> parameters.threads;
          Seed           -> parameters.randomseed;
          MIPFocus=1     -> parameters.emphasis.mip = 1;
          Heuristics=0.8 -> parameters.mip.strategy.heuristiceffort;
          Cutoff         -> uppercutoff (min) / lowercutoff (max), same
                            ``native_incumbent - 1e-6 * model_sense`` formula.
- Ranged rows ('R') in LP proxy metrics are evaluated against both interval
  endpoints; Gurobi exposes these as ordinary rows, so behavior on instances
  with range rows differs slightly (documented, not silently ignored).
- ``variable.Start``     -> ``MIP_starts.add((names, values), effort_level)``.
"""

from __future__ import annotations

import gzip
import math
import time
from pathlib import Path
from typing import Any

import cplex
from cplex.exceptions import CplexError

from .metrics import relative_gap
from .models import NodeState

__all__ = ["CplexRelaxationAdapter"]


def _silence(model: "cplex.Cplex") -> "cplex.Cplex":
    model.set_results_stream(None)
    model.set_log_stream(None)
    model.set_warning_stream(None)
    return model


class CplexRelaxationAdapter:
    def __init__(self, instance_path: str | Path, threads: int, seed: int,
                 time_basis: str = "wall") -> None:
        self.instance_path = Path(instance_path)
        self._threads = threads
        self._seed = seed
        self._time_basis = time_basis

        source = self._prepare_readable_instance(self.instance_path)
        self.original_model = _silence(cplex.Cplex())
        self.original_model.read(str(source))
        self._apply_base_parameters(self.original_model)

        sense = self.original_model.objective.get_sense()
        self.model_sense = 1 if sense == self.original_model.objective.sense.minimize else -1

        types = self.original_model.variables.get_types()
        names = self.original_model.variables.get_names()
        int_type = self.original_model.variables.type.integer
        bin_type = self.original_model.variables.type.binary
        self.integer_var_names = [n for n, t in zip(names, types) if t in {int_type, bin_type}]
        self.integer_var_name_set = set(self.integer_var_names)
        self.binary_var_names = [n for n, t in zip(names, types) if t == bin_type]
        self.binary_var_name_set = set(self.binary_var_names)
        self.pseudocosts: dict[str, dict[str, float]] = {
            name: {"down_sum": 0.0, "down_count": 0.0, "up_sum": 0.0, "up_count": 0.0}
            for name in self.integer_var_names
        }
        lowers = self.original_model.variables.get_lower_bounds()
        uppers = self.original_model.variables.get_upper_bounds()
        self.variable_bounds = {n: {"lb": lb, "ub": ub} for n, lb, ub in zip(names, lowers, uppers)}
        self._int_var_indices = [i for i, t in enumerate(types) if t in {int_type, bin_type}]
        self._int_var_index_by_name = {names[i]: i for i in self._int_var_indices}
        self._name_by_index = {i: n for i, n in enumerate(names)}

        self._lp = cplex.Cplex(self.original_model)
        _silence(self._lp)
        self._apply_base_parameters(self._lp)
        self._lp.variables.set_types(
            [(i, self._lp.variables.type.continuous) for i in range(len(names))]
        )
        self._lp_touched: set[str] = set()

    # ------------------------------------------------------------------ util

    @staticmethod
    def _prepare_readable_instance(instance_path: Path) -> Path:
        if instance_path.suffix == ".gz":
            cache_dir = instance_path.parent / "_decompressed_cache"
            cache_dir.mkdir(exist_ok=True)
            target = cache_dir / (instance_path.name[: -len(".gz")])
            if not target.exists():
                with gzip.open(instance_path, "rb") as zf, open(target, "wb") as out:
                    out.write(zf.read())
            return target
        return instance_path

    def _apply_base_parameters(self, model: "cplex.Cplex") -> None:
        model.parameters.threads.set(self._threads)
        try:
            model.parameters.randomseed.set(self._seed)
        except CplexError:
            pass

    def _restore_lp_bounds(self) -> None:
        if not self._lp_touched:
            return
        names = list(self._lp_touched)
        self._lp.variables.set_lower_bounds(
            [(n, self.variable_bounds[n]["lb"]) for n in names]
        )
        self._lp.variables.set_upper_bounds(
            [(n, self.variable_bounds[n]["ub"]) for n in names]
        )
        self._lp_touched.clear()

    def _collect_integer_solution(self, model: "cplex.Cplex") -> dict[str, float]:
        values = model.solution.get_values()
        return {name: values[self._int_var_index_by_name[name]] for name in self.integer_var_names}

    def _native_objective_from_standardized(self, objective_value: float) -> float:
        return self.model_sense * objective_value

    def _improving_cutoff(self, incumbent_objective: float) -> float:
        native_incumbent = self._native_objective_from_standardized(incumbent_objective)
        return native_incumbent - 1e-6 * self.model_sense

    def _configure_heuristic_model(
        self,
        model: "cplex.Cplex",
        time_limit_seconds: float,
        incumbent_objective: float | None = None,
    ) -> None:
        _silence(model)
        model.parameters.threads.set(1)
        model.parameters.timelimit.set(max(time_limit_seconds, 1e-3))
        if getattr(self, "_time_basis", "wall") == "cpu":
            model.parameters.clocktype.set(2)
        try:
            model.parameters.emphasis.mip.set(1)
            model.parameters.mip.strategy.heuristiceffort.set(0.8)
        except CplexError:
            pass
        if incumbent_objective is not None:
            cutoff = self._improving_cutoff(incumbent_objective)
            if self.model_sense == 1:
                model.parameters.mip.tolerances.uppercutoff.set(cutoff)
            else:
                model.parameters.mip.tolerances.lowercutoff.set(cutoff)

    def _set_bounds(self, model: "cplex.Cplex", assignments: list[tuple[str, float, float]]) -> bool:
        """Apply (name, lb, ub) bound clamps; False if any window empties."""
        lower_pairs: list[tuple[str, float]] = []
        upper_pairs: list[tuple[str, float]] = []
        for var_name, lb, ub in assignments:
            base = self.variable_bounds[var_name]
            new_lb = max(base["lb"], lb) if lb is not None else base["lb"]
            new_ub = min(base["ub"], ub) if ub is not None else base["ub"]
            if new_lb > new_ub + 1e-9:
                return False
            lower_pairs.append((var_name, new_lb))
            upper_pairs.append((var_name, new_ub))
        if lower_pairs:
            model.variables.set_lower_bounds(lower_pairs)
            model.variables.set_upper_bounds(upper_pairs)
        return True

    def _apply_node_fixings(self, node: NodeState, model: "cplex.Cplex") -> bool:
        assignments = []
        for var_name, bounds in node.fixings.items():
            assignments.append((var_name, bounds.get("lb"), bounds.get("ub")))
        return self._set_bounds(model, assignments)

    def _rounded_value(self, var_name: str, value: float) -> float:
        bounds = self.variable_bounds[var_name]
        rounded = round(value)
        return min(max(rounded, bounds["lb"]), bounds["ub"])

    def _anchor_value(
        self,
        var_name: str,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
    ) -> float:
        if incumbent_solution is not None and var_name in incumbent_solution:
            return self._rounded_value(var_name, incumbent_solution[var_name])
        return self._rounded_value(var_name, node.integer_solution.get(var_name, 0.0))

    def _apply_start_values(
        self,
        node: NodeState,
        model: "cplex.Cplex",
        incumbent_solution: dict[str, float] | None = None,
    ) -> None:
        starts: dict[str, float] = {}
        for var_name, lp_value in node.integer_solution.items():
            if var_name not in self._int_var_index_by_name:
                continue
            lp_rounded = self._rounded_value(var_name, lp_value)
            incumbent_rounded = self._anchor_value(var_name, node, incumbent_solution)
            preferred_start = incumbent_rounded
            if abs(lp_value - lp_rounded) + 1e-9 < abs(lp_value - incumbent_rounded):
                preferred_start = lp_rounded
            starts[var_name] = preferred_start
        self._add_mip_start(model, starts)

    @staticmethod
    def _add_mip_start(model: "cplex.Cplex", starts: dict[str, float]) -> None:
        if not starts:
            return
        names = list(starts.keys())
        values = [starts[n] for n in names]
        model.MIP_starts.add((names, values), model.MIP_starts.effort_level.auto)

    # ------------------------------------------------- LP proxy feature math

    @staticmethod
    def _sample_constraint_indices(total: int, sample_limit: int) -> list[int]:
        if sample_limit <= 0:
            raise ValueError("constraint sample limit must be positive")
        if sample_limit >= total:
            return list(range(total))
        raw_indices = [min(int((rank + 0.5) * total / sample_limit), total - 1) for rank in range(sample_limit)]
        indices: list[int] = []
        used: set[int] = set()
        for raw_index in raw_indices:
            candidate = raw_index
            while candidate in used and candidate < total - 1:
                candidate += 1
            while candidate in used and candidate > 0:
                candidate -= 1
            if candidate in used:
                continue
            used.add(candidate)
            indices.append(candidate)
        if len(indices) < sample_limit:
            for candidate in range(total):
                if candidate in used:
                    continue
                used.add(candidate)
                indices.append(candidate)
                if len(indices) >= sample_limit:
                    break
        return sorted(indices[:sample_limit])

    def _compute_lp_proxy_metrics(
        self,
        node: NodeState,
        *,
        lp_proxy_mode: str,
        constraint_sample_limit: int | None,
    ) -> dict[str, Any]:
        if lp_proxy_mode not in {"full", "sampled", "variable_only", "skip"}:
            raise ValueError(f"Unsupported root LP proxy mode: {lp_proxy_mode}")
        integer_count = max(len(self.integer_var_names), 1)
        binary_count = max(len(self.binary_var_names), 1)

        model = self._lp
        total_constraints = model.linear_constraints.get_num()
        metrics: dict[str, Any] = {
            "root_lp_proxy_mode": lp_proxy_mode,
            "root_lp_proxy_total_constraints": total_constraints,
            "root_lp_proxy_evaluated_constraints": 0,
            "root_lp_proxy_constraint_sample_limit": constraint_sample_limit,
        }

        if lp_proxy_mode == "skip":
            metrics.update(
                {
                    "root_integrality_gap_l1": None,
                    "root_binary_near_integral_ratio": None,
                    "root_binary_midpoint_ratio": None,
                    "root_lp_rounding_violated_row_ratio": None,
                    "root_lp_rounding_violation_mean": None,
                    "root_lp_rounding_violation_max": None,
                }
            )
            return metrics

        metrics.update(
            {
                "root_integrality_gap_l1": (
                    sum(abs(value - self._rounded_value(var_name, value)) for var_name, value in node.integer_solution.items())
                    / integer_count
                ),
                "root_binary_near_integral_ratio": (
                    sum(
                        1
                        for var_name in self.binary_var_names
                        if abs(
                            node.integer_solution.get(var_name, 0.0)
                            - self._rounded_value(var_name, node.integer_solution.get(var_name, 0.0))
                        )
                        <= 0.05 + 1e-9
                    )
                    / binary_count
                ),
                "root_binary_midpoint_ratio": (
                    sum(
                        1
                        for var_name in self.binary_var_names
                        if 0.25 - 1e-9 <= node.integer_solution.get(var_name, 0.0) <= 0.75 + 1e-9
                    )
                    / binary_count
                ),
            }
        )
        if lp_proxy_mode == "variable_only":
            metrics.update(
                {
                    "root_lp_rounding_violated_row_ratio": None,
                    "root_lp_rounding_violation_mean": None,
                    "root_lp_rounding_violation_max": None,
                }
            )
            return metrics

        if lp_proxy_mode == "sampled" and total_constraints > 0:
            if constraint_sample_limit is None or constraint_sample_limit <= 0:
                raise ValueError("sampled root LP proxy mode requires a positive constraint sample limit")
            row_indices = self._sample_constraint_indices(total_constraints, min(constraint_sample_limit, total_constraints))
        else:
            row_indices = range(total_constraints)

        senses = model.linear_constraints.get_senses()
        rhs_values = model.linear_constraints.get_rhs()
        range_values = None
        if "R" in senses:
            try:
                range_values = model.linear_constraints.get_range_values()
            except CplexError:
                range_values = None
        lp_values = {
            name: value
            for name, value in zip(
                self._lp.variables.get_names(),
                self._lp.solution.get_values(),
            )
        }

        violated_rows = 0
        total_violation = 0.0
        max_violation = 0.0
        for row_index in row_indices:
            row = model.linear_constraints.get_rows(row_index)
            lhs_value = 0.0
            for var_name, coefficient in zip(row.ind, row.val):
                if var_name in self.integer_var_name_set:
                    value = self._rounded_value(var_name, node.integer_solution.get(var_name, 0.0))
                else:
                    value = lp_values.get(var_name, 0.0)
                lhs_value += coefficient * value

            rhs_value = rhs_values[row_index]
            sense = senses[row_index]
            if sense == "R" and range_values is not None:
                span = range_values[row_index]
                lo = rhs_value if span > 0 else rhs_value + span
                hi = rhs_value + span if span > 0 else rhs_value
                scale = max(abs(rhs_value), 1.0)
                normalized_violation = max(lo - lhs_value, lhs_value - hi, 0.0) / scale
            else:
                scale = max(abs(rhs_value), 1.0)
                if sense == "L":
                    normalized_violation = max(lhs_value - rhs_value, 0.0) / scale
                elif sense == "G":
                    normalized_violation = max(rhs_value - lhs_value, 0.0) / scale
                else:
                    normalized_violation = abs(lhs_value - rhs_value) / scale

            if normalized_violation > 1e-9:
                violated_rows += 1
            total_violation += normalized_violation
            max_violation = max(max_violation, normalized_violation)

        evaluated_constraint_count = len(list(row_indices))
        denominator = max(evaluated_constraint_count, 1)
        metrics.update(
            {
                "root_lp_proxy_evaluated_constraints": evaluated_constraint_count,
                "root_lp_rounding_violated_row_ratio": violated_rows / denominator,
                "root_lp_rounding_violation_mean": total_violation / denominator,
                "root_lp_rounding_violation_max": max_violation,
            }
        )
        return metrics

    # ------------------------------------------------------------ heuristics

    def _binary_center_value(
        self,
        var_name: str,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
    ) -> float:
        node_bounds = node.fixings.get(var_name, {})
        if node_bounds.get("lb") is not None and node_bounds.get("ub") is not None and abs(node_bounds["lb"] - node_bounds["ub"]) <= 1e-9:
            return float(node_bounds["lb"])
        if incumbent_solution is not None and var_name in incumbent_solution:
            return self._rounded_value(var_name, incumbent_solution[var_name])
        value = node.integer_solution.get(var_name, 0.0)
        return self._rounded_value(var_name, value)

    def _candidate_local_branching_vars(
        self,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
        max_binary_vars: int,
    ) -> list[tuple[str, float]]:
        candidates: list[tuple[str, float, float, str]] = []
        for var_name in self.binary_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            center_value = self._binary_center_value(var_name, node, incumbent_solution)
            disagreement = abs(lp_value - center_value)
            ambiguity = 0.5 - abs(lp_value - 0.5)
            candidates.append((var_name, center_value, disagreement, ambiguity))
        candidates.sort(key=lambda item: (-item[2], -item[3], item[0]))
        return [(var_name, center_value) for var_name, center_value, _, _ in candidates[: max(0, max_binary_vars)]]

    def _solve_local_branching_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_solution: dict[str, float] | None,
        radius: int,
        max_binary_vars: int,
        incumbent_objective: float | None = None,
    ) -> tuple[float | None, dict[str, float] | None]:
        if time_limit_seconds <= 0.0 or radius <= 0 or max_binary_vars <= 0:
            return None, None
        candidates = self._candidate_local_branching_vars(node, incumbent_solution, max_binary_vars)
        if not candidates:
            return None, None

        radius = min(radius, len(candidates))
        model = cplex.Cplex(self.original_model)
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        if not self._apply_node_fixings(node, model):
            return None, None

        starts: dict[str, float] = {}
        constraint_names: list[str] = []
        constraint_coeffs: list[float] = []
        center_ones = 0
        for var_name, center_value in candidates:
            constraint_names.append(var_name)
            if center_value >= 0.5:
                # distance term: (1 - x); accumulate as -x with a constant on the rhs
                constraint_coeffs.append(-1.0)
                center_ones += 1
                starts[var_name] = 1.0
            else:
                constraint_coeffs.append(1.0)
                starts[var_name] = 0.0
        for var_name, value in node.integer_solution.items():
            if var_name in self._int_var_index_by_name and var_name not in {name for name, _ in candidates}:
                starts[var_name] = self._rounded_value(var_name, value)
        self._add_mip_start(model, starts)

        model.linear_constraints.add(
            lin_expr=[(constraint_names, constraint_coeffs)],
            senses=["L"],
            rhs=[float(radius - center_ones)],
            names=["plan4_local_branching"],
        )
        if not self._solve_mip_for_solution(model):
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.solution.get_objective_value(), solution

    def _solve_plain_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_objective: float | None = None,
        incumbent_solution: dict[str, float] | None = None,
    ) -> tuple[float | None, dict[str, float] | None]:
        if time_limit_seconds <= 0.0:
            return None, None
        model = cplex.Cplex(self.original_model)
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        if not self._apply_node_fixings(node, model):
            return None, None
        self._apply_start_values(node, model, incumbent_solution)
        if not self._solve_mip_for_solution(model):
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.solution.get_objective_value(), solution

    def _solve_repair_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        max_free_integer_vars: int,
        incumbent_objective: float | None = None,
        incumbent_solution: dict[str, float] | None = None,
    ) -> tuple[float | None, dict[str, float] | None]:
        if time_limit_seconds <= 0.0:
            return None, None
        model = cplex.Cplex(self.original_model)
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        if not self._apply_node_fixings(node, model):
            return None, None

        fractional_items = sorted(
            node.fractional_variables.items(),
            key=lambda item: abs(item[1] - round(item[1])),
            reverse=True,
        )
        free_variables = {
            var_name
            for var_name, _ in fractional_items[: max(0, max_free_integer_vars)]
        }
        starts: dict[str, float] = {}
        assignments: list[tuple[str, float, float]] = []
        fixed_count = 0
        for var_name, value in node.integer_solution.items():
            anchor_value = self._anchor_value(var_name, node, incumbent_solution)
            lp_rounded = self._rounded_value(var_name, value)
            starts[var_name] = anchor_value if abs(value - anchor_value) <= abs(value - lp_rounded) else lp_rounded
            if var_name in free_variables:
                continue
            target_value = anchor_value if abs(lp_rounded - anchor_value) <= 1e-9 else lp_rounded
            assignments.append((var_name, target_value, target_value))

        fixed_count = len(assignments)
        if not self._set_bounds(model, assignments):
            return None, None
        self._add_mip_start(model, starts)

        if fixed_count == 0:
            return None, None
        if not self._solve_mip_for_solution(model):
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.solution.get_objective_value(), solution

    @staticmethod
    def _solve_mip_for_solution(model: "cplex.Cplex") -> bool:
        try:
            model.solve()
        except CplexError:
            return False
        try:
            return model.solution.get_solution_type() != model.solution.type.none
        except CplexError:
            return False

    def _candidate_diving_vars(
        self,
        node: NodeState,
        incumbent_solution: dict[str, float],
    ) -> list[tuple[str, float]]:
        candidates: list[tuple[float, str, float]] = []
        for var_name in self.integer_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            incumbent_value = self._anchor_value(var_name, node, incumbent_solution)
            lp_rounded = self._rounded_value(var_name, lp_value)
            disagreement = abs(lp_rounded - incumbent_value)
            fractionality = abs(lp_value - round(lp_value))
            pseudo = self.pseudocosts.get(var_name, {})
            pseudo_strength = 0.0
            if pseudo.get("down_count", 0.0) > 0.0:
                pseudo_strength += pseudo["down_sum"] / pseudo["down_count"]
            if pseudo.get("up_count", 0.0) > 0.0:
                pseudo_strength += pseudo["up_sum"] / pseudo["up_count"]
            priority = 3.0 * disagreement + fractionality + 0.05 * math.log1p(max(pseudo_strength, 0.0))
            candidates.append((priority, var_name, incumbent_value))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [(var_name, incumbent_value) for _, var_name, incumbent_value in candidates]

    def diving_disagreement_count(
        self,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
    ) -> int | None:
        if incumbent_solution is None:
            return None
        count = 0
        for var_name in self.integer_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            lp_rounded = self._rounded_value(var_name, lp_value)
            incumbent_rounded = self._anchor_value(var_name, node, incumbent_solution)
            if abs(lp_rounded - incumbent_rounded) > 1e-9:
                count += 1
        return count

    def _candidate_lp_guided_vars(
        self,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
    ) -> list[tuple[str, float, float]]:
        candidates: list[tuple[float, str, float, float]] = []
        for var_name in self.integer_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            lp_rounded = self._rounded_value(var_name, lp_value)
            fractionality = abs(lp_value - round(lp_value))
            disagreement = 0.0
            if incumbent_solution is not None:
                anchor_value = self._anchor_value(var_name, node, incumbent_solution)
                disagreement = 1.0 if abs(lp_rounded - anchor_value) > 1e-9 else 0.0
            pseudo = self.pseudocosts.get(var_name, {})
            pseudo_count = pseudo.get("down_count", 0.0) + pseudo.get("up_count", 0.0)
            pseudo_strength = 0.0
            if pseudo.get("down_count", 0.0) > 0.0:
                pseudo_strength += pseudo["down_sum"] / pseudo["down_count"]
            if pseudo.get("up_count", 0.0) > 0.0:
                pseudo_strength += pseudo["up_sum"] / pseudo["up_count"]
            uncertainty = (
                3.0 * fractionality
                + 1.5 * disagreement
                + 1.0 / (pseudo_count + 1.0)
                - 0.02 * math.log1p(max(pseudo_strength, 0.0))
            )
            candidates.append((uncertainty, var_name, lp_rounded, fractionality))
        candidates.sort(key=lambda item: (-item[0], -item[3], item[1]))
        return [(var_name, lp_rounded, fractionality) for _, var_name, lp_rounded, fractionality in candidates]

    def rins_agreement_count(
        self,
        node: NodeState,
        incumbent_solution: dict[str, float] | None,
        integral_tolerance: float,
    ) -> int | None:
        if incumbent_solution is None:
            return None
        count = 0
        for var_name in self.integer_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            lp_rounded = self._rounded_value(var_name, lp_value)
            incumbent_rounded = self._anchor_value(var_name, node, incumbent_solution)
            if (
                abs(lp_value - lp_rounded) <= max(integral_tolerance, 0.0) + 1e-9
                and abs(lp_rounded - incumbent_rounded) <= 1e-9
            ):
                count += 1
        return count

    def _solve_incumbent_diving_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_objective: float | None,
        incumbent_solution: dict[str, float] | None,
        base_free_integer_vars: int,
        stage_factor: int,
    ) -> tuple[float | None, dict[str, float] | None]:
        if (
            time_limit_seconds <= 0.0
            or incumbent_objective is None
            or incumbent_solution is None
            or base_free_integer_vars <= 0
        ):
            return None, None

        candidates = self._candidate_diving_vars(node, incumbent_solution)
        if not candidates:
            return None, None

        stage_counts = [min(len(candidates), base_free_integer_vars)]
        expanded = min(len(candidates), max(base_free_integer_vars * max(stage_factor, 1), base_free_integer_vars))
        if expanded > stage_counts[0]:
            stage_counts.append(expanded)

        remaining_time = time_limit_seconds
        for stage_index, free_count in enumerate(stage_counts):
            budget = remaining_time / max(len(stage_counts) - stage_index, 1)
            model = cplex.Cplex(self.original_model)
            self._configure_heuristic_model(model, budget, incumbent_objective)
            if not self._apply_node_fixings(node, model):
                return None, None

            free_variables = {var_name for var_name, _ in candidates[:free_count]}
            starts: dict[str, float] = {}
            assignments: list[tuple[str, float, float]] = []
            for var_name, lp_value in node.integer_solution.items():
                anchor_value = self._anchor_value(var_name, node, incumbent_solution)
                lp_rounded = self._rounded_value(var_name, lp_value)
                starts[var_name] = anchor_value if abs(lp_value - anchor_value) <= abs(lp_value - lp_rounded) else lp_rounded
                if var_name in free_variables:
                    continue
                target_value = anchor_value if abs(lp_rounded - anchor_value) <= 1.0 else lp_rounded
                assignments.append((var_name, target_value, target_value))

            if not assignments:
                continue
            if not self._set_bounds(model, assignments):
                return None, None
            self._add_mip_start(model, starts)
            if self._solve_mip_for_solution(model):
                solution = self._collect_integer_solution(model)
                return self.model_sense * model.solution.get_objective_value(), solution
            remaining_time = max(remaining_time - budget, 0.0)
            if remaining_time < 1e-2:
                break
        return None, None

    def _solve_lp_guided_diving_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_objective: float | None,
        incumbent_solution: dict[str, float] | None,
        base_free_integer_vars: int,
        stage_factor: int,
        integral_tolerance: float,
    ) -> tuple[float | None, dict[str, float] | None]:
        if time_limit_seconds <= 0.0 or base_free_integer_vars <= 0:
            return None, None

        candidates = self._candidate_lp_guided_vars(node, incumbent_solution)
        if not candidates:
            return None, None

        stage_counts = [min(len(candidates), base_free_integer_vars)]
        expanded = min(len(candidates), max(base_free_integer_vars * max(stage_factor, 1), base_free_integer_vars))
        if expanded > stage_counts[0]:
            stage_counts.append(expanded)

        remaining_time = time_limit_seconds
        for stage_index, free_count in enumerate(stage_counts):
            budget = remaining_time / max(len(stage_counts) - stage_index, 1)
            model = cplex.Cplex(self.original_model)
            self._configure_heuristic_model(model, budget, incumbent_objective)
            if not self._apply_node_fixings(node, model):
                return None, None

            self._apply_start_values(node, model, incumbent_solution)

            default_free = [
                var_name
                for var_name, _, fractionality in candidates
                if fractionality > integral_tolerance + 1e-9
            ]
            if len(default_free) > free_count:
                free_variables = [var_name for var_name, _, _ in candidates[:free_count]]
            else:
                free_variables = list(default_free)
                for var_name, _, _ in candidates:
                    if var_name in free_variables:
                        continue
                    if len(free_variables) >= free_count:
                        break
                    free_variables.append(var_name)
            free_set = set(free_variables)

            assignments: list[tuple[str, float, float]] = []
            for var_name, lp_value in node.integer_solution.items():
                if var_name in free_set:
                    continue
                target_value = self._rounded_value(var_name, lp_value)
                assignments.append((var_name, target_value, target_value))

            if not assignments:
                continue
            if not self._set_bounds(model, assignments):
                return None, None
            if self._solve_mip_for_solution(model):
                solution = self._collect_integer_solution(model)
                return self.model_sense * model.solution.get_objective_value(), solution
            remaining_time = max(remaining_time - budget, 0.0)
            if remaining_time < 1e-2:
                break
        return None, None

    def _solve_rins_like_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_objective: float | None,
        incumbent_solution: dict[str, float] | None,
        max_free_integer_vars: int,
        min_agreement_vars: int,
        integral_tolerance: float,
    ) -> tuple[float | None, dict[str, float] | None]:
        if (
            time_limit_seconds <= 0.0
            or incumbent_objective is None
            or incumbent_solution is None
            or max_free_integer_vars <= 0
        ):
            return None, None

        agreement_count = self.rins_agreement_count(node, incumbent_solution, integral_tolerance)
        if agreement_count is None or agreement_count < max(min_agreement_vars, 0):
            return None, None

        model = cplex.Cplex(self.original_model)
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        if not self._apply_node_fixings(node, model):
            return None, None
        self._apply_start_values(node, model, incumbent_solution)

        uncertain_candidates: list[tuple[float, str, float, float]] = []
        agreement_assignments: list[tuple[str, float, float]] = []
        uncertain_assignments: list[tuple[str, float, float]] = []
        for var_name in self.integer_var_names:
            node_bounds = node.fixings.get(var_name, {})
            lower_bound = node_bounds.get("lb")
            upper_bound = node_bounds.get("ub")
            if lower_bound is not None and upper_bound is not None and abs(lower_bound - upper_bound) <= 1e-9:
                continue
            lp_value = node.integer_solution.get(var_name)
            if lp_value is None:
                continue
            incumbent_value = self._anchor_value(var_name, node, incumbent_solution)
            lp_rounded = self._rounded_value(var_name, lp_value)
            fractionality = abs(lp_value - round(lp_value))
            agreement_fixable = (
                abs(lp_value - lp_rounded) <= max(integral_tolerance, 0.0) + 1e-9
                and abs(lp_rounded - incumbent_value) <= 1e-9
            )
            if agreement_fixable:
                agreement_assignments.append((var_name, incumbent_value, incumbent_value))
                continue
            pseudo = self.pseudocosts.get(var_name, {})
            pseudo_count = pseudo.get("down_count", 0.0) + pseudo.get("up_count", 0.0)
            pseudo_strength = 0.0
            if pseudo.get("down_count", 0.0) > 0.0:
                pseudo_strength += pseudo["down_sum"] / pseudo["down_count"]
            if pseudo.get("up_count", 0.0) > 0.0:
                pseudo_strength += pseudo["up_sum"] / pseudo["up_count"]
            disagreement = 1.0 if abs(lp_rounded - incumbent_value) > 1e-9 else 0.0
            priority = (
                2.5 * disagreement
                + 2.0 * fractionality
                + 1.0 / (pseudo_count + 1.0)
                - 0.02 * math.log1p(max(pseudo_strength, 0.0))
            )
            uncertain_candidates.append((priority, var_name, incumbent_value, lp_rounded))

        uncertain_candidates.sort(key=lambda item: (-item[0], item[1]))
        free_set = {var_name for _, var_name, _, _ in uncertain_candidates[: max(0, max_free_integer_vars)]}

        for _, var_name, incumbent_value, lp_rounded in uncertain_candidates:
            if var_name in free_set:
                continue
            target_value = incumbent_value if abs(lp_rounded - incumbent_value) > 1e-9 else lp_rounded
            uncertain_assignments.append((var_name, target_value, target_value))

        fixed_count = len(agreement_assignments) + len(uncertain_assignments)
        if fixed_count <= 0:
            return None, None
        if not self._set_bounds(model, agreement_assignments + uncertain_assignments):
            return None, None
        if not self._solve_mip_for_solution(model):
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.solution.get_objective_value(), solution

    # ----------------------------------------------------------- dispatcher

    def heuristic_incumbent(
        self,
        node: NodeState,
        time_limit_seconds: float,
        max_free_integer_vars: int,
        incumbent_objective: float | None = None,
        incumbent_solution: dict[str, float] | None = None,
        diving_enabled: bool = True,
        diving_free_integer_vars: int = 24,
        diving_stage_factor: int = 2,
        diving_min_depth: int = 8,
        diving_max_disagreement_vars: int = 64,
        lp_guided_enabled: bool = False,
        lp_guided_free_integer_vars: int = 48,
        lp_guided_stage_factor: int = 3,
        lp_guided_min_depth: int = 8,
        lp_guided_min_disagreement_vars: int = 96,
        lp_guided_integral_tolerance: float = 0.15,
        rins_enabled: bool = False,
        rins_free_integer_vars: int = 96,
        rins_min_depth: int = 10,
        rins_min_agreement_vars: int = 64,
        rins_integral_tolerance: float = 0.05,
        local_branching_enabled: bool = True,
        local_branching_radius: int = 8,
        local_branching_max_binary_vars: int = 64,
        local_branching_max_fractional_vars: int = 24,
        local_branching_max_gap: float = 0.20,
        plain_enabled: bool = True,
    ) -> tuple[float | None, dict[str, float] | None, str | None]:
        if time_limit_seconds <= 0.0:
            return None, None, None
        if not plain_enabled:
            repair_obj, repair_solution = self._solve_repair_heuristic(
                node,
                time_limit_seconds,
                max_free_integer_vars,
                incumbent_objective,
                incumbent_solution,
            )
            return repair_obj, repair_solution, ("repair" if repair_obj is not None else None)
        plain_budget = time_limit_seconds
        diving_disagreement_vars = self.diving_disagreement_count(node, incumbent_solution)
        diving_allowed = (
            diving_enabled
            and incumbent_solution is not None
            and incumbent_objective is not None
            and diving_free_integer_vars > 0
            and node.depth >= max(diving_min_depth, 0)
            and diving_disagreement_vars is not None
            and diving_disagreement_vars <= max(diving_max_disagreement_vars, 0)
        )
        lp_guided_allowed = (
            lp_guided_enabled
            and incumbent_solution is not None
            and lp_guided_free_integer_vars > 0
            and node.depth >= max(lp_guided_min_depth, 0)
            and diving_disagreement_vars is not None
            and diving_disagreement_vars >= max(lp_guided_min_disagreement_vars, 0)
        )
        rins_agreement_vars = self.rins_agreement_count(node, incumbent_solution, rins_integral_tolerance)
        rins_allowed = (
            rins_enabled
            and incumbent_solution is not None
            and incumbent_objective is not None
            and rins_free_integer_vars > 0
            and node.depth >= max(rins_min_depth, 0)
            and rins_agreement_vars is not None
            and rins_agreement_vars >= max(rins_min_agreement_vars, 0)
        )
        incumbent_gap = relative_gap(incumbent_objective, node.lp_objective)
        local_branching_allowed = (
            local_branching_enabled
            and incumbent_solution is not None
            and len(node.fractional_variables) <= max(local_branching_max_fractional_vars, 0)
            and incumbent_gap is not None
            and incumbent_gap <= local_branching_max_gap + 1e-9
        )
        if diving_allowed or lp_guided_allowed or rins_allowed or local_branching_allowed:
            plain_budget = max(min(time_limit_seconds * 0.5, 0.04), min(time_limit_seconds, 0.02))
        plain_start = time.monotonic()
        plain_obj, plain_solution = self._solve_plain_heuristic(
            node,
            plain_budget,
            incumbent_objective,
            incumbent_solution,
        )
        plain_elapsed = time.monotonic() - plain_start
        remaining = max(time_limit_seconds - plain_elapsed, 0.0)

        plain_improved_incumbent = (
            plain_obj is not None
            and (incumbent_objective is None or plain_obj < incumbent_objective - 1e-9)
        )
        if plain_improved_incumbent and not (diving_allowed or lp_guided_allowed or local_branching_allowed):
            return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)
        if remaining < 1e-2:
            return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if diving_allowed and remaining >= 1e-2:
            dive_start = time.monotonic()
            diving_obj, diving_solution = self._solve_incumbent_diving_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                diving_free_integer_vars,
                diving_stage_factor,
            )
            dive_elapsed = time.monotonic() - dive_start
            if diving_obj is not None and (plain_obj is None or diving_obj <= plain_obj + 1e-9):
                return diving_obj, diving_solution, "incumbent_diving"
            remaining = max(remaining - dive_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if lp_guided_allowed and remaining >= 1e-2:
            lp_guided_start = time.monotonic()
            lp_guided_obj, lp_guided_solution = self._solve_lp_guided_diving_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                lp_guided_free_integer_vars,
                lp_guided_stage_factor,
                lp_guided_integral_tolerance,
            )
            lp_guided_elapsed = time.monotonic() - lp_guided_start
            if lp_guided_obj is not None and (plain_obj is None or lp_guided_obj <= plain_obj + 1e-9):
                return lp_guided_obj, lp_guided_solution, "lp_guided_diving"
            remaining = max(remaining - lp_guided_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if rins_allowed and remaining >= 1e-2:
            rins_start = time.monotonic()
            rins_obj, rins_solution = self._solve_rins_like_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                rins_free_integer_vars,
                rins_min_agreement_vars,
                rins_integral_tolerance,
            )
            rins_elapsed = time.monotonic() - rins_start
            if rins_obj is not None and (plain_obj is None or rins_obj <= plain_obj + 1e-9):
                return rins_obj, rins_solution, "rins_like"
            remaining = max(remaining - rins_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if local_branching_allowed and remaining >= 1e-2:
            lb_start = time.monotonic()
            local_obj, local_solution = self._solve_local_branching_heuristic(
                node,
                remaining,
                incumbent_solution,
                local_branching_radius,
                local_branching_max_binary_vars,
                incumbent_objective,
            )
            lb_elapsed = time.monotonic() - lb_start
            if local_obj is not None and (plain_obj is None or local_obj <= plain_obj + 1e-9):
                return local_obj, local_solution, "local_branching"
            remaining = max(remaining - lb_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        repair_obj, repair_solution = self._solve_repair_heuristic(
            node,
            remaining,
            max_free_integer_vars,
            incumbent_objective,
            incumbent_solution,
        )
        if repair_obj is None:
            return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)
        if plain_obj is None or repair_obj <= plain_obj + 1e-9:
            return repair_obj, repair_solution, "repair"
        return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

    # ---------------------------------------------------------- B&B support

    def solve_node(
        self,
        node: NodeState,
        time_limit_seconds: float | None = None,
        lp_proxy_mode: str = "skip",
        lp_proxy_constraint_sample_limit: int | None = None,
    ) -> NodeState:
        self._restore_lp_bounds()

        assignments: list[tuple[str, float | None, float | None]] = [
            (var_name, bounds.get("lb"), bounds.get("ub")) for var_name, bounds in node.fixings.items()
        ]
        empty_window = False
        lower_pairs: list[tuple[str, float]] = []
        upper_pairs: list[tuple[str, float]] = []
        for var_name, lb, ub in assignments:
            current_lb = self._lp.variables.get_lower_bounds(var_name)
            current_ub = self._lp.variables.get_upper_bounds(var_name)
            new_lb = max(current_lb, lb) if lb is not None else current_lb
            new_ub = min(current_ub, ub) if ub is not None else current_ub
            if new_lb > new_ub + 1e-9:
                empty_window = True
                break
            lower_pairs.append((var_name, new_lb))
            upper_pairs.append((var_name, new_ub))
            self._lp_touched.add(var_name)
        if empty_window:
            node.status = "invalid_branch"
            return node
        if lower_pairs:
            self._lp.variables.set_lower_bounds(lower_pairs)
            self._lp.variables.set_upper_bounds(upper_pairs)

        if time_limit_seconds is not None:
            self._lp.parameters.timelimit.set(max(time_limit_seconds, 1e-3))
            if getattr(self, "_time_basis", "wall") == "cpu":
                self._lp.parameters.clocktype.set(2)
        try:
            self._lp.solve()
        except CplexError as exc:
            node.status = f"solver_error_{type(exc).__name__}"
            return node

        status = self._lp.solution.get_status()
        # A relaxed copy of a MIP keeps problem_type == MIP in the Python API, so
        # the solver reports MIP status codes (101/121/...) even though the model
        # has no integer variables left; both code families are accepted here.
        optimal_statuses = {1, 101, 102}  # CPX_STAT_OPTIMAL / CPXMIP_OPTIMAL(_TOL)
        infeasible_statuses = {3, 103, 119, 121}  # LP/MIP infeasible + infeasible-or-unbounded
        if status in infeasible_statuses:
            node.status = "infeasible"
            return node
        if status not in optimal_statuses:  # not optimal
            node.status = f"solver_status_{status}"
            return node

        native_objective = self._lp.solution.get_objective_value()
        standardized_objective = self.model_sense * native_objective
        node.lp_objective = standardized_objective
        node.reported_lp_objective = native_objective

        values = self._lp.solution.get_values()
        integer_solution = {}
        fractional = {}
        for name, index in self._int_var_index_by_name.items():
            value = values[index]
            integer_solution[name] = value
            if abs(value - round(value)) > 1e-6:
                fractional[name] = value
        node.integer_solution = integer_solution
        node.fractional_variables = fractional
        node.status = "integral" if not fractional else "fractional"
        if lp_proxy_mode != "skip":
            proxy_start = time.monotonic()
            node.lp_proxy_metrics = self._compute_lp_proxy_metrics(
                node,
                lp_proxy_mode=lp_proxy_mode,
                constraint_sample_limit=lp_proxy_constraint_sample_limit,
            )
            node.lp_proxy_metrics["root_lp_proxy_elapsed_seconds"] = time.monotonic() - proxy_start
        else:
            node.lp_proxy_metrics = {
                "root_lp_proxy_mode": "skip",
                "root_lp_proxy_total_constraints": self._lp.linear_constraints.get_num(),
                "root_lp_proxy_evaluated_constraints": 0,
                "root_lp_proxy_constraint_sample_limit": lp_proxy_constraint_sample_limit,
                "root_lp_proxy_elapsed_seconds": 0.0,
            }
        return node

    def select_branch_variable(self, node: NodeState) -> tuple[str, float] | None:
        if not node.fractional_variables:
            return None
        scored = []
        for var_name, value in node.fractional_variables.items():
            fractionality = abs(value - round(value))
            floor_value = math.floor(value)
            frac_down = max(value - floor_value, 1e-6)
            frac_up = max(math.ceil(value) - value, 1e-6)
            pseudo = self.pseudocosts[var_name]
            has_down = pseudo["down_count"] > 0
            has_up = pseudo["up_count"] > 0
            if has_down and has_up:
                down_score = (pseudo["down_sum"] / pseudo["down_count"]) * frac_down
                up_score = (pseudo["up_sum"] / pseudo["up_count"]) * frac_up
                score = min(down_score, up_score) + 1e-6 * max(down_score, up_score)
            else:
                score = 0.5 - abs(value - 0.5 - floor_value)
            scored.append((score, var_name, value))
        scored.sort(reverse=True)
        _, var_name, value = scored[0]
        return var_name, value

    def update_pseudocosts(self, parent: NodeState, children: list[NodeState]) -> None:
        if parent.branch_variable is None or parent.branch_value is None or parent.lp_objective is None:
            return
        variable = parent.branch_variable
        pseudo = self.pseudocosts.get(variable)
        if pseudo is None:
            return
        floor_value = math.floor(parent.branch_value)
        frac_down = max(parent.branch_value - floor_value, 1e-6)
        frac_up = max(math.ceil(parent.branch_value) - parent.branch_value, 1e-6)

        for child in children:
            if child.lp_objective is None:
                continue
            improvement = max(child.lp_objective - parent.lp_objective, 0.0)
            if child.branch_direction == "le":
                pseudo["down_sum"] += improvement / frac_down
                pseudo["down_count"] += 1.0
            elif child.branch_direction == "ge":
                pseudo["up_sum"] += improvement / frac_up
                pseudo["up_count"] += 1.0

    def branch(self, node: NodeState, created_step: int, child_index_start: int) -> list[NodeState]:
        branch = self.select_branch_variable(node)
        if branch is None:
            return []
        var_name, value = branch
        floor_value = math.floor(value)
        ceil_value = math.ceil(value)
        current_bounds = node.fixings.get(var_name, {})
        original_bounds = self.variable_bounds[var_name]
        lower_bound = current_bounds.get("lb", original_bounds["lb"])
        upper_bound = current_bounds.get("ub", original_bounds["ub"])

        children: list[NodeState] = []
        candidates = [
            ("le", {"lb": lower_bound, "ub": min(upper_bound, floor_value)}),
            ("ge", {"lb": max(lower_bound, ceil_value), "ub": upper_bound}),
        ]
        for offset, (direction, bounds) in enumerate(candidates):
            if bounds["lb"] > bounds["ub"] + 1e-9:
                continue
            child_fixings = {name: dict(val) for name, val in node.fixings.items()}
            child_fixings[var_name] = bounds
            child = NodeState(
                node_id=f"n{child_index_start + offset}",
                parent_id=node.node_id,
                depth=node.depth + 1,
                created_step=created_step,
                fixings=child_fixings,
                branch_variable=var_name,
                branch_value=value,
                branch_direction=direction,
            )
            children.append(child)
        return children
