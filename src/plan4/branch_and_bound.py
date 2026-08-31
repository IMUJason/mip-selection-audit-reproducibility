from __future__ import annotations

import json
import math
import time

# Switchable process clock: wall (default) or CPU time for robustness runs.
_CLOCK = {"fn": time.monotonic}


def _now() -> float:
    return _CLOCK["fn"]()
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # gurobipy is unavailable on the current machine; kept for archival parity
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only on CPLEX-only hosts
    gp = None
    GRB = None

from .cplex_adapter import CplexRelaxationAdapter

from .controller import AdaptiveBetaController
from .learning_selector import (
    ROOT_PRIMAL_SELECTOR_FEATURES,
    collect_tree_split_features,
    load_selector_model,
    predict_selector_policy,
    required_features_for_partial_tree,
)
from .metrics import relative_gap
from .models import BetaConfig, NodeState, RunSummary, ScoreConfig
from .provenance import ensure_run_paths, json_dump, sha256_file, utc_timestamp
from .selectors import NodeSelector, SelectorContext

ROOT_VARIABLE_PROXY_FEATURES = {
    "root_integrality_gap_l1",
    "root_binary_near_integral_ratio",
    "root_binary_midpoint_ratio",
}

ROOT_ROW_PROXY_FEATURES = {
    "root_lp_rounding_violated_row_ratio",
    "root_lp_rounding_violation_mean",
    "root_lp_rounding_violation_max",
}


@dataclass
class BranchAndBoundConfig:
    instance_path: str
    policy: str = "boltzmann_adaptive"
    portfolio_short_policy: str = "boltzmann_adaptive"
    portfolio_long_policy: str = "best_estimate"
    portfolio_time_threshold_seconds: float = 180.0
    instance_aware_short_small_policy: str = "boltzmann_adaptive"
    instance_aware_short_large_policy: str = "safeguarded_hybrid"
    instance_aware_long_policy: str = "best_estimate"
    instance_aware_time_threshold_seconds: float = 180.0
    instance_aware_root_fractional_threshold: int = 5000
    multi_feature_short_low_complexity_policy: str = "boltzmann_adaptive"
    multi_feature_short_high_complexity_policy: str = "safeguarded_hybrid"
    multi_feature_long_policy: str = "best_estimate"
    multi_feature_time_threshold_seconds: float = 180.0
    multi_feature_root_fractional_threshold: int = 5000
    multi_feature_root_fractional_ratio_threshold: float = 0.08
    multi_feature_integer_variable_threshold: int = 100000
    multi_feature_root_solve_seconds_threshold: float = 1.0
    multi_feature_complexity_votes_required: int = 2
    learned_policy_model_path: str | None = None
    learned_policy_default: str = "boltzmann_adaptive"
    root_lp_proxy_mode: str = "auto"
    root_lp_proxy_constraint_sample_limit: int | None = None
    root_primal_probe_time_limit_seconds: float = 0.0
    root_primal_probe_max_free_integer_vars: int = 12
    root_primal_probe_plain_enabled: bool = True
    batch_size: int = 1
    shortlist_size: int | None = 32
    node_limit: int = 300
    time_limit_seconds: float = 60.0
    random_seed: int = 0
    threads: int = 1
    run_tag: str = "run"
    output_root: str | None = None
    heuristic_enabled: bool = True
    heuristic_time_limit_seconds: float = 0.05
    heuristic_every_n_selected: int = 1
    heuristic_repair_max_free_integer_vars: int = 12
    heuristic_diving_enabled: bool = False
    heuristic_diving_free_integer_vars: int = 24
    heuristic_diving_stage_factor: int = 2
    heuristic_diving_min_depth: int = 8
    heuristic_diving_max_disagreement_vars: int = 64
    heuristic_lp_guided_enabled: bool = False
    heuristic_lp_guided_free_integer_vars: int = 48
    heuristic_lp_guided_stage_factor: int = 3
    heuristic_lp_guided_min_depth: int = 8
    heuristic_lp_guided_min_disagreement_vars: int = 96
    heuristic_lp_guided_integral_tolerance: float = 0.15
    heuristic_rins_enabled: bool = False
    heuristic_rins_time_limit_seconds: float = 0.25
    heuristic_rins_every_n_selected: int = 8
    heuristic_rins_free_integer_vars: int = 96
    heuristic_rins_min_depth: int = 10
    heuristic_rins_min_agreement_vars: int = 64
    heuristic_rins_integral_tolerance: float = 0.05
    heuristic_local_branching_enabled: bool = True
    heuristic_local_branching_radius: int = 8
    heuristic_local_branching_max_binary_vars: int = 64
    heuristic_local_branching_max_fractional_vars: int = 24
    heuristic_local_branching_max_gap: float = 0.20
    score_config: ScoreConfig = field(default_factory=ScoreConfig)
    beta_config: BetaConfig = field(default_factory=BetaConfig)
    solver_backend: str = "cplex"
    time_basis: str = "wall"  # "wall" | "cpu" (robustness runs)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score_config"] = self.score_config.to_dict()
        payload["beta_config"] = self.beta_config.to_dict()
        payload["time_basis"] = self.time_basis
        return payload


class GurobiRelaxationAdapter:
    def __init__(self, instance_path: str | Path, threads: int, seed: int) -> None:
        self.instance_path = Path(instance_path)
        self.original_model = gp.read(str(self.instance_path))
        self.original_model.Params.OutputFlag = 0
        self.original_model.Params.Threads = threads
        self.original_model.Params.Seed = seed
        self.relaxed_model = self.original_model.relax()
        self.relaxed_model.Params.OutputFlag = 0
        self.relaxed_model.Params.Threads = threads
        self.relaxed_model.Params.Seed = seed
        self.model_sense = self.original_model.ModelSense
        self.integer_var_names = [
            variable.VarName
            for variable in self.original_model.getVars()
            if variable.VType in {GRB.BINARY, GRB.INTEGER}
        ]
        self.integer_var_name_set = set(self.integer_var_names)
        self.binary_var_names = [
            variable.VarName
            for variable in self.original_model.getVars()
            if variable.VType == GRB.BINARY
        ]
        self.binary_var_name_set = set(self.binary_var_names)
        self.pseudocosts: dict[str, dict[str, float]] = {
            name: {
                "down_sum": 0.0,
                "down_count": 0.0,
                "up_sum": 0.0,
                "up_count": 0.0,
            }
            for name in self.integer_var_names
        }
        self.variable_bounds = {
            variable.VarName: {"lb": variable.LB, "ub": variable.UB}
            for variable in self.original_model.getVars()
        }

    def _collect_integer_solution(self, model: gp.Model) -> dict[str, float]:
        variables = {variable.VarName: variable for variable in model.getVars()}
        return {name: variables[name].X for name in self.integer_var_names}

    def _native_objective_from_standardized(self, objective_value: float) -> float:
        return self.model_sense * objective_value

    def _improving_cutoff(self, incumbent_objective: float) -> float:
        native_incumbent = self._native_objective_from_standardized(incumbent_objective)
        return native_incumbent - 1e-6 * self.model_sense

    def _configure_heuristic_model(
        self,
        model: gp.Model,
        time_limit_seconds: float,
        incumbent_objective: float | None = None,
    ) -> None:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = max(time_limit_seconds, 1e-3)
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.8
        if incumbent_objective is not None:
            model.Params.Cutoff = self._improving_cutoff(incumbent_objective)

    def _apply_node_fixings(self, node: NodeState, variables: dict[str, gp.Var]) -> bool:
        for var_name, bounds in node.fixings.items():
            variable = variables[var_name]
            if bounds.get("lb") is not None:
                variable.LB = max(variable.LB, bounds["lb"])
            if bounds.get("ub") is not None:
                variable.UB = min(variable.UB, bounds["ub"])
            if variable.LB > variable.UB + 1e-9:
                return False
        return True

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
        variables: dict[str, gp.Var],
        incumbent_solution: dict[str, float] | None = None,
    ) -> None:
        for var_name, lp_value in node.integer_solution.items():
            if var_name not in variables:
                continue
            lp_rounded = self._rounded_value(var_name, lp_value)
            incumbent_rounded = self._anchor_value(var_name, node, incumbent_solution)
            preferred_start = incumbent_rounded
            if abs(lp_value - lp_rounded) + 1e-9 < abs(lp_value - incumbent_rounded):
                preferred_start = lp_rounded
            variables[var_name].Start = preferred_start

    @staticmethod
    def _sample_constraints(constraints: list[gp.Constr], sample_limit: int) -> list[gp.Constr]:
        if sample_limit <= 0:
            raise ValueError("constraint sample limit must be positive")
        total = len(constraints)
        if sample_limit >= total:
            return list(constraints)
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
        return [constraints[index] for index in sorted(indices[:sample_limit])]

    def _compute_lp_proxy_metrics(
        self,
        model: gp.Model,
        node: NodeState,
        variables: dict[str, gp.Var],
        *,
        lp_proxy_mode: str,
        constraint_sample_limit: int | None,
    ) -> dict[str, Any]:
        if lp_proxy_mode not in {"full", "sampled", "variable_only", "skip"}:
            raise ValueError(f"Unsupported root LP proxy mode: {lp_proxy_mode}")
        integer_count = max(len(self.integer_var_names), 1)
        binary_count = max(len(self.binary_var_names), 1)

        constraints = model.getConstrs()
        total_constraints = len(constraints)
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

        sampled_constraints = constraints
        if lp_proxy_mode == "sampled" and total_constraints > 0:
            if constraint_sample_limit is None or constraint_sample_limit <= 0:
                raise ValueError("sampled root LP proxy mode requires a positive constraint sample limit")
            sampled_constraints = self._sample_constraints(list(constraints), min(constraint_sample_limit, total_constraints))

        violated_rows = 0
        total_violation = 0.0
        max_violation = 0.0
        for constraint in sampled_constraints:
            row = model.getRow(constraint)
            lhs_value = 0.0
            for index in range(row.size()):
                variable = row.getVar(index)
                coefficient = row.getCoeff(index)
                if variable.VarName in self.integer_var_name_set:
                    value = self._rounded_value(variable.VarName, node.integer_solution.get(variable.VarName, 0.0))
                else:
                    value = variables[variable.VarName].X
                lhs_value += coefficient * value

            rhs_value = constraint.RHS
            scale = max(abs(rhs_value), 1.0)
            if constraint.Sense == "<":
                normalized_violation = max(lhs_value - rhs_value, 0.0) / scale
            elif constraint.Sense == ">":
                normalized_violation = max(rhs_value - lhs_value, 0.0) / scale
            else:
                normalized_violation = abs(lhs_value - rhs_value) / scale

            if normalized_violation > 1e-9:
                violated_rows += 1
            total_violation += normalized_violation
            max_violation = max(max_violation, normalized_violation)

        evaluated_constraint_count = len(sampled_constraints)
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
        model = self.original_model.copy()
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        variables = {variable.VarName: variable for variable in model.getVars()}
        if not self._apply_node_fixings(node, variables):
            return None, None

        neighborhood_expr = gp.LinExpr()
        for var_name, center_value in candidates:
            variable = variables[var_name]
            if center_value >= 0.5:
                neighborhood_expr += 1.0 - variable
                variable.Start = 1.0
            else:
                neighborhood_expr += variable
                variable.Start = 0.0

        for var_name, value in node.integer_solution.items():
            if var_name in variables and var_name not in {name for name, _ in candidates}:
                variables[var_name].Start = self._rounded_value(var_name, value)

        model.addConstr(neighborhood_expr <= radius, name="plan4_local_branching")
        model.optimize()
        if model.SolCount <= 0:
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.ObjVal, solution

    def _solve_plain_heuristic(
        self,
        node: NodeState,
        time_limit_seconds: float,
        incumbent_objective: float | None = None,
        incumbent_solution: dict[str, float] | None = None,
    ) -> tuple[float | None, dict[str, float] | None]:
        if time_limit_seconds <= 0.0:
            return None, None
        model = self.original_model.copy()
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        variables = {variable.VarName: variable for variable in model.getVars()}
        if not self._apply_node_fixings(node, variables):
            return None, None
        self._apply_start_values(node, variables, incumbent_solution)
        model.optimize()
        if model.SolCount <= 0:
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.ObjVal, solution

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
        model = self.original_model.copy()
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        variables = {variable.VarName: variable for variable in model.getVars()}
        if not self._apply_node_fixings(node, variables):
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
        fixed_count = 0
        for var_name, value in node.integer_solution.items():
            variable = variables[var_name]
            anchor_value = self._anchor_value(var_name, node, incumbent_solution)
            lp_rounded = self._rounded_value(var_name, value)
            variable.Start = anchor_value if abs(value - anchor_value) <= abs(value - lp_rounded) else lp_rounded
            if var_name in free_variables:
                continue
            target_value = anchor_value if abs(lp_rounded - anchor_value) <= 1e-9 else lp_rounded
            variable.LB = max(variable.LB, target_value)
            variable.UB = min(variable.UB, target_value)
            if variable.LB > variable.UB + 1e-9:
                return None, None
            fixed_count += 1

        if fixed_count == 0:
            return None, None
        model.optimize()
        if model.SolCount <= 0:
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.ObjVal, solution

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
            model = self.original_model.copy()
            self._configure_heuristic_model(model, budget, incumbent_objective)
            variables = {variable.VarName: variable for variable in model.getVars()}
            if not self._apply_node_fixings(node, variables):
                return None, None

            free_variables = {var_name for var_name, _ in candidates[:free_count]}
            fixed_count = 0
            for var_name, lp_value in node.integer_solution.items():
                variable = variables[var_name]
                anchor_value = self._anchor_value(var_name, node, incumbent_solution)
                lp_rounded = self._rounded_value(var_name, lp_value)
                variable.Start = anchor_value if abs(lp_value - anchor_value) <= abs(lp_value - lp_rounded) else lp_rounded
                if var_name in free_variables:
                    continue
                target_value = anchor_value if abs(lp_rounded - anchor_value) <= 1.0 else lp_rounded
                variable.LB = max(variable.LB, target_value)
                variable.UB = min(variable.UB, target_value)
                if variable.LB > variable.UB + 1e-9:
                    fixed_count = -1
                    break
                fixed_count += 1

            if fixed_count <= 0:
                continue
            model.optimize()
            if model.SolCount > 0:
                solution = self._collect_integer_solution(model)
                return self.model_sense * model.ObjVal, solution
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
            model = self.original_model.copy()
            self._configure_heuristic_model(model, budget, incumbent_objective)
            variables = {variable.VarName: variable for variable in model.getVars()}
            if not self._apply_node_fixings(node, variables):
                return None, None

            self._apply_start_values(node, variables, incumbent_solution)

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

            fixed_count = 0
            for var_name, lp_value in node.integer_solution.items():
                variable = variables[var_name]
                if var_name in free_set:
                    continue
                target_value = self._rounded_value(var_name, lp_value)
                variable.LB = max(variable.LB, target_value)
                variable.UB = min(variable.UB, target_value)
                if variable.LB > variable.UB + 1e-9:
                    fixed_count = -1
                    break
                fixed_count += 1

            if fixed_count <= 0:
                continue
            model.optimize()
            if model.SolCount > 0:
                solution = self._collect_integer_solution(model)
                return self.model_sense * model.ObjVal, solution
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

        model = self.original_model.copy()
        self._configure_heuristic_model(model, time_limit_seconds, incumbent_objective)
        variables = {variable.VarName: variable for variable in model.getVars()}
        if not self._apply_node_fixings(node, variables):
            return None, None
        self._apply_start_values(node, variables, incumbent_solution)

        uncertain_candidates: list[tuple[float, str, float, float]] = []
        fixed_count = 0
        for var_name in self.integer_var_names:
            if var_name not in variables:
                continue
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
                variable = variables[var_name]
                variable.LB = max(variable.LB, incumbent_value)
                variable.UB = min(variable.UB, incumbent_value)
                if variable.LB > variable.UB + 1e-9:
                    return None, None
                fixed_count += 1
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
            variable = variables[var_name]
            target_value = incumbent_value if abs(lp_rounded - incumbent_value) > 1e-9 else lp_rounded
            variable.LB = max(variable.LB, target_value)
            variable.UB = min(variable.UB, target_value)
            if variable.LB > variable.UB + 1e-9:
                return None, None
            fixed_count += 1

        if fixed_count <= 0:
            return None, None
        model.optimize()
        if model.SolCount <= 0:
            return None, None
        solution = self._collect_integer_solution(model)
        return self.model_sense * model.ObjVal, solution

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
        plain_start = _now()
        plain_obj, plain_solution = self._solve_plain_heuristic(
            node,
            plain_budget,
            incumbent_objective,
            incumbent_solution,
        )
        plain_elapsed = _now() - plain_start
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
            dive_start = _now()
            diving_obj, diving_solution = self._solve_incumbent_diving_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                diving_free_integer_vars,
                diving_stage_factor,
            )
            dive_elapsed = _now() - dive_start
            if diving_obj is not None and (plain_obj is None or diving_obj <= plain_obj + 1e-9):
                return diving_obj, diving_solution, "incumbent_diving"
            remaining = max(remaining - dive_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if lp_guided_allowed and remaining >= 1e-2:
            lp_guided_start = _now()
            lp_guided_obj, lp_guided_solution = self._solve_lp_guided_diving_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                lp_guided_free_integer_vars,
                lp_guided_stage_factor,
                lp_guided_integral_tolerance,
            )
            lp_guided_elapsed = _now() - lp_guided_start
            if lp_guided_obj is not None and (plain_obj is None or lp_guided_obj <= plain_obj + 1e-9):
                return lp_guided_obj, lp_guided_solution, "lp_guided_diving"
            remaining = max(remaining - lp_guided_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if rins_allowed and remaining >= 1e-2:
            rins_start = _now()
            rins_obj, rins_solution = self._solve_rins_like_heuristic(
                node,
                remaining,
                incumbent_objective,
                incumbent_solution,
                rins_free_integer_vars,
                rins_min_agreement_vars,
                rins_integral_tolerance,
            )
            rins_elapsed = _now() - rins_start
            if rins_obj is not None and (plain_obj is None or rins_obj <= plain_obj + 1e-9):
                return rins_obj, rins_solution, "rins_like"
            remaining = max(remaining - rins_elapsed, 0.0)
            if remaining < 1e-2:
                return plain_obj, plain_solution, ("plain" if plain_obj is not None else None)

        if local_branching_allowed and remaining >= 1e-2:
            lb_start = _now()
            local_obj, local_solution = self._solve_local_branching_heuristic(
                node,
                remaining,
                incumbent_solution,
                local_branching_radius,
                local_branching_max_binary_vars,
                incumbent_objective,
            )
            lb_elapsed = _now() - lb_start
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

    def solve_node(
        self,
        node: NodeState,
        time_limit_seconds: float | None = None,
        lp_proxy_mode: str = "skip",
        lp_proxy_constraint_sample_limit: int | None = None,
    ) -> NodeState:
        model = self.relaxed_model.copy()
        model.Params.OutputFlag = 0
        if time_limit_seconds is not None:
            model.Params.TimeLimit = max(time_limit_seconds, 1e-3)
        variables = {variable.VarName: variable for variable in model.getVars()}

        for var_name, bounds in node.fixings.items():
            variable = variables[var_name]
            if bounds.get("lb") is not None:
                variable.LB = max(variable.LB, bounds["lb"])
            if bounds.get("ub") is not None:
                variable.UB = min(variable.UB, bounds["ub"])
            if variable.LB > variable.UB + 1e-9:
                node.status = "invalid_branch"
                return node

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            node.status = "infeasible"
            return node
        if model.Status != GRB.OPTIMAL:
            node.status = f"solver_status_{model.Status}"
            return node

        standardized_objective = self.model_sense * model.ObjVal
        node.lp_objective = standardized_objective
        node.reported_lp_objective = model.ObjVal

        integer_solution = {}
        fractional = {}
        for name in self.integer_var_names:
            value = variables[name].X
            integer_solution[name] = value
            if abs(value - round(value)) > 1e-6:
                fractional[name] = value
        node.integer_solution = integer_solution
        node.fractional_variables = fractional
        node.status = "integral" if not fractional else "fractional"
        if lp_proxy_mode != "skip":
            proxy_start = _now()
            node.lp_proxy_metrics = self._compute_lp_proxy_metrics(
                model,
                node,
                variables,
                lp_proxy_mode=lp_proxy_mode,
                constraint_sample_limit=lp_proxy_constraint_sample_limit,
            )
            node.lp_proxy_metrics["root_lp_proxy_elapsed_seconds"] = _now() - proxy_start
        else:
            node.lp_proxy_metrics = {
                "root_lp_proxy_mode": "skip",
                "root_lp_proxy_total_constraints": len(model.getConstrs()),
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


class BranchAndBoundEngine:
    @staticmethod
    def _collect_tree_split_features(tree: dict[str, Any]) -> set[str]:
        return collect_tree_split_features(tree)

    def __init__(self, config: BranchAndBoundConfig) -> None:
        self.config = config
        if config.solver_backend == "gurobi":
            if gp is None:
                raise RuntimeError("solver_backend='gurobi' requested but gurobipy is not installed")
            self.adapter = GurobiRelaxationAdapter(config.instance_path, config.threads, config.random_seed)
        elif config.solver_backend == "cplex":
            self.adapter = CplexRelaxationAdapter(
                config.instance_path, config.threads, config.random_seed,
                time_basis=config.time_basis,
            )
        else:
            raise ValueError(f"Unsupported solver_backend: {config.solver_backend}")
        self.root_fractional_count: int | None = None
        self.root_integer_variable_count: int | None = len(self.adapter.integer_var_names)
        self.root_binary_variable_count: int | None = len(self.adapter.binary_var_names)
        self.root_fractional_ratio: float | None = None
        self.root_solve_seconds: float | None = None
        self.root_lp_objective: float | None = None
        self.root_fractionality_mean: float | None = None
        self.root_fractionality_max: float | None = None
        self.root_integrality_gap_l1: float | None = None
        self.root_binary_near_integral_ratio: float | None = None
        self.root_binary_midpoint_ratio: float | None = None
        self.root_lp_rounding_violated_row_ratio: float | None = None
        self.root_lp_rounding_violation_mean: float | None = None
        self.root_lp_rounding_violation_max: float | None = None
        self.root_lp_proxy_mode: str | None = None
        self.root_lp_proxy_total_constraints: int | None = None
        self.root_lp_proxy_evaluated_constraints: int | None = None
        self.root_lp_proxy_constraint_sample_limit: int | None = None
        self.root_lp_proxy_elapsed_seconds: float | None = None
        self.root_primal_probe_found_feasible: int | None = None
        self.root_primal_probe_relative_gap: float | None = None
        self.root_primal_probe_objective: float | None = None
        self.root_primal_probe_backend: str | None = None
        self.root_primal_probe_elapsed_seconds: float | None = None
        self.root_primal_probe_features_observed = False
        self.policy_resolution_reason: str | None = None
        self.learned_policy_model = (
            load_selector_model(config.learned_policy_model_path)
            if config.policy == "learned_portfolio" and config.learned_policy_model_path is not None
            else None
        )
        self.learned_policy_feature_names = set(self.learned_policy_model["feature_names"]) if self.learned_policy_model is not None else set()
        self.learned_policy_active_feature_names = (
            self._collect_tree_split_features(self.learned_policy_model["tree"])
            if self.learned_policy_model is not None
            else set()
        )
        self.resolved_policy, self.policy_resolution_reason = self._resolve_selector_policy()
        self.selector = NodeSelector(
            policy=self.resolved_policy,
            score_config=config.score_config,
            shortlist_size=config.shortlist_size,
            batch_size=config.batch_size,
            random_seed=config.random_seed,
            starvation_age_ratio=config.beta_config.starvation_age_ratio,
        )
        self.controller = AdaptiveBetaController(config.beta_config, config.score_config)

    def _known_selector_feature_values(self) -> dict[str, float]:
        feature_values = {"time_limit_seconds": float(self.config.time_limit_seconds)}

        scalar_features = {
            "root_fractional_count": self.root_fractional_count,
            "root_fractional_ratio": self.root_fractional_ratio,
            "root_integer_variable_count": self.root_integer_variable_count,
            "root_binary_variable_count": self.root_binary_variable_count,
            "root_solve_seconds": self.root_solve_seconds,
            "root_lp_objective": self.root_lp_objective,
            "root_fractionality_mean": self.root_fractionality_mean,
            "root_fractionality_max": self.root_fractionality_max,
            "root_integrality_gap_l1": self.root_integrality_gap_l1,
            "root_binary_near_integral_ratio": self.root_binary_near_integral_ratio,
            "root_binary_midpoint_ratio": self.root_binary_midpoint_ratio,
            "root_lp_rounding_violated_row_ratio": self.root_lp_rounding_violated_row_ratio,
            "root_lp_rounding_violation_mean": self.root_lp_rounding_violation_mean,
            "root_lp_rounding_violation_max": self.root_lp_rounding_violation_max,
            "root_lp_proxy_elapsed_seconds": self.root_lp_proxy_elapsed_seconds,
        }
        for feature_name, value in scalar_features.items():
            if value is None:
                continue
            feature_values[feature_name] = float(value)

        if self.root_primal_probe_features_observed:
            feature_values["root_primal_probe_found_feasible"] = float(self.root_primal_probe_found_feasible or 0)
            feature_values["root_primal_probe_relative_gap"] = float(
                self.root_primal_probe_relative_gap if self.root_primal_probe_relative_gap is not None else 1.0
            )
            feature_values["root_primal_probe_elapsed_seconds"] = float(self.root_primal_probe_elapsed_seconds or 0.0)

        return feature_values

    def _required_selector_features(self) -> set[str]:
        if self.learned_policy_model is None:
            return set()
        return required_features_for_partial_tree(
            self.learned_policy_model["tree"],
            self._known_selector_feature_values(),
        )

    def _resolve_root_lp_proxy_mode(self) -> str:
        requested_mode = self.config.root_lp_proxy_mode
        if requested_mode not in {"auto", "full", "sampled", "variable_only", "skip"}:
            raise ValueError(f"Unsupported root_lp_proxy_mode: {requested_mode}")
        if requested_mode != "auto":
            return requested_mode
        if self.config.policy != "learned_portfolio" or self.learned_policy_model is None:
            return "full"
        required_features = self._required_selector_features()
        if required_features & ROOT_ROW_PROXY_FEATURES:
            return "full"
        if required_features & ROOT_VARIABLE_PROXY_FEATURES:
            return "variable_only"
        return "skip"

    def _resolve_selector_policy(self, root: NodeState | None = None) -> tuple[str, str]:
        if self.config.policy == "budgeted_portfolio":
            if self.config.time_limit_seconds <= self.config.portfolio_time_threshold_seconds:
                return self.config.portfolio_short_policy, "budgeted_portfolio_short_budget"
            return self.config.portfolio_long_policy, "budgeted_portfolio_long_budget"
        if self.config.policy == "instance_aware_portfolio":
            if self.config.time_limit_seconds > self.config.instance_aware_time_threshold_seconds:
                return self.config.instance_aware_long_policy, "instance_aware_long_budget"
            if root is None:
                return self.config.instance_aware_short_small_policy, "instance_aware_pending_root"
            root_fractional_count = len(root.fractional_variables)
            self.root_fractional_count = root_fractional_count
            if root_fractional_count >= self.config.instance_aware_root_fractional_threshold:
                return (
                    self.config.instance_aware_short_large_policy,
                    f"instance_aware_large_root_fractional_{root_fractional_count}",
                )
            return (
                self.config.instance_aware_short_small_policy,
                f"instance_aware_small_root_fractional_{root_fractional_count}",
            )
        if self.config.policy == "multi_feature_portfolio":
            if self.config.time_limit_seconds > self.config.multi_feature_time_threshold_seconds:
                return self.config.multi_feature_long_policy, "multi_feature_long_budget"
            if root is None:
                return self.config.multi_feature_short_low_complexity_policy, "multi_feature_pending_root"
            root_fractional_count = len(root.fractional_variables)
            self.root_fractional_count = root_fractional_count
            integer_var_count = self.root_integer_variable_count or len(self.adapter.integer_var_names)
            self.root_integer_variable_count = integer_var_count
            self.root_fractional_ratio = root_fractional_count / max(integer_var_count, 1)
            root_solve_seconds = self.root_solve_seconds or 0.0
            votes = [
                root_fractional_count >= self.config.multi_feature_root_fractional_threshold,
                (self.root_fractional_ratio or 0.0) >= self.config.multi_feature_root_fractional_ratio_threshold,
                integer_var_count >= self.config.multi_feature_integer_variable_threshold,
                root_solve_seconds >= self.config.multi_feature_root_solve_seconds_threshold,
            ]
            hit_count = sum(votes)
            if hit_count >= self.config.multi_feature_complexity_votes_required:
                return (
                    self.config.multi_feature_short_high_complexity_policy,
                    f"multi_feature_high_complexity_votes_{hit_count}_of_4",
                )
            return (
                self.config.multi_feature_short_low_complexity_policy,
                f"multi_feature_low_complexity_votes_{hit_count}_of_4",
            )
        if self.config.policy == "learned_portfolio":
            if self.learned_policy_model is None:
                raise ValueError("learned_portfolio requires learned_policy_model_path.")
            if root is None:
                return self.config.learned_policy_default, "learned_portfolio_pending_root"
            resolved_policy, explanation = predict_selector_policy(
                self.learned_policy_model,
                self._known_selector_feature_values(),
            )
            return resolved_policy, f"learned_portfolio_{explanation}"
        return self.config.policy, "direct_policy"

    def _update_root_metrics(self, root: NodeState) -> None:
        self.root_integer_variable_count = self.root_integer_variable_count or len(self.adapter.integer_var_names)
        self.root_binary_variable_count = self.root_binary_variable_count or len(self.adapter.binary_var_names)
        self.root_fractional_count = len(root.fractional_variables)
        self.root_fractional_ratio = self.root_fractional_count / max(self.root_integer_variable_count or 1, 1)
        self.root_lp_objective = root.lp_objective
        fractionality_distances = [abs(value - round(value)) for value in root.fractional_variables.values()]
        self.root_fractionality_mean = (
            sum(fractionality_distances) / len(fractionality_distances) if fractionality_distances else 0.0
        )
        self.root_fractionality_max = max(fractionality_distances, default=0.0)
        self.root_integrality_gap_l1 = root.lp_proxy_metrics.get("root_integrality_gap_l1")
        self.root_binary_near_integral_ratio = root.lp_proxy_metrics.get("root_binary_near_integral_ratio")
        self.root_binary_midpoint_ratio = root.lp_proxy_metrics.get("root_binary_midpoint_ratio")
        self.root_lp_rounding_violated_row_ratio = root.lp_proxy_metrics.get("root_lp_rounding_violated_row_ratio")
        self.root_lp_rounding_violation_mean = root.lp_proxy_metrics.get("root_lp_rounding_violation_mean")
        self.root_lp_rounding_violation_max = root.lp_proxy_metrics.get("root_lp_rounding_violation_max")
        self.root_lp_proxy_mode = root.lp_proxy_metrics.get("root_lp_proxy_mode")
        self.root_lp_proxy_total_constraints = root.lp_proxy_metrics.get("root_lp_proxy_total_constraints")
        self.root_lp_proxy_evaluated_constraints = root.lp_proxy_metrics.get("root_lp_proxy_evaluated_constraints")
        self.root_lp_proxy_constraint_sample_limit = root.lp_proxy_metrics.get("root_lp_proxy_constraint_sample_limit")
        self.root_lp_proxy_elapsed_seconds = root.lp_proxy_metrics.get("root_lp_proxy_elapsed_seconds")

    def _run_root_primal_probe(self, root: NodeState) -> None:
        self.root_primal_probe_found_feasible = 0
        self.root_primal_probe_relative_gap = 1.0
        self.root_primal_probe_objective = None
        self.root_primal_probe_backend = None
        self.root_primal_probe_elapsed_seconds = 0.0
        self.root_primal_probe_features_observed = False

        if root.status == "integral" and root.lp_objective is not None:
            self.root_primal_probe_found_feasible = 1
            self.root_primal_probe_relative_gap = 0.0
            self.root_primal_probe_objective = root.lp_objective
            self.root_primal_probe_backend = "root_integral"
            self.root_primal_probe_features_observed = True
            return

        probe_features_required = set(ROOT_PRIMAL_SELECTOR_FEATURES) & self._required_selector_features()
        if self.config.root_primal_probe_time_limit_seconds <= 0.0:
            if self.config.policy == "learned_portfolio" and probe_features_required:
                raise ValueError(
                    "learned_portfolio model requires root primal probe features, "
                    "but root_primal_probe_time_limit_seconds <= 0."
                )
            return
        if root.status != "fractional":
            return
        if self.config.policy == "learned_portfolio" and not probe_features_required:
            return

        probe_start = _now()
        objective, _, backend = self.adapter.heuristic_incumbent(
            root,
            time_limit_seconds=self.config.root_primal_probe_time_limit_seconds,
            max_free_integer_vars=self.config.root_primal_probe_max_free_integer_vars,
            incumbent_objective=None,
            incumbent_solution=None,
            diving_enabled=False,
            lp_guided_enabled=False,
            rins_enabled=False,
            local_branching_enabled=False,
            plain_enabled=self.config.root_primal_probe_plain_enabled,
        )
        self.root_primal_probe_features_observed = True
        self.root_primal_probe_elapsed_seconds = _now() - probe_start
        self.root_primal_probe_objective = objective
        self.root_primal_probe_backend = backend
        if objective is not None:
            self.root_primal_probe_found_feasible = 1
            self.root_primal_probe_relative_gap = relative_gap(objective, root.lp_objective) or 0.0

    @staticmethod
    def _prune_active_nodes(
        active_nodes: list[NodeState],
        incumbent_objective: float | None,
    ) -> tuple[list[NodeState], int]:
        if incumbent_objective is None:
            return active_nodes, 0
        retained: list[NodeState] = []
        pruned = 0
        for node in active_nodes:
            if node.lp_objective is not None and node.lp_objective >= incumbent_objective - 1e-9:
                pruned += 1
                continue
            retained.append(node)
        return retained, pruned

    def run(self) -> RunSummary:
        run_id = f"{self.config.run_tag}_{int(time.time())}"
        paths = ensure_run_paths(run_id, self.config.output_root)
        instance_hash = sha256_file(self.config.instance_path)
        _CLOCK["fn"] = (
            time.process_time
            if self.config.time_basis == "cpu"
            else time.monotonic
        )
        start = _now()
        trace_handle = paths.trace_path.open("w", encoding="utf-8")

        try:
            beta = self.config.beta_config.initial_beta
            incumbent_objective: float | None = None
            incumbent_solution: dict[str, float] | None = None
            incumbents_found = 0
            time_to_first_feasible: float | None = None
            steps_since_improvement = 0
            nodes_evaluated = 0
            nodes_selected = 0
            node_counter = 1
            status = "unknown"

            root = NodeState(node_id="n0", parent_id=None, depth=0, created_step=0)
            root_start = _now()
            root_lp_proxy_mode = self._resolve_root_lp_proxy_mode()
            root = self.adapter.solve_node(
                root,
                time_limit_seconds=self.config.time_limit_seconds,
                lp_proxy_mode=root_lp_proxy_mode,
                lp_proxy_constraint_sample_limit=self.config.root_lp_proxy_constraint_sample_limit,
            )
            self.root_solve_seconds = _now() - root_start
            self._update_root_metrics(root)
            self._run_root_primal_probe(root)
            nodes_evaluated += 1

            active_nodes: list[NodeState] = []
            if root.status == "integral":
                incumbent_objective = root.lp_objective
                incumbent_solution = root.integer_solution
                incumbents_found = 1
                time_to_first_feasible = 0.0
                status = "optimal"
            elif root.status == "fractional":
                active_nodes.append(root)
            elif root.status in {"infeasible", "invalid_branch"}:
                status = "infeasible"
            else:
                status = root.status

            if self.config.policy in {"instance_aware_portfolio", "multi_feature_portfolio", "learned_portfolio"}:
                self.resolved_policy, self.policy_resolution_reason = self._resolve_selector_policy(root)
                self.selector = NodeSelector(
                    policy=self.resolved_policy,
                    score_config=self.config.score_config,
                    shortlist_size=self.config.shortlist_size,
                    batch_size=self.config.batch_size,
                    random_seed=self.config.random_seed,
                    starvation_age_ratio=self.config.beta_config.starvation_age_ratio,
                )

            trace_handle.write(
                json.dumps(
                    {
                        "timestamp_utc": utc_timestamp(),
                        "event": "policy_resolution",
                        "configured_policy": self.config.policy,
                        "resolved_policy": self.resolved_policy,
                        "policy_resolution_reason": self.policy_resolution_reason,
                        "root_status": root.status,
                        "root_lp_objective": root.lp_objective,
                        "root_fractional_count": self.root_fractional_count,
                        "root_integer_variable_count": self.root_integer_variable_count,
                        "root_binary_variable_count": self.root_binary_variable_count,
                        "root_fractional_ratio": self.root_fractional_ratio,
                        "root_solve_seconds": self.root_solve_seconds,
                        "root_fractionality_mean": self.root_fractionality_mean,
                        "root_fractionality_max": self.root_fractionality_max,
                        "root_integrality_gap_l1": self.root_integrality_gap_l1,
                        "root_binary_near_integral_ratio": self.root_binary_near_integral_ratio,
                        "root_binary_midpoint_ratio": self.root_binary_midpoint_ratio,
                        "root_lp_rounding_violated_row_ratio": self.root_lp_rounding_violated_row_ratio,
                        "root_lp_rounding_violation_mean": self.root_lp_rounding_violation_mean,
                        "root_lp_rounding_violation_max": self.root_lp_rounding_violation_max,
                        "root_lp_proxy_mode": self.root_lp_proxy_mode,
                        "root_lp_proxy_total_constraints": self.root_lp_proxy_total_constraints,
                        "root_lp_proxy_evaluated_constraints": self.root_lp_proxy_evaluated_constraints,
                        "root_lp_proxy_constraint_sample_limit": self.root_lp_proxy_constraint_sample_limit,
                        "root_lp_proxy_elapsed_seconds": self.root_lp_proxy_elapsed_seconds,
                        "root_primal_probe_found_feasible": self.root_primal_probe_found_feasible,
                        "root_primal_probe_relative_gap": self.root_primal_probe_relative_gap,
                        "root_primal_probe_objective": self.root_primal_probe_objective,
                        "root_primal_probe_backend": self.root_primal_probe_backend,
                        "root_primal_probe_elapsed_seconds": self.root_primal_probe_elapsed_seconds,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            step = 0
            while active_nodes:
                elapsed = _now() - start
                if elapsed >= self.config.time_limit_seconds:
                    status = "time_limit"
                    break
                if nodes_evaluated >= self.config.node_limit:
                    status = "node_limit"
                    break

                step += 1
                context = SelectorContext(
                    current_step=step,
                    incumbent_objective=incumbent_objective,
                    beta=beta,
                    batch_size=self.config.batch_size,
                    elapsed_seconds=elapsed,
                    time_limit_seconds=self.config.time_limit_seconds,
                    pseudocosts=self.adapter.pseudocosts,
                )
                selected_nodes, diagnostics = self.selector.select(active_nodes, context)
                if not selected_nodes:
                    status = "stalled"
                    break

                incumbent_before = incumbent_objective
                improvement_in_batch = False

                for batch_index, node in enumerate(selected_nodes):
                    if node not in active_nodes:
                        continue
                    active_nodes.remove(node)
                    nodes_selected += 1
                    node.visit_count += 1
                    node.last_selected_step = step
                    global_pruned_nodes = 0
                    heuristic_backend: str | None = None
                    diving_disagreement_vars = self.adapter.diving_disagreement_count(node, incumbent_solution)
                    diving_allowed = (
                        self.config.heuristic_enabled
                        and self.config.heuristic_diving_enabled
                        and incumbent_solution is not None
                        and incumbent_objective is not None
                        and node.depth >= max(self.config.heuristic_diving_min_depth, 0)
                        and diving_disagreement_vars is not None
                        and diving_disagreement_vars <= max(self.config.heuristic_diving_max_disagreement_vars, 0)
                    )
                    lp_guided_allowed = (
                        self.config.heuristic_enabled
                        and self.config.heuristic_lp_guided_enabled
                        and incumbent_solution is not None
                        and node.depth >= max(self.config.heuristic_lp_guided_min_depth, 0)
                        and diving_disagreement_vars is not None
                        and diving_disagreement_vars >= max(self.config.heuristic_lp_guided_min_disagreement_vars, 0)
                    )
                    rins_agreement_vars = self.adapter.rins_agreement_count(
                        node,
                        incumbent_solution,
                        self.config.heuristic_rins_integral_tolerance,
                    )
                    rins_allowed = (
                        self.config.heuristic_enabled
                        and self.config.heuristic_rins_enabled
                        and incumbent_solution is not None
                        and incumbent_objective is not None
                        and node.depth >= max(self.config.heuristic_rins_min_depth, 0)
                        and rins_agreement_vars is not None
                        and rins_agreement_vars >= max(self.config.heuristic_rins_min_agreement_vars, 0)
                        and (
                            self.config.heuristic_rins_every_n_selected <= 0
                            or nodes_selected % self.config.heuristic_rins_every_n_selected == 0
                        )
                    )

                    if incumbent_objective is not None and node.lp_objective is not None and node.lp_objective >= incumbent_objective - 1e-9:
                        event = {
                            "timestamp_utc": utc_timestamp(),
                            "step": step,
                            "batch_index": batch_index,
                            "selected_node_id": node.node_id,
                            "status": "pruned_by_bound",
                            "node_lp_objective": node.lp_objective,
                            "reported_node_lp_objective": node.reported_lp_objective,
                            "depth": node.depth,
                            "beta_before": beta,
                            "candidate_pool_size": diagnostics.candidate_pool_size,
                            "shortlist_size": diagnostics.shortlist_size,
                            "entropy": diagnostics.entropy,
                            "normalized_entropy": diagnostics.normalized_entropy,
                            "dispersion": diagnostics.dispersion,
                            "starvation_ratio": diagnostics.starvation_ratio,
                            "heuristic_diving_allowed": diving_allowed,
                            "heuristic_lp_guided_allowed": lp_guided_allowed,
                            "heuristic_rins_allowed": rins_allowed,
                            "heuristic_diving_disagreement_vars": diving_disagreement_vars,
                            "heuristic_rins_agreement_vars": rins_agreement_vars,
                            "incumbent_before": incumbent_before,
                            "incumbent_after": incumbent_objective,
                        }
                        trace_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                        continue

                    if (
                        self.config.heuristic_enabled
                        and self.config.heuristic_every_n_selected > 0
                        and nodes_selected % self.config.heuristic_every_n_selected == 0
                    ):
                        heuristic_budget = self.config.heuristic_time_limit_seconds
                        if rins_allowed:
                            heuristic_budget = max(heuristic_budget, self.config.heuristic_rins_time_limit_seconds)
                        heuristic_time = min(
                            heuristic_budget,
                            max(self.config.time_limit_seconds - (_now() - start), 1e-3),
                        )
                        heuristic_obj, heuristic_solution, heuristic_backend = self.adapter.heuristic_incumbent(
                            node,
                            heuristic_time,
                            self.config.heuristic_repair_max_free_integer_vars,
                            incumbent_objective=incumbent_objective,
                            incumbent_solution=incumbent_solution,
                            diving_enabled=self.config.heuristic_diving_enabled,
                            diving_free_integer_vars=self.config.heuristic_diving_free_integer_vars,
                            diving_stage_factor=self.config.heuristic_diving_stage_factor,
                            diving_min_depth=self.config.heuristic_diving_min_depth,
                            diving_max_disagreement_vars=self.config.heuristic_diving_max_disagreement_vars,
                            lp_guided_enabled=self.config.heuristic_lp_guided_enabled,
                            lp_guided_free_integer_vars=self.config.heuristic_lp_guided_free_integer_vars,
                            lp_guided_stage_factor=self.config.heuristic_lp_guided_stage_factor,
                            lp_guided_min_depth=self.config.heuristic_lp_guided_min_depth,
                            lp_guided_min_disagreement_vars=self.config.heuristic_lp_guided_min_disagreement_vars,
                            lp_guided_integral_tolerance=self.config.heuristic_lp_guided_integral_tolerance,
                            rins_enabled=rins_allowed,
                            rins_free_integer_vars=self.config.heuristic_rins_free_integer_vars,
                            rins_min_depth=self.config.heuristic_rins_min_depth,
                            rins_min_agreement_vars=self.config.heuristic_rins_min_agreement_vars,
                            rins_integral_tolerance=self.config.heuristic_rins_integral_tolerance,
                            local_branching_enabled=self.config.heuristic_local_branching_enabled,
                            local_branching_radius=self.config.heuristic_local_branching_radius,
                            local_branching_max_binary_vars=self.config.heuristic_local_branching_max_binary_vars,
                            local_branching_max_fractional_vars=self.config.heuristic_local_branching_max_fractional_vars,
                            local_branching_max_gap=self.config.heuristic_local_branching_max_gap,
                        )
                        if heuristic_obj is not None and (incumbent_objective is None or heuristic_obj < incumbent_objective - 1e-9):
                            incumbent_objective = heuristic_obj
                            incumbent_solution = heuristic_solution
                            incumbents_found += 1
                            improvement_in_batch = True
                            steps_since_improvement = 0
                            active_nodes, pruned_nodes = self._prune_active_nodes(active_nodes, incumbent_objective)
                            global_pruned_nodes += pruned_nodes
                            if time_to_first_feasible is None:
                                time_to_first_feasible = _now() - start

                    children = self.adapter.branch(node, step, node_counter)
                    node_counter += len(children)
                    solved_children: list[NodeState] = []
                    for child in children:
                        remaining = max(self.config.time_limit_seconds - (_now() - start), 1e-3)
                        child = self.adapter.solve_node(child, time_limit_seconds=remaining)
                        solved_children.append(child)
                        nodes_evaluated += 1
                        if child.status in {"infeasible", "invalid_branch"}:
                            continue
                        if child.status == "integral":
                            if incumbent_objective is None or (child.lp_objective is not None and child.lp_objective < incumbent_objective - 1e-9):
                                incumbent_objective = child.lp_objective
                                incumbent_solution = child.integer_solution
                                incumbents_found += 1
                                improvement_in_batch = True
                                steps_since_improvement = 0
                                active_nodes, pruned_nodes = self._prune_active_nodes(active_nodes, incumbent_objective)
                                global_pruned_nodes += pruned_nodes
                                if time_to_first_feasible is None:
                                    time_to_first_feasible = _now() - start
                            continue
                        if incumbent_objective is not None and child.lp_objective is not None and child.lp_objective >= incumbent_objective - 1e-9:
                            continue
                        active_nodes.append(child)

                    self.adapter.update_pseudocosts(node, solved_children)

                    event = {
                        "timestamp_utc": utc_timestamp(),
                        "step": step,
                        "batch_index": batch_index,
                        "selected_node_id": node.node_id,
                        "resolved_policy": self.resolved_policy,
                        "selection_mode": diagnostics.selection_mode,
                        "root_fractional_count": self.root_fractional_count,
                        "root_integer_variable_count": self.root_integer_variable_count,
                        "root_binary_variable_count": self.root_binary_variable_count,
                        "root_fractional_ratio": self.root_fractional_ratio,
                        "root_solve_seconds": self.root_solve_seconds,
                        "root_lp_objective": self.root_lp_objective,
                        "root_fractionality_mean": self.root_fractionality_mean,
                        "root_fractionality_max": self.root_fractionality_max,
                        "root_integrality_gap_l1": self.root_integrality_gap_l1,
                        "root_binary_near_integral_ratio": self.root_binary_near_integral_ratio,
                        "root_binary_midpoint_ratio": self.root_binary_midpoint_ratio,
                        "root_lp_rounding_violated_row_ratio": self.root_lp_rounding_violated_row_ratio,
                        "root_lp_rounding_violation_mean": self.root_lp_rounding_violation_mean,
                        "root_lp_rounding_violation_max": self.root_lp_rounding_violation_max,
                        "root_lp_proxy_mode": self.root_lp_proxy_mode,
                        "root_lp_proxy_total_constraints": self.root_lp_proxy_total_constraints,
                        "root_lp_proxy_evaluated_constraints": self.root_lp_proxy_evaluated_constraints,
                        "root_lp_proxy_constraint_sample_limit": self.root_lp_proxy_constraint_sample_limit,
                        "root_lp_proxy_elapsed_seconds": self.root_lp_proxy_elapsed_seconds,
                        "root_primal_probe_found_feasible": self.root_primal_probe_found_feasible,
                        "root_primal_probe_relative_gap": self.root_primal_probe_relative_gap,
                        "root_primal_probe_objective": self.root_primal_probe_objective,
                        "root_primal_probe_backend": self.root_primal_probe_backend,
                        "root_primal_probe_elapsed_seconds": self.root_primal_probe_elapsed_seconds,
                        "status": node.status,
                        "node_lp_objective": node.lp_objective,
                        "reported_node_lp_objective": node.reported_lp_objective,
                        "depth": node.depth,
                        "node_fractional_count": len(node.fractional_variables),
                        "beta_before": beta,
                        "candidate_pool_size": diagnostics.candidate_pool_size,
                        "shortlist_size": diagnostics.shortlist_size,
                        "entropy": diagnostics.entropy,
                        "normalized_entropy": diagnostics.normalized_entropy,
                        "dispersion": diagnostics.dispersion,
                        "starvation_ratio": diagnostics.starvation_ratio,
                        "selected_probability": diagnostics.probabilities.get(node.node_id),
                        "heuristic_backend": heuristic_backend if self.config.heuristic_enabled else None,
                        "heuristic_diving_allowed": diving_allowed,
                        "heuristic_lp_guided_allowed": lp_guided_allowed,
                        "heuristic_rins_allowed": rins_allowed,
                        "heuristic_diving_disagreement_vars": diving_disagreement_vars,
                        "heuristic_rins_agreement_vars": rins_agreement_vars,
                        "global_pruned_after_incumbent_update": global_pruned_nodes,
                        "incumbent_before": incumbent_before,
                        "incumbent_after": incumbent_objective,
                        "children_created": len(children),
                    }
                    trace_handle.write(json.dumps(event, ensure_ascii=False) + "\n")

                if self.resolved_policy in {"boltzmann_adaptive", "phase_switch_hybrid", "ramped_estimate_hybrid"}:
                    beta, controller_metrics = self.controller.update(
                        current_beta=beta,
                        active_nodes=active_nodes,
                        current_step=step,
                        incumbent_objective=incumbent_objective,
                        incumbent_improved=improvement_in_batch,
                        steps_since_improvement=steps_since_improvement,
                    )
                    diagnostics.beta_after = beta
                    trace_handle.write(
                        json.dumps(
                            {
                                "timestamp_utc": utc_timestamp(),
                                "step": step,
                                "event": "beta_update",
                                "beta_after": beta,
                                "controller_metrics": controller_metrics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if improvement_in_batch:
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if not active_nodes:
                    status = "optimal" if incumbent_objective is not None else "infeasible"

            elapsed = _now() - start
            active_nodes, _ = self._prune_active_nodes(active_nodes, incumbent_objective)
            best_bound = max((node.lp_objective for node in active_nodes if node.lp_objective is not None), default=incumbent_objective)
            summary = RunSummary(
                run_id=run_id,
                status=status,
                instance_path=str(Path(self.config.instance_path).resolve()),
                policy=self.config.policy,
                resolved_policy=self.resolved_policy,
                policy_resolution_reason=self.policy_resolution_reason,
                root_fractional_count=self.root_fractional_count,
                root_integer_variable_count=self.root_integer_variable_count,
                root_binary_variable_count=self.root_binary_variable_count,
                root_fractional_ratio=self.root_fractional_ratio,
                root_solve_seconds=self.root_solve_seconds,
                root_lp_objective=self.root_lp_objective,
                root_fractionality_mean=self.root_fractionality_mean,
                root_fractionality_max=self.root_fractionality_max,
                root_integrality_gap_l1=self.root_integrality_gap_l1,
                root_binary_near_integral_ratio=self.root_binary_near_integral_ratio,
                root_binary_midpoint_ratio=self.root_binary_midpoint_ratio,
                root_lp_rounding_violated_row_ratio=self.root_lp_rounding_violated_row_ratio,
                root_lp_rounding_violation_mean=self.root_lp_rounding_violation_mean,
                root_lp_rounding_violation_max=self.root_lp_rounding_violation_max,
                root_lp_proxy_mode=self.root_lp_proxy_mode,
                root_lp_proxy_total_constraints=self.root_lp_proxy_total_constraints,
                root_lp_proxy_evaluated_constraints=self.root_lp_proxy_evaluated_constraints,
                root_lp_proxy_constraint_sample_limit=self.root_lp_proxy_constraint_sample_limit,
                root_lp_proxy_elapsed_seconds=self.root_lp_proxy_elapsed_seconds,
                root_primal_probe_found_feasible=self.root_primal_probe_found_feasible,
                root_primal_probe_relative_gap=self.root_primal_probe_relative_gap,
                root_primal_probe_objective=self.root_primal_probe_objective,
                root_primal_probe_backend=self.root_primal_probe_backend,
                root_primal_probe_elapsed_seconds=self.root_primal_probe_elapsed_seconds,
                selector_mode="adaptive" if self.resolved_policy in {"boltzmann_adaptive", "phase_switch_hybrid", "ramped_estimate_hybrid"} else "fixed",
                node_limit=self.config.node_limit,
                time_limit_seconds=self.config.time_limit_seconds,
                nodes_evaluated=nodes_evaluated,
                nodes_selected=nodes_selected,
                incumbents_found=incumbents_found,
                best_bound=best_bound,
                incumbent_objective=incumbent_objective,
                final_gap=relative_gap(incumbent_objective, best_bound),
                time_to_first_feasible=time_to_first_feasible,
                elapsed_seconds=elapsed,
                beta_initial=self.config.beta_config.initial_beta,
                beta_final=beta,
                summary_path=str(paths.summary_path),
                trace_path=str(paths.trace_path),
                manifest_path=str(paths.manifest_path),
            )

            manifest = {
                "run_id": run_id,
                "created_at_utc": utc_timestamp(),
                "config": self.config.to_dict(),
                "instance_sha256": instance_hash,
                "paths": paths.to_dict(),
                "summary": summary.to_dict(),
            }
            json_dump(paths.summary_path, summary.to_dict())
            json_dump(paths.manifest_path, manifest)
            return summary
        finally:
            trace_handle.close()
