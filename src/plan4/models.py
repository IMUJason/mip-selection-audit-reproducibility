from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class ScoreConfig:
    depth_weight: float = 0.05
    age_weight: float = 0.05
    estimate_weight: float = 0.10
    fractional_weight: float = 0.10
    uct_weight: float = 0.10
    epsilon_floor: float = 0.02
    best_bound_freq: int = 25
    depth_rescue_freq: int = 10
    depth_rescue_topk: int = 8
    primal_rescue_freq: int = 8
    primal_rescue_topk: int = 8
    plateau_rescue_freq: int = 6
    plateau_rescue_topk: int = 24
    plateau_rescue_min_pool: int = 12
    plateau_rescue_bound_band_ratio: float = 1e-6
    phase_switch_time_ratio: float = 0.35
    phase_switch_min_step: int = 8
    estimate_rescue_freq: int = 30
    estimate_rescue_late_freq: int = 6
    estimate_rescue_time_ratio: float = 0.75
    estimate_rescue_min_step: int = 8
    epsilon: float = 1e-9

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BetaConfig:
    initial_beta: float = 1.0
    min_beta: float = 0.1
    max_beta: float = 20.0
    improve_gain: float = 0.15
    stagnation_penalty: float = 0.20
    dispersion_gain: float = 0.10
    starvation_penalty: float = 0.15
    stagnation_window: int = 10
    starvation_age_ratio: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeState:
    node_id: str
    parent_id: str | None
    depth: int
    created_step: int
    fixings: dict[str, dict[str, float | None]] = field(default_factory=dict)
    lp_objective: float | None = None
    reported_lp_objective: float | None = None
    status: str = "pending"
    fractional_variables: dict[str, float] = field(default_factory=dict)
    integer_solution: dict[str, float] = field(default_factory=dict)
    branch_variable: str | None = None
    branch_value: float | None = None
    branch_direction: str | None = None
    visit_count: int = 0
    last_selected_step: int | None = None
    lp_proxy_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_integral(self) -> bool:
        return self.status == "integral" or not self.fractional_variables

    def age(self, current_step: int) -> int:
        return max(current_step - self.created_step, 0)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class SelectionDiagnostics:
    policy: str
    candidate_pool_size: int
    shortlist_size: int
    batch_size: int
    beta_before: float
    beta_after: float
    entropy: float
    normalized_entropy: float
    dispersion: float
    starvation_ratio: float
    selected_node_ids: list[str]
    probabilities: dict[str, float]
    scores: dict[str, float]
    selection_mode: str = "policy"

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class RunPaths:
    output_root: Path
    raw_results_dir: Path
    processed_results_dir: Path
    figures_dir: Path
    logs_dir: Path
    summary_path: Path
    trace_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass
class RunSummary:
    run_id: str
    status: str
    instance_path: str
    policy: str
    resolved_policy: str
    policy_resolution_reason: str | None
    root_fractional_count: int | None
    root_integer_variable_count: int | None
    root_binary_variable_count: int | None
    root_fractional_ratio: float | None
    root_solve_seconds: float | None
    root_lp_objective: float | None
    root_fractionality_mean: float | None
    root_fractionality_max: float | None
    root_integrality_gap_l1: float | None
    root_binary_near_integral_ratio: float | None
    root_binary_midpoint_ratio: float | None
    root_lp_rounding_violated_row_ratio: float | None
    root_lp_rounding_violation_mean: float | None
    root_lp_rounding_violation_max: float | None
    root_lp_proxy_mode: str | None
    root_lp_proxy_total_constraints: int | None
    root_lp_proxy_evaluated_constraints: int | None
    root_lp_proxy_constraint_sample_limit: int | None
    root_lp_proxy_elapsed_seconds: float | None
    root_primal_probe_found_feasible: int | None
    root_primal_probe_relative_gap: float | None
    root_primal_probe_objective: float | None
    root_primal_probe_backend: str | None
    root_primal_probe_elapsed_seconds: float | None
    selector_mode: str
    node_limit: int
    time_limit_seconds: float
    nodes_evaluated: int
    nodes_selected: int
    incumbents_found: int
    best_bound: float | None
    incumbent_objective: float | None
    final_gap: float | None
    time_to_first_feasible: float | None
    elapsed_seconds: float
    beta_initial: float
    beta_final: float
    summary_path: str
    trace_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))
