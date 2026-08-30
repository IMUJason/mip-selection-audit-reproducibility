from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from .manifest import DatasetEntry, find_dataset_entry, load_dataset_manifest
from .metrics import relative_gap
from .provenance import json_dump, sha256_file


DEFAULT_SELECTOR_FEATURES = [
    "time_limit_seconds",
    "root_fractional_count",
    "root_fractional_ratio",
    "root_integer_variable_count",
    "root_solve_seconds",
]

ROOT_PRIMAL_SELECTOR_FEATURES = [
    "root_primal_probe_found_feasible",
    "root_primal_probe_relative_gap",
    "root_primal_probe_elapsed_seconds",
]


def extract_root_features(
    instance_path: str | Path,
    *,
    dataset_id: str | None = None,
    group: str | None = None,
    instance_name: str | None = None,
    threads: int = 1,
    seed: int = 0,
    lp_proxy_mode: str = "full",
    lp_proxy_constraint_sample_limit: int | None = None,
    primal_probe_time_limit_seconds: float = 0.0,
    primal_probe_max_free_integer_vars: int = 12,
    primal_probe_plain_enabled: bool = True,
) -> dict[str, Any]:
    from .branch_and_bound import GurobiRelaxationAdapter
    from .models import NodeState

    source_path = Path(instance_path)
    adapter = GurobiRelaxationAdapter(source_path, threads=threads, seed=seed)
    start = time.monotonic()
    root = adapter.solve_node(
        NodeState(node_id="root", parent_id=None, depth=0, created_step=0),
        lp_proxy_mode=lp_proxy_mode,
        lp_proxy_constraint_sample_limit=lp_proxy_constraint_sample_limit,
    )
    solve_seconds = time.monotonic() - start

    standardized_objective = root.lp_objective
    fractional_variables = root.fractional_variables
    status = root.status

    root_primal_probe_found_feasible = 0
    root_primal_probe_relative_gap = 1.0
    root_primal_probe_objective = None
    root_primal_probe_backend = None
    root_primal_probe_elapsed_seconds = 0.0

    if root.status == "integral" and root.lp_objective is not None:
        root_primal_probe_found_feasible = 1
        root_primal_probe_relative_gap = 0.0
        root_primal_probe_objective = root.lp_objective
        root_primal_probe_backend = "root_integral"
    elif root.status == "fractional" and primal_probe_time_limit_seconds > 0.0:
        probe_start = time.monotonic()
        probe_objective, _, probe_backend = adapter.heuristic_incumbent(
            root,
            time_limit_seconds=primal_probe_time_limit_seconds,
            max_free_integer_vars=primal_probe_max_free_integer_vars,
            incumbent_objective=None,
            incumbent_solution=None,
            diving_enabled=False,
            lp_guided_enabled=False,
            rins_enabled=False,
            local_branching_enabled=False,
            plain_enabled=primal_probe_plain_enabled,
        )
        root_primal_probe_elapsed_seconds = time.monotonic() - probe_start
        root_primal_probe_objective = probe_objective
        root_primal_probe_backend = probe_backend
        if probe_objective is not None:
            root_primal_probe_found_feasible = 1
            root_primal_probe_relative_gap = relative_gap(probe_objective, root.lp_objective) or 0.0
    else:
        root_primal_probe_elapsed_seconds = 0.0

    fractionality_distances = [abs(value - round(value)) for value in fractional_variables.values()]
    fractional_count = len(fractional_variables)
    integer_count = len(adapter.integer_var_names)

    return {
        "dataset_id": dataset_id,
        "group": group,
        "instance_name": instance_name or source_path.name,
        "instance_path": str(source_path),
        "instance_sha256": sha256_file(source_path),
        "root_status": status,
        "root_integer_variable_count": integer_count,
        "root_binary_variable_count": len(adapter.binary_var_names),
        "root_fractional_count": fractional_count,
        "root_fractional_ratio": fractional_count / max(integer_count, 1),
        "root_solve_seconds": solve_seconds,
        "root_lp_objective": standardized_objective,
        "root_integrality_gap_l1": root.lp_proxy_metrics.get("root_integrality_gap_l1"),
        "root_binary_near_integral_ratio": root.lp_proxy_metrics.get("root_binary_near_integral_ratio"),
        "root_binary_midpoint_ratio": root.lp_proxy_metrics.get("root_binary_midpoint_ratio"),
        "root_lp_rounding_violated_row_ratio": root.lp_proxy_metrics.get("root_lp_rounding_violated_row_ratio"),
        "root_lp_rounding_violation_mean": root.lp_proxy_metrics.get("root_lp_rounding_violation_mean"),
        "root_lp_rounding_violation_max": root.lp_proxy_metrics.get("root_lp_rounding_violation_max"),
        "root_lp_proxy_mode": root.lp_proxy_metrics.get("root_lp_proxy_mode"),
        "root_lp_proxy_total_constraints": root.lp_proxy_metrics.get("root_lp_proxy_total_constraints"),
        "root_lp_proxy_evaluated_constraints": root.lp_proxy_metrics.get("root_lp_proxy_evaluated_constraints"),
        "root_lp_proxy_constraint_sample_limit": root.lp_proxy_metrics.get("root_lp_proxy_constraint_sample_limit"),
        "root_lp_proxy_elapsed_seconds": root.lp_proxy_metrics.get("root_lp_proxy_elapsed_seconds"),
        "root_primal_probe_found_feasible": root_primal_probe_found_feasible,
        "root_primal_probe_relative_gap": root_primal_probe_relative_gap,
        "root_primal_probe_objective": root_primal_probe_objective,
        "root_primal_probe_backend": root_primal_probe_backend,
        "root_primal_probe_elapsed_seconds": root_primal_probe_elapsed_seconds,
        "root_primal_probe_time_limit_seconds": primal_probe_time_limit_seconds,
        "root_primal_probe_max_free_integer_vars": primal_probe_max_free_integer_vars,
        "root_primal_probe_plain_enabled": 1 if primal_probe_plain_enabled else 0,
        "root_fractionality_mean": (
            sum(fractionality_distances) / len(fractionality_distances) if fractionality_distances else 0.0
        ),
        "root_fractionality_max": max(fractionality_distances, default=0.0),
        "threads": threads,
        "seed": seed,
    }


def extract_root_features_from_manifest(
    manifest_path: str | Path,
    *,
    instance_ids: list[str] | None = None,
    threads: int = 1,
    seed: int = 0,
    lp_proxy_mode: str = "full",
    lp_proxy_constraint_sample_limit: int | None = None,
    primal_probe_time_limit_seconds: float = 0.0,
    primal_probe_max_free_integer_vars: int = 12,
    primal_probe_plain_enabled: bool = True,
) -> list[dict[str, Any]]:
    entries = load_dataset_manifest(manifest_path)
    selected_entries: list[DatasetEntry]
    if instance_ids:
        selected_entries = [find_dataset_entry(entries, instance_id) for instance_id in instance_ids]
    else:
        selected_entries = entries

    rows = []
    for entry in selected_entries:
        rows.append(
            extract_root_features(
                entry.source_path,
                dataset_id=entry.data_id,
                group=entry.group,
                instance_name=entry.instance_name,
                threads=threads,
                seed=seed,
                lp_proxy_mode=lp_proxy_mode,
                lp_proxy_constraint_sample_limit=lp_proxy_constraint_sample_limit,
                primal_probe_time_limit_seconds=primal_probe_time_limit_seconds,
                primal_probe_max_free_integer_vars=primal_probe_max_free_integer_vars,
                primal_probe_plain_enabled=primal_probe_plain_enabled,
            )
        )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {file_path}")
        return payload
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported file format: {file_path}")


def load_summary_rows(summary_paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list in {summary_path}")
        rows.extend(payload)
    return rows


def _coerce_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compute_policy_regrets(
    rows: list[dict[str, Any]],
    candidate_policies: list[str],
    *,
    time_weight: float = 0.05,
) -> tuple[dict[str, float], str]:
    candidate_rows = [row for row in rows if row.get("policy") in candidate_policies]
    if len(candidate_rows) != len(candidate_policies):
        seen = sorted({row.get("policy") for row in candidate_rows})
        raise ValueError(f"Incomplete policy group. Expected {candidate_policies}, got {seen}")

    feasible_rows = [row for row in candidate_rows if _is_finite_number(row.get("final_gap"))]
    if feasible_rows:
        gap_values = [_coerce_float(row["final_gap"]) for row in feasible_rows]
        min_gap = min(gap_values)
        max_gap = max(gap_values)
        gap_span = max(max_gap - min_gap, 1e-12)

        tff_values = [
            _coerce_float(row["time_to_first_feasible"])
            for row in feasible_rows
            if _is_finite_number(row.get("time_to_first_feasible"))
        ]
        min_tff = min(tff_values, default=0.0)
        max_tff = max(tff_values, default=min_tff)
        tff_span = max(max_tff - min_tff, 1e-12)

        regrets: dict[str, float] = {}
        for row in candidate_rows:
            policy = row["policy"]
            if not _is_finite_number(row.get("final_gap")):
                regrets[policy] = 2.0
                continue
            gap_regret = (_coerce_float(row["final_gap"]) - min_gap) / gap_span
            tff_value = (
                _coerce_float(row["time_to_first_feasible"])
                if _is_finite_number(row.get("time_to_first_feasible"))
                else max_tff
            )
            time_regret = (tff_value - min_tff) / tff_span if tff_span > 0.0 else 0.0
            regrets[policy] = gap_regret + time_weight * time_regret
        return regrets, "final_gap+time_to_first_feasible"

    finite_bound_rows = [row for row in candidate_rows if _is_finite_number(row.get("best_bound"))]
    if not finite_bound_rows:
        return {row["policy"]: 0.0 for row in candidate_rows}, "no_signal"

    bound_values = [_coerce_float(row["best_bound"]) for row in finite_bound_rows]
    max_bound = max(bound_values)
    min_bound = min(bound_values)
    bound_span = max(max_bound - min_bound, 1e-12)
    regrets = {}
    for row in candidate_rows:
        if not _is_finite_number(row.get("best_bound")):
            regrets[row["policy"]] = 0.0
            continue
        regrets[row["policy"]] = (max_bound - _coerce_float(row["best_bound"])) / bound_span
    return regrets, "best_bound_only"


def build_selector_dataset(
    feature_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
    *,
    candidate_policies: list[str],
    feature_names: list[str] | None = None,
    time_weight: float = 0.05,
) -> list[dict[str, Any]]:
    feature_names = feature_names or list(DEFAULT_SELECTOR_FEATURES)
    feature_map: dict[str, dict[str, Any]] = {}
    for row in feature_rows:
        dataset_id = row.get("dataset_id")
        if not dataset_id:
            raise ValueError("Feature row missing dataset_id.")
        if dataset_id in feature_map:
            raise ValueError(f"Duplicate feature row for dataset_id={dataset_id}")
        feature_map[dataset_id] = row

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    seen_triplets: set[tuple[str, float, str]] = set()
    for row in performance_rows:
        dataset_id = row.get("dataset_id")
        policy = row.get("policy")
        budget = row.get("time_limit_seconds")
        if dataset_id is None or policy is None or budget is None:
            continue
        triplet = (str(dataset_id), float(budget), str(policy))
        if triplet in seen_triplets:
            raise ValueError(f"Duplicate performance row for {triplet}")
        seen_triplets.add(triplet)
        grouped.setdefault((str(dataset_id), float(budget)), []).append(row)

    dataset_rows: list[dict[str, Any]] = []
    for (dataset_id, budget), rows in sorted(grouped.items()):
        if dataset_id not in feature_map:
            raise KeyError(f"Missing root features for dataset_id={dataset_id}")
        regrets, metric_basis = compute_policy_regrets(rows, candidate_policies, time_weight=time_weight)
        feature_row = feature_map[dataset_id]
        feature_payload = {
            name: (float(budget) if name == "time_limit_seconds" else float(feature_row[name]))
            for name in feature_names
        }
        best_regret = min(regrets.values())
        best_policies = sorted([policy for policy, regret in regrets.items() if abs(regret - best_regret) <= 1e-9])
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "group": feature_row.get("group"),
                "instance_name": feature_row.get("instance_name"),
                "instance_path": feature_row.get("instance_path"),
                "time_limit_seconds": float(budget),
                "feature_values": feature_payload,
                "policy_regrets": regrets,
                "best_policies": best_policies,
                "metric_basis": metric_basis,
                "source_rows": [
                    {
                        "policy": row["policy"],
                        "final_gap": row.get("final_gap"),
                        "best_bound": row.get("best_bound"),
                        "incumbent_objective": row.get("incumbent_objective"),
                        "time_to_first_feasible": row.get("time_to_first_feasible"),
                        "summary_path": row.get("summary_path"),
                        "manifest_path": row.get("manifest_path"),
                        "trace_path": row.get("trace_path"),
                    }
                    for row in sorted(rows, key=lambda item: item["policy"])
                    if row.get("policy") in candidate_policies
                ],
            }
        )
    return dataset_rows


def _leaf_policy_costs(rows: list[dict[str, Any]], candidate_policies: list[str]) -> dict[str, float]:
    return {
        policy: sum(float(row["policy_regrets"][policy]) for row in rows)
        for policy in candidate_policies
    }


def _make_leaf(rows: list[dict[str, Any]], candidate_policies: list[str]) -> tuple[dict[str, Any], float]:
    policy_costs = _leaf_policy_costs(rows, candidate_policies)
    selected_policy = min(candidate_policies, key=lambda policy: (policy_costs[policy], policy))
    total_cost = policy_costs[selected_policy]
    return (
        {
            "kind": "leaf",
            "policy": selected_policy,
            "rows": len(rows),
            "avg_regret": total_cost / max(len(rows), 1),
            "policy_costs": policy_costs,
            "best_policies": sorted({policy for row in rows for policy in row["best_policies"]}),
            "dataset_ids": [row["dataset_id"] for row in rows],
        },
        total_cost,
    )


def _candidate_thresholds(rows: list[dict[str, Any]], feature_name: str) -> list[float]:
    values = sorted({float(row["feature_values"][feature_name]) for row in rows})
    if len(values) <= 1:
        return []
    return [(left + right) / 2.0 for left, right in zip(values, values[1:])]


def fit_regret_tree(
    rows: list[dict[str, Any]],
    *,
    candidate_policies: list[str],
    feature_names: list[str] | None = None,
    max_depth: int = 2,
    min_samples_leaf: int = 1,
    complexity_penalty: float = 0.01,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot fit selector without training rows.")
    feature_names = feature_names or list(DEFAULT_SELECTOR_FEATURES)

    def _fit(subset: list[dict[str, Any]], depth: int) -> tuple[dict[str, Any], float]:
        leaf, leaf_cost = _make_leaf(subset, candidate_policies)
        if depth >= max_depth or len(subset) < 2 * min_samples_leaf:
            return leaf, leaf_cost

        best_node = leaf
        best_cost = leaf_cost
        for feature_name in feature_names:
            for threshold in _candidate_thresholds(subset, feature_name):
                left_rows = [row for row in subset if float(row["feature_values"][feature_name]) < threshold]
                right_rows = [row for row in subset if float(row["feature_values"][feature_name]) >= threshold]
                if len(left_rows) < min_samples_leaf or len(right_rows) < min_samples_leaf:
                    continue
                left_node, left_cost = _fit(left_rows, depth + 1)
                right_node, right_cost = _fit(right_rows, depth + 1)
                split_cost = left_cost + right_cost + complexity_penalty
                if split_cost + 1e-12 < best_cost:
                    best_cost = split_cost
                    best_node = {
                        "kind": "split",
                        "feature": feature_name,
                        "threshold": threshold,
                        "rows": len(subset),
                        "left": left_node,
                        "right": right_node,
                    }
        return best_node, best_cost

    tree, objective_value = _fit(rows, depth=0)
    evaluation = evaluate_selector_model(
        {
            "model_type": "plan4_regret_tree_v1",
            "candidate_policies": candidate_policies,
            "feature_names": feature_names,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "complexity_penalty": complexity_penalty,
            "tree": tree,
        },
        rows,
    )
    return {
        "model_type": "plan4_regret_tree_v1",
        "candidate_policies": candidate_policies,
        "feature_names": feature_names,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "complexity_penalty": complexity_penalty,
        "objective_value": objective_value,
        "tree": tree,
        "training_summary": evaluation,
    }


def predict_selector_policy(model: dict[str, Any], feature_values: dict[str, float]) -> tuple[str, str]:
    node = model["tree"]
    path: list[str] = []
    while node["kind"] == "split":
        feature_name = node["feature"]
        if feature_name not in feature_values:
            raise KeyError(f"Missing feature '{feature_name}' for selector prediction.")
        value = float(feature_values[feature_name])
        threshold = float(node["threshold"])
        if value < threshold:
            path.append(f"{feature_name}={value:.6g}<{threshold:.6g}")
            node = node["left"]
        else:
            path.append(f"{feature_name}={value:.6g}>={threshold:.6g}")
            node = node["right"]
    path.append(f"policy={node['policy']}")
    return str(node["policy"]), " -> ".join(path)


def load_selector_model(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("model_type") != "plan4_regret_tree_v1":
        raise ValueError(f"Unsupported selector model type: {payload.get('model_type')}")
    return payload


def evaluate_selector_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_policies = list(model["candidate_policies"])
    assignments = []
    selector_regret_total = 0.0
    baseline_totals = {policy: 0.0 for policy in candidate_policies}

    for row in rows:
        predicted_policy, reason = predict_selector_policy(model, row["feature_values"])
        predicted_regret = float(row["policy_regrets"][predicted_policy])
        selector_regret_total += predicted_regret
        for policy in candidate_policies:
            baseline_totals[policy] += float(row["policy_regrets"][policy])
        assignments.append(
            {
                "dataset_id": row["dataset_id"],
                "time_limit_seconds": row["time_limit_seconds"],
                "predicted_policy": predicted_policy,
                "predicted_regret": predicted_regret,
                "best_policies": row["best_policies"],
                "metric_basis": row["metric_basis"],
                "reason": reason,
                "feature_values": row["feature_values"],
                "policy_regrets": row["policy_regrets"],
            }
        )

    row_count = max(len(rows), 1)
    baseline_average_regrets = {
        policy: total / row_count
        for policy, total in baseline_totals.items()
    }
    selector_average_regret = selector_regret_total / row_count
    split_count = count_tree_splits(model["tree"])

    return {
        "rows": len(rows),
        "split_count": split_count,
        "selector_average_regret": selector_average_regret,
        "selector_total_regret": selector_regret_total,
        "baseline_average_regrets": baseline_average_regrets,
        "baseline_total_regrets": baseline_totals,
        "assignments": assignments,
    }


def count_tree_splits(node: dict[str, Any]) -> int:
    if node["kind"] != "split":
        return 0
    return 1 + count_tree_splits(node["left"]) + count_tree_splits(node["right"])


def collect_tree_split_features(node: dict[str, Any]) -> set[str]:
    if node["kind"] != "split":
        return set()
    return {
        str(node["feature"]),
        *collect_tree_split_features(node["left"]),
        *collect_tree_split_features(node["right"]),
    }


def required_features_for_partial_tree(
    node: dict[str, Any],
    known_feature_values: dict[str, float],
) -> set[str]:
    if node["kind"] != "split":
        return set()
    feature_name = str(node["feature"])
    if feature_name not in known_feature_values:
        return collect_tree_split_features(node)
    value = float(known_feature_values[feature_name])
    threshold = float(node["threshold"])
    if value < threshold:
        return required_features_for_partial_tree(node["left"], known_feature_values)
    return required_features_for_partial_tree(node["right"], known_feature_values)


def train_selector_from_files(
    *,
    feature_path: str | Path,
    summary_paths: list[str | Path],
    candidate_policies: list[str],
    feature_names: list[str] | None = None,
    time_weight: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int = 1,
    complexity_penalty: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_rows = load_rows(feature_path)
    performance_rows = load_summary_rows(summary_paths)
    dataset_rows = build_selector_dataset(
        feature_rows,
        performance_rows,
        candidate_policies=candidate_policies,
        feature_names=feature_names,
        time_weight=time_weight,
    )
    model = fit_regret_tree(
        dataset_rows,
        candidate_policies=candidate_policies,
        feature_names=feature_names,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        complexity_penalty=complexity_penalty,
    )
    return dataset_rows, model


def export_rows_json_and_csv(
    rows: list[dict[str, Any]],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_dump(json_path, rows)
    write_rows_csv(csv_path, rows)
