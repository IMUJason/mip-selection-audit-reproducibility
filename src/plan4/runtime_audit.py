from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .learning_selector import load_summary_rows


def infer_candidate_policies(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        policy = row.get("policy")
        if policy is not None and policy not in seen:
            seen[str(policy)] = None
    return list(seen.keys())


def audit_runtime_selector_from_files(
    *,
    runtime_summary_path: str | Path,
    baseline_summary_paths: list[str | Path],
    candidate_policies: list[str],
    time_weight: float = 0.05,
) -> dict[str, Any]:
    runtime_rows = load_summary_rows([runtime_summary_path])
    baseline_rows = load_summary_rows(baseline_summary_paths)
    payload = audit_runtime_selector_summary(
        runtime_rows,
        baseline_rows,
        candidate_policies=candidate_policies,
        time_weight=time_weight,
    )
    payload["runtime_summary_path"] = str(Path(runtime_summary_path).resolve())
    payload["baseline_summary_paths"] = [str(Path(path).resolve()) for path in baseline_summary_paths]
    payload["candidate_policies"] = list(candidate_policies)
    payload["time_weight"] = time_weight
    return payload


def audit_runtime_selector_summary(
    runtime_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    candidate_policies: list[str],
    time_weight: float = 0.05,
) -> dict[str, Any]:
    baseline_index: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        key = _summary_key(row)
        baseline_index.setdefault(key, []).append(row)

    audit_rows: list[dict[str, Any]] = []
    realized_total = 0.0

    for runtime_row in runtime_rows:
        key = _summary_key(runtime_row)
        if key not in baseline_index:
            raise ValueError(f"Missing baseline group for runtime row {key}")
        audit_row = _audit_single_runtime_row(
            runtime_row,
            baseline_index[key],
            candidate_policies=candidate_policies,
            time_weight=time_weight,
        )
        audit_rows.append(audit_row)
        realized_total += float(audit_row["realized_regret"])

    row_count = max(len(audit_rows), 1)
    return {
        "rows": audit_rows,
        "average_realized_regret": realized_total / row_count,
    }


def _audit_single_runtime_row(
    runtime_row: dict[str, Any],
    baseline_group: list[dict[str, Any]],
    *,
    candidate_policies: list[str],
    time_weight: float,
) -> dict[str, Any]:
    reference_rows: dict[str, dict[str, Any]] = {}
    for row in baseline_group:
        policy = row.get("policy")
        if policy in candidate_policies:
            if policy in reference_rows:
                raise ValueError(f"Duplicate baseline row for policy '{policy}' in group {_summary_key(row)}")
            reference_rows[str(policy)] = row

    missing_policies = [policy for policy in candidate_policies if policy not in reference_rows]
    if missing_policies:
        raise ValueError(f"Incomplete baseline group for {_summary_key(runtime_row)}: missing {missing_policies}")

    resolved_policy = str(runtime_row.get("resolved_policy") or runtime_row.get("policy"))
    if resolved_policy not in reference_rows:
        raise ValueError(
            f"Resolved policy '{resolved_policy}' not present in baseline group for {_summary_key(runtime_row)}"
        )

    ordered_reference_rows = [reference_rows[policy] for policy in candidate_policies]
    feasible_reference_rows = [row for row in ordered_reference_rows if _is_finite_number(row.get("final_gap"))]

    if feasible_reference_rows:
        metric_basis = "final_gap+time_to_first_feasible"
        gap_values = [float(row["final_gap"]) for row in feasible_reference_rows]
        min_gap = min(gap_values)
        max_gap = max(gap_values)
        gap_span = max(max_gap - min_gap, 1e-12)

        tff_values = [
            float(row["time_to_first_feasible"])
            for row in feasible_reference_rows
            if _is_finite_number(row.get("time_to_first_feasible"))
        ]
        min_tff = min(tff_values, default=0.0)
        max_tff = max(tff_values, default=min_tff)
        tff_span = max(max_tff - min_tff, 1e-12)

        if not _is_finite_number(runtime_row.get("final_gap")):
            realized_regret = 2.0
            gap_regret = None
            time_regret = None
        else:
            gap_regret = (float(runtime_row["final_gap"]) - min_gap) / gap_span
            runtime_tff = (
                float(runtime_row["time_to_first_feasible"])
                if _is_finite_number(runtime_row.get("time_to_first_feasible"))
                else max_tff
            )
            time_regret = (runtime_tff - min_tff) / tff_span if tff_span > 0.0 else 0.0
            realized_regret = gap_regret + time_weight * time_regret
    else:
        finite_bound_rows = [row for row in ordered_reference_rows if _is_finite_number(row.get("best_bound"))]
        if not finite_bound_rows:
            metric_basis = "no_signal"
            realized_regret = 0.0
            gap_regret = None
            time_regret = None
            reference_row = reference_rows[resolved_policy]
            return {
                "dataset_id": runtime_row["dataset_id"],
                "time_limit_seconds": float(runtime_row["time_limit_seconds"]),
                "resolved_policy": resolved_policy,
                "realized_regret": realized_regret,
                "metric_basis": metric_basis,
                "gap_regret": gap_regret,
                "time_regret": time_regret,
                "policy_resolution_reason": runtime_row.get("policy_resolution_reason"),
                "runtime_final_gap": runtime_row.get("final_gap"),
                "runtime_time_to_first_feasible": runtime_row.get("time_to_first_feasible"),
                "runtime_best_bound": runtime_row.get("best_bound"),
                "direct_reference_final_gap": reference_row.get("final_gap"),
                "direct_reference_time_to_first_feasible": reference_row.get("time_to_first_feasible"),
                "direct_reference_best_bound": reference_row.get("best_bound"),
                "root_primal_probe_found_feasible": runtime_row.get("root_primal_probe_found_feasible"),
                "root_primal_probe_relative_gap": runtime_row.get("root_primal_probe_relative_gap"),
                "root_lp_proxy_elapsed_seconds": runtime_row.get("root_lp_proxy_elapsed_seconds"),
                "root_primal_probe_elapsed_seconds": runtime_row.get("root_primal_probe_elapsed_seconds"),
                "root_primal_probe_time_limit_seconds": runtime_row.get("root_primal_probe_time_limit_seconds"),
                "summary_path": runtime_row.get("summary_path"),
                "trace_path": runtime_row.get("trace_path"),
                "manifest_path": runtime_row.get("manifest_path"),
            }
        metric_basis = "best_bound_only"
        bound_values = [float(row["best_bound"]) for row in finite_bound_rows]
        max_bound = max(bound_values)
        min_bound = min(bound_values)
        bound_span = max(max_bound - min_bound, 1e-12)
        if not _is_finite_number(runtime_row.get("best_bound")):
            realized_regret = 0.0
        else:
            realized_regret = (max_bound - float(runtime_row["best_bound"])) / bound_span
        gap_regret = None
        time_regret = None

    reference_row = reference_rows[resolved_policy]
    return {
        "dataset_id": runtime_row["dataset_id"],
        "time_limit_seconds": float(runtime_row["time_limit_seconds"]),
        "resolved_policy": resolved_policy,
        "realized_regret": realized_regret,
        "metric_basis": metric_basis,
        "gap_regret": gap_regret,
        "time_regret": time_regret,
        "policy_resolution_reason": runtime_row.get("policy_resolution_reason"),
        "runtime_final_gap": runtime_row.get("final_gap"),
        "runtime_time_to_first_feasible": runtime_row.get("time_to_first_feasible"),
        "runtime_best_bound": runtime_row.get("best_bound"),
        "direct_reference_final_gap": reference_row.get("final_gap"),
        "direct_reference_time_to_first_feasible": reference_row.get("time_to_first_feasible"),
        "direct_reference_best_bound": reference_row.get("best_bound"),
        "root_primal_probe_found_feasible": runtime_row.get("root_primal_probe_found_feasible"),
        "root_primal_probe_relative_gap": runtime_row.get("root_primal_probe_relative_gap"),
        "root_lp_proxy_elapsed_seconds": runtime_row.get("root_lp_proxy_elapsed_seconds"),
        "root_primal_probe_elapsed_seconds": runtime_row.get("root_primal_probe_elapsed_seconds"),
        "root_primal_probe_time_limit_seconds": runtime_row.get("root_primal_probe_time_limit_seconds"),
        "summary_path": runtime_row.get("summary_path"),
        "trace_path": runtime_row.get("trace_path"),
        "manifest_path": runtime_row.get("manifest_path"),
    }


def _summary_key(row: dict[str, Any]) -> tuple[str, float]:
    dataset_id = row.get("dataset_id")
    time_limit_seconds = row.get("time_limit_seconds")
    if dataset_id is None or time_limit_seconds is None:
        raise ValueError(f"Summary row is missing dataset/time identifiers: {row}")
    return str(dataset_id), float(time_limit_seconds)


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
