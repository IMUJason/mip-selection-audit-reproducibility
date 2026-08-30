from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .branch_and_bound import BranchAndBoundConfig, BranchAndBoundEngine
from .learning_selector import (
    DEFAULT_SELECTOR_FEATURES,
    build_selector_dataset,
    evaluate_selector_model,
    export_rows_json_and_csv,
    extract_root_features_from_manifest,
    load_rows,
    load_summary_rows,
    load_selector_model,
    train_selector_from_files,
)
from .manifest import find_dataset_entry, load_dataset_manifest
from .models import BetaConfig, ScoreConfig
from .provenance import ensure_run_paths, json_dump, verify_dataset_manifest, verify_literature_registry
from .runtime_audit import audit_runtime_selector_from_files, infer_candidate_policies


def _print_verification(result: dict) -> None:
    print(f"valid={result['valid']}")
    for item in result["items"]:
        print(item)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _resolve_optional_path(base_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _build_branch_and_bound_config(
    instance_path: str,
    run_tag: str,
    output_root: str | None,
    common: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
) -> BranchAndBoundConfig:
    common = common or {}
    override = override or {}
    defaults = BranchAndBoundConfig(instance_path=instance_path, run_tag=run_tag, output_root=output_root)
    score_config = ScoreConfig(
        **_merge_dicts(
            ScoreConfig().to_dict(),
            _merge_dicts(common.get("score_config", {}), override.get("score_config", {})),
        )
    )
    beta_config = BetaConfig(
        **_merge_dicts(
            BetaConfig().to_dict(),
            _merge_dicts(common.get("beta_config", {}), override.get("beta_config", {})),
        )
    )
    return BranchAndBoundConfig(
        instance_path=instance_path,
        policy=override.get("name", common.get("policy", defaults.policy)),
        portfolio_short_policy=override.get(
            "portfolio_short_policy",
            common.get("portfolio_short_policy", defaults.portfolio_short_policy),
        ),
        portfolio_long_policy=override.get(
            "portfolio_long_policy",
            common.get("portfolio_long_policy", defaults.portfolio_long_policy),
        ),
        portfolio_time_threshold_seconds=override.get(
            "portfolio_time_threshold_seconds",
            common.get("portfolio_time_threshold_seconds", defaults.portfolio_time_threshold_seconds),
        ),
        instance_aware_short_small_policy=override.get(
            "instance_aware_short_small_policy",
            common.get("instance_aware_short_small_policy", defaults.instance_aware_short_small_policy),
        ),
        instance_aware_short_large_policy=override.get(
            "instance_aware_short_large_policy",
            common.get("instance_aware_short_large_policy", defaults.instance_aware_short_large_policy),
        ),
        instance_aware_long_policy=override.get(
            "instance_aware_long_policy",
            common.get("instance_aware_long_policy", defaults.instance_aware_long_policy),
        ),
        instance_aware_time_threshold_seconds=override.get(
            "instance_aware_time_threshold_seconds",
            common.get("instance_aware_time_threshold_seconds", defaults.instance_aware_time_threshold_seconds),
        ),
        instance_aware_root_fractional_threshold=override.get(
            "instance_aware_root_fractional_threshold",
            common.get("instance_aware_root_fractional_threshold", defaults.instance_aware_root_fractional_threshold),
        ),
        multi_feature_short_low_complexity_policy=override.get(
            "multi_feature_short_low_complexity_policy",
            common.get("multi_feature_short_low_complexity_policy", defaults.multi_feature_short_low_complexity_policy),
        ),
        multi_feature_short_high_complexity_policy=override.get(
            "multi_feature_short_high_complexity_policy",
            common.get("multi_feature_short_high_complexity_policy", defaults.multi_feature_short_high_complexity_policy),
        ),
        multi_feature_long_policy=override.get(
            "multi_feature_long_policy",
            common.get("multi_feature_long_policy", defaults.multi_feature_long_policy),
        ),
        multi_feature_time_threshold_seconds=override.get(
            "multi_feature_time_threshold_seconds",
            common.get("multi_feature_time_threshold_seconds", defaults.multi_feature_time_threshold_seconds),
        ),
        multi_feature_root_fractional_threshold=override.get(
            "multi_feature_root_fractional_threshold",
            common.get("multi_feature_root_fractional_threshold", defaults.multi_feature_root_fractional_threshold),
        ),
        multi_feature_root_fractional_ratio_threshold=override.get(
            "multi_feature_root_fractional_ratio_threshold",
            common.get("multi_feature_root_fractional_ratio_threshold", defaults.multi_feature_root_fractional_ratio_threshold),
        ),
        multi_feature_integer_variable_threshold=override.get(
            "multi_feature_integer_variable_threshold",
            common.get("multi_feature_integer_variable_threshold", defaults.multi_feature_integer_variable_threshold),
        ),
        multi_feature_root_solve_seconds_threshold=override.get(
            "multi_feature_root_solve_seconds_threshold",
            common.get("multi_feature_root_solve_seconds_threshold", defaults.multi_feature_root_solve_seconds_threshold),
        ),
        multi_feature_complexity_votes_required=override.get(
            "multi_feature_complexity_votes_required",
            common.get("multi_feature_complexity_votes_required", defaults.multi_feature_complexity_votes_required),
        ),
        learned_policy_model_path=override.get(
            "learned_policy_model_path",
            common.get("learned_policy_model_path", defaults.learned_policy_model_path),
        ),
        learned_policy_default=override.get(
            "learned_policy_default",
            common.get("learned_policy_default", defaults.learned_policy_default),
        ),
        root_lp_proxy_mode=override.get(
            "root_lp_proxy_mode",
            common.get("root_lp_proxy_mode", defaults.root_lp_proxy_mode),
        ),
        root_lp_proxy_constraint_sample_limit=override.get(
            "root_lp_proxy_constraint_sample_limit",
            common.get("root_lp_proxy_constraint_sample_limit", defaults.root_lp_proxy_constraint_sample_limit),
        ),
        root_primal_probe_time_limit_seconds=override.get(
            "root_primal_probe_time_limit_seconds",
            common.get("root_primal_probe_time_limit_seconds", defaults.root_primal_probe_time_limit_seconds),
        ),
        root_primal_probe_max_free_integer_vars=override.get(
            "root_primal_probe_max_free_integer_vars",
            common.get("root_primal_probe_max_free_integer_vars", defaults.root_primal_probe_max_free_integer_vars),
        ),
        root_primal_probe_plain_enabled=override.get(
            "root_primal_probe_plain_enabled",
            common.get("root_primal_probe_plain_enabled", defaults.root_primal_probe_plain_enabled),
        ),
        batch_size=override.get("batch_size", common.get("batch_size", defaults.batch_size)),
        shortlist_size=override.get("shortlist_size", common.get("shortlist_size", defaults.shortlist_size)),
        node_limit=override.get("node_limit", common.get("node_limit", defaults.node_limit)),
        time_limit_seconds=override.get("time_limit_seconds", common.get("time_limit_seconds", defaults.time_limit_seconds)),
        random_seed=override.get("random_seed", common.get("random_seed", defaults.random_seed)),
        threads=override.get("threads", common.get("threads", defaults.threads)),
        solver_backend=override.get("solver_backend", common.get("solver_backend", defaults.solver_backend)),
        run_tag=run_tag,
        output_root=output_root,
        heuristic_enabled=override.get("heuristic_enabled", common.get("heuristic_enabled", defaults.heuristic_enabled)),
        heuristic_time_limit_seconds=override.get(
            "heuristic_time_limit_seconds",
            common.get("heuristic_time_limit_seconds", defaults.heuristic_time_limit_seconds),
        ),
        heuristic_every_n_selected=override.get(
            "heuristic_every_n_selected",
            common.get("heuristic_every_n_selected", defaults.heuristic_every_n_selected),
        ),
        heuristic_repair_max_free_integer_vars=override.get(
            "heuristic_repair_max_free_integer_vars",
            common.get("heuristic_repair_max_free_integer_vars", defaults.heuristic_repair_max_free_integer_vars),
        ),
        heuristic_diving_enabled=override.get(
            "heuristic_diving_enabled",
            common.get("heuristic_diving_enabled", defaults.heuristic_diving_enabled),
        ),
        heuristic_diving_free_integer_vars=override.get(
            "heuristic_diving_free_integer_vars",
            common.get("heuristic_diving_free_integer_vars", defaults.heuristic_diving_free_integer_vars),
        ),
        heuristic_diving_stage_factor=override.get(
            "heuristic_diving_stage_factor",
            common.get("heuristic_diving_stage_factor", defaults.heuristic_diving_stage_factor),
        ),
        heuristic_diving_min_depth=override.get(
            "heuristic_diving_min_depth",
            common.get("heuristic_diving_min_depth", defaults.heuristic_diving_min_depth),
        ),
        heuristic_diving_max_disagreement_vars=override.get(
            "heuristic_diving_max_disagreement_vars",
            common.get("heuristic_diving_max_disagreement_vars", defaults.heuristic_diving_max_disagreement_vars),
        ),
        heuristic_lp_guided_enabled=override.get(
            "heuristic_lp_guided_enabled",
            common.get("heuristic_lp_guided_enabled", defaults.heuristic_lp_guided_enabled),
        ),
        heuristic_lp_guided_free_integer_vars=override.get(
            "heuristic_lp_guided_free_integer_vars",
            common.get("heuristic_lp_guided_free_integer_vars", defaults.heuristic_lp_guided_free_integer_vars),
        ),
        heuristic_lp_guided_stage_factor=override.get(
            "heuristic_lp_guided_stage_factor",
            common.get("heuristic_lp_guided_stage_factor", defaults.heuristic_lp_guided_stage_factor),
        ),
        heuristic_lp_guided_min_depth=override.get(
            "heuristic_lp_guided_min_depth",
            common.get("heuristic_lp_guided_min_depth", defaults.heuristic_lp_guided_min_depth),
        ),
        heuristic_lp_guided_min_disagreement_vars=override.get(
            "heuristic_lp_guided_min_disagreement_vars",
            common.get("heuristic_lp_guided_min_disagreement_vars", defaults.heuristic_lp_guided_min_disagreement_vars),
        ),
        heuristic_lp_guided_integral_tolerance=override.get(
            "heuristic_lp_guided_integral_tolerance",
            common.get("heuristic_lp_guided_integral_tolerance", defaults.heuristic_lp_guided_integral_tolerance),
        ),
        heuristic_rins_enabled=override.get(
            "heuristic_rins_enabled",
            common.get("heuristic_rins_enabled", defaults.heuristic_rins_enabled),
        ),
        heuristic_rins_time_limit_seconds=override.get(
            "heuristic_rins_time_limit_seconds",
            common.get("heuristic_rins_time_limit_seconds", defaults.heuristic_rins_time_limit_seconds),
        ),
        heuristic_rins_every_n_selected=override.get(
            "heuristic_rins_every_n_selected",
            common.get("heuristic_rins_every_n_selected", defaults.heuristic_rins_every_n_selected),
        ),
        heuristic_rins_free_integer_vars=override.get(
            "heuristic_rins_free_integer_vars",
            common.get("heuristic_rins_free_integer_vars", defaults.heuristic_rins_free_integer_vars),
        ),
        heuristic_rins_min_depth=override.get(
            "heuristic_rins_min_depth",
            common.get("heuristic_rins_min_depth", defaults.heuristic_rins_min_depth),
        ),
        heuristic_rins_min_agreement_vars=override.get(
            "heuristic_rins_min_agreement_vars",
            common.get("heuristic_rins_min_agreement_vars", defaults.heuristic_rins_min_agreement_vars),
        ),
        heuristic_rins_integral_tolerance=override.get(
            "heuristic_rins_integral_tolerance",
            common.get("heuristic_rins_integral_tolerance", defaults.heuristic_rins_integral_tolerance),
        ),
        heuristic_local_branching_enabled=override.get(
            "heuristic_local_branching_enabled",
            common.get("heuristic_local_branching_enabled", defaults.heuristic_local_branching_enabled),
        ),
        heuristic_local_branching_radius=override.get(
            "heuristic_local_branching_radius",
            common.get("heuristic_local_branching_radius", defaults.heuristic_local_branching_radius),
        ),
        heuristic_local_branching_max_binary_vars=override.get(
            "heuristic_local_branching_max_binary_vars",
            common.get("heuristic_local_branching_max_binary_vars", defaults.heuristic_local_branching_max_binary_vars),
        ),
        heuristic_local_branching_max_fractional_vars=override.get(
            "heuristic_local_branching_max_fractional_vars",
            common.get("heuristic_local_branching_max_fractional_vars", defaults.heuristic_local_branching_max_fractional_vars),
        ),
        heuristic_local_branching_max_gap=override.get(
            "heuristic_local_branching_max_gap",
            common.get("heuristic_local_branching_max_gap", defaults.heuristic_local_branching_max_gap),
        ),
        score_config=score_config,
        beta_config=beta_config,
    )


def run_single(args: argparse.Namespace) -> None:
    override: dict[str, Any] = {
        "name": args.policy,
        "batch_size": args.batch_size,
        "shortlist_size": args.shortlist_size,
        "node_limit": args.node_limit,
        "time_limit_seconds": args.time_limit,
        "random_seed": args.seed,
        "threads": args.threads,
        "solver_backend": args.solver_backend,
    }
    if args.portfolio_short_policy is not None:
        override["portfolio_short_policy"] = args.portfolio_short_policy
    if args.portfolio_long_policy is not None:
        override["portfolio_long_policy"] = args.portfolio_long_policy
    if args.portfolio_time_threshold_seconds is not None:
        override["portfolio_time_threshold_seconds"] = args.portfolio_time_threshold_seconds
    if args.instance_aware_short_small_policy is not None:
        override["instance_aware_short_small_policy"] = args.instance_aware_short_small_policy
    if args.instance_aware_short_large_policy is not None:
        override["instance_aware_short_large_policy"] = args.instance_aware_short_large_policy
    if args.instance_aware_long_policy is not None:
        override["instance_aware_long_policy"] = args.instance_aware_long_policy
    if args.instance_aware_time_threshold_seconds is not None:
        override["instance_aware_time_threshold_seconds"] = args.instance_aware_time_threshold_seconds
    if args.instance_aware_root_fractional_threshold is not None:
        override["instance_aware_root_fractional_threshold"] = args.instance_aware_root_fractional_threshold
    if args.multi_feature_short_low_complexity_policy is not None:
        override["multi_feature_short_low_complexity_policy"] = args.multi_feature_short_low_complexity_policy
    if args.multi_feature_short_high_complexity_policy is not None:
        override["multi_feature_short_high_complexity_policy"] = args.multi_feature_short_high_complexity_policy
    if args.multi_feature_long_policy is not None:
        override["multi_feature_long_policy"] = args.multi_feature_long_policy
    if args.multi_feature_time_threshold_seconds is not None:
        override["multi_feature_time_threshold_seconds"] = args.multi_feature_time_threshold_seconds
    if args.multi_feature_root_fractional_threshold is not None:
        override["multi_feature_root_fractional_threshold"] = args.multi_feature_root_fractional_threshold
    if args.multi_feature_root_fractional_ratio_threshold is not None:
        override["multi_feature_root_fractional_ratio_threshold"] = args.multi_feature_root_fractional_ratio_threshold
    if args.multi_feature_integer_variable_threshold is not None:
        override["multi_feature_integer_variable_threshold"] = args.multi_feature_integer_variable_threshold
    if args.multi_feature_root_solve_seconds_threshold is not None:
        override["multi_feature_root_solve_seconds_threshold"] = args.multi_feature_root_solve_seconds_threshold
    if args.multi_feature_complexity_votes_required is not None:
        override["multi_feature_complexity_votes_required"] = args.multi_feature_complexity_votes_required
    if args.learned_policy_model_path is not None:
        override["learned_policy_model_path"] = args.learned_policy_model_path
    if args.learned_policy_default is not None:
        override["learned_policy_default"] = args.learned_policy_default
    if args.root_lp_proxy_mode is not None:
        override["root_lp_proxy_mode"] = args.root_lp_proxy_mode
    if args.root_lp_proxy_constraint_sample_limit is not None:
        override["root_lp_proxy_constraint_sample_limit"] = args.root_lp_proxy_constraint_sample_limit
    if args.root_primal_probe_time_limit_seconds is not None:
        override["root_primal_probe_time_limit_seconds"] = args.root_primal_probe_time_limit_seconds
    if args.root_primal_probe_max_free_integer_vars is not None:
        override["root_primal_probe_max_free_integer_vars"] = args.root_primal_probe_max_free_integer_vars
    if args.root_primal_probe_plain_enabled is not None:
        override["root_primal_probe_plain_enabled"] = args.root_primal_probe_plain_enabled
    if args.heuristic_enabled is not None:
        override["heuristic_enabled"] = args.heuristic_enabled
    if args.heuristic_time_limit is not None:
        override["heuristic_time_limit_seconds"] = args.heuristic_time_limit
    if args.heuristic_every_n_selected is not None:
        override["heuristic_every_n_selected"] = args.heuristic_every_n_selected
    if args.heuristic_repair_max_free_integer_vars is not None:
        override["heuristic_repair_max_free_integer_vars"] = args.heuristic_repair_max_free_integer_vars
    if args.heuristic_diving_enabled is not None:
        override["heuristic_diving_enabled"] = args.heuristic_diving_enabled
    if args.heuristic_diving_free_integer_vars is not None:
        override["heuristic_diving_free_integer_vars"] = args.heuristic_diving_free_integer_vars
    if args.heuristic_diving_stage_factor is not None:
        override["heuristic_diving_stage_factor"] = args.heuristic_diving_stage_factor
    if args.heuristic_diving_min_depth is not None:
        override["heuristic_diving_min_depth"] = args.heuristic_diving_min_depth
    if args.heuristic_diving_max_disagreement_vars is not None:
        override["heuristic_diving_max_disagreement_vars"] = args.heuristic_diving_max_disagreement_vars
    if args.heuristic_lp_guided_enabled is not None:
        override["heuristic_lp_guided_enabled"] = args.heuristic_lp_guided_enabled
    if args.heuristic_lp_guided_free_integer_vars is not None:
        override["heuristic_lp_guided_free_integer_vars"] = args.heuristic_lp_guided_free_integer_vars
    if args.heuristic_lp_guided_stage_factor is not None:
        override["heuristic_lp_guided_stage_factor"] = args.heuristic_lp_guided_stage_factor
    if args.heuristic_lp_guided_min_depth is not None:
        override["heuristic_lp_guided_min_depth"] = args.heuristic_lp_guided_min_depth
    if args.heuristic_lp_guided_min_disagreement_vars is not None:
        override["heuristic_lp_guided_min_disagreement_vars"] = args.heuristic_lp_guided_min_disagreement_vars
    if args.heuristic_lp_guided_integral_tolerance is not None:
        override["heuristic_lp_guided_integral_tolerance"] = args.heuristic_lp_guided_integral_tolerance
    if args.heuristic_rins_enabled is not None:
        override["heuristic_rins_enabled"] = args.heuristic_rins_enabled
    if args.heuristic_rins_time_limit is not None:
        override["heuristic_rins_time_limit_seconds"] = args.heuristic_rins_time_limit
    if args.heuristic_rins_every_n_selected is not None:
        override["heuristic_rins_every_n_selected"] = args.heuristic_rins_every_n_selected
    if args.heuristic_rins_free_integer_vars is not None:
        override["heuristic_rins_free_integer_vars"] = args.heuristic_rins_free_integer_vars
    if args.heuristic_rins_min_depth is not None:
        override["heuristic_rins_min_depth"] = args.heuristic_rins_min_depth
    if args.heuristic_rins_min_agreement_vars is not None:
        override["heuristic_rins_min_agreement_vars"] = args.heuristic_rins_min_agreement_vars
    if args.heuristic_rins_integral_tolerance is not None:
        override["heuristic_rins_integral_tolerance"] = args.heuristic_rins_integral_tolerance
    if args.heuristic_local_branching_enabled is not None:
        override["heuristic_local_branching_enabled"] = args.heuristic_local_branching_enabled
    if args.heuristic_local_branching_radius is not None:
        override["heuristic_local_branching_radius"] = args.heuristic_local_branching_radius
    if args.heuristic_local_branching_max_binary_vars is not None:
        override["heuristic_local_branching_max_binary_vars"] = args.heuristic_local_branching_max_binary_vars
    if args.heuristic_local_branching_max_fractional_vars is not None:
        override["heuristic_local_branching_max_fractional_vars"] = args.heuristic_local_branching_max_fractional_vars
    if args.heuristic_local_branching_max_gap is not None:
        override["heuristic_local_branching_max_gap"] = args.heuristic_local_branching_max_gap

    config = _build_branch_and_bound_config(
        instance_path=args.instance,
        run_tag=args.tag,
        output_root=args.output_root,
        override=override,
    )
    summary = BranchAndBoundEngine(config).run()
    print(summary.to_dict())


def run_batch(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    dataset_manifest_path = (config_path.parent / payload["dataset_manifest"]).resolve()
    entries = load_dataset_manifest(dataset_manifest_path)
    experiment_name = payload["experiment_name"]
    common = dict(payload.get("common", {}))
    common["learned_policy_model_path"] = _resolve_optional_path(
        config_path.parent,
        common.get("learned_policy_model_path"),
    )
    instance_ids = payload["instance_ids"]
    policies = []
    for policy_payload in payload["policies"]:
        resolved_payload = dict(policy_payload)
        resolved_payload["learned_policy_model_path"] = _resolve_optional_path(
            config_path.parent,
            resolved_payload.get("learned_policy_model_path"),
        )
        policies.append(resolved_payload)
    output_root = payload.get("output_root")
    timestamp_tag = payload.get("tag", experiment_name)

    run_paths = ensure_run_paths(f"{timestamp_tag}_batch", output_root)
    rows = []
    for instance_id in instance_ids:
        entry = find_dataset_entry(entries, instance_id)
        for policy_payload in policies:
            config = _build_branch_and_bound_config(
                instance_path=entry.source_path,
                run_tag=f"{experiment_name}_{instance_id}_{policy_payload['name']}",
                output_root=output_root,
                common=common,
                override=policy_payload,
            )
            summary = BranchAndBoundEngine(config).run()
            row = summary.to_dict()
            row["dataset_id"] = instance_id
            row["group"] = entry.group
            rows.append(row)

    csv_path = run_paths.processed_results_dir / f"{experiment_name}_summary.csv"
    json_path = run_paths.processed_results_dir / f"{experiment_name}_summary.json"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        json_dump(json_path, rows)
    print({"rows": len(rows), "csv": str(csv_path), "json": str(json_path)})


def verify_dataset(args: argparse.Namespace) -> None:
    result = verify_dataset_manifest(args.manifest)
    _print_verification(result)


def verify_literature(args: argparse.Namespace) -> None:
    result = verify_literature_registry(args.registry)
    _print_verification(result)


def _output_pair(prefix: str | Path) -> tuple[Path, Path]:
    base = Path(prefix)
    if base.suffix:
        base = base.with_suffix("")
    return base.with_suffix(".json"), base.with_suffix(".csv")


def extract_root_features_command(args: argparse.Namespace) -> None:
    rows = extract_root_features_from_manifest(
        args.manifest,
        instance_ids=args.instance_id,
        threads=args.threads,
        seed=args.seed,
        lp_proxy_mode=args.lp_proxy_mode,
        lp_proxy_constraint_sample_limit=args.lp_proxy_constraint_sample_limit,
        primal_probe_time_limit_seconds=args.primal_probe_time_limit_seconds,
        primal_probe_max_free_integer_vars=args.primal_probe_max_free_integer_vars,
        primal_probe_plain_enabled=args.primal_probe_plain_enabled,
    )
    json_path, csv_path = _output_pair(args.output_prefix)
    export_rows_json_and_csv(rows, json_path=json_path, csv_path=csv_path)
    print({"rows": len(rows), "json": str(json_path), "csv": str(csv_path)})


def fit_selector_command(args: argparse.Namespace) -> None:
    feature_names = args.feature_name or list(DEFAULT_SELECTOR_FEATURES)
    dataset_rows, model = train_selector_from_files(
        feature_path=args.features,
        summary_paths=args.summary,
        candidate_policies=args.candidate_policy,
        feature_names=feature_names,
        time_weight=args.time_weight,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        complexity_penalty=args.complexity_penalty,
    )
    model_payload = dict(model)
    model_payload["training_artifacts"] = {
        "feature_path": str(Path(args.features).resolve()),
        "summary_paths": [str(Path(path).resolve()) for path in args.summary],
        "candidate_policies": args.candidate_policy,
        "feature_names": feature_names,
        "time_weight": args.time_weight,
    }
    json_dump(args.output_model, model_payload)
    if args.output_dataset_prefix is not None:
        dataset_json, dataset_csv = _output_pair(args.output_dataset_prefix)
        export_rows_json_and_csv(dataset_rows, json_path=dataset_json, csv_path=dataset_csv)
    report_payload = {
        "model_path": str(Path(args.output_model).resolve()),
        "training_summary": model_payload["training_summary"],
        "training_artifacts": model_payload["training_artifacts"],
    }
    if args.output_report is not None:
        json_dump(args.output_report, report_payload)
    print(
        {
            "rows": len(dataset_rows),
            "model": str(args.output_model),
            "report": str(args.output_report) if args.output_report is not None else None,
            "splits": model_payload["training_summary"]["split_count"],
            "avg_regret": model_payload["training_summary"]["selector_average_regret"],
        }
    )


def evaluate_selector_command(args: argparse.Namespace) -> None:
    model = load_selector_model(args.model)
    feature_rows = load_rows(args.features)
    performance_rows = load_summary_rows(args.summary)
    dataset_rows = build_selector_dataset(
        feature_rows,
        performance_rows,
        candidate_policies=list(model["candidate_policies"]),
        feature_names=list(model["feature_names"]),
        time_weight=args.time_weight,
    )
    evaluation = evaluate_selector_model(model, dataset_rows)
    payload = {
        "model_path": str(Path(args.model).resolve()),
        "feature_path": str(Path(args.features).resolve()),
        "summary_paths": [str(Path(path).resolve()) for path in args.summary],
        "evaluation": evaluation,
    }
    json_dump(args.output_report, payload)
    if args.output_assignments_prefix is not None:
        assignments_json, assignments_csv = _output_pair(args.output_assignments_prefix)
        export_rows_json_and_csv(
            evaluation["assignments"],
            json_path=assignments_json,
            csv_path=assignments_csv,
        )
    print(
        {
            "rows": evaluation["rows"],
            "report": str(args.output_report),
            "selector_average_regret": evaluation["selector_average_regret"],
            "split_count": evaluation["split_count"],
        }
    )


def audit_runtime_selector_command(args: argparse.Namespace) -> None:
    baseline_rows = load_summary_rows(args.baseline_summary)
    candidate_policies = args.candidate_policy or infer_candidate_policies(baseline_rows)
    payload = audit_runtime_selector_from_files(
        runtime_summary_path=args.runtime_summary,
        baseline_summary_paths=args.baseline_summary,
        candidate_policies=candidate_policies,
        time_weight=args.time_weight,
    )
    json_dump(args.output_report, payload)
    if args.output_rows_prefix is not None:
        rows_json, rows_csv = _output_pair(args.output_rows_prefix)
        export_rows_json_and_csv(payload["rows"], json_path=rows_json, csv_path=rows_csv)
    print(
        {
            "rows": len(payload["rows"]),
            "report": str(args.output_report),
            "average_realized_regret": payload["average_realized_regret"],
            "candidate_policies": candidate_policies,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan 4 adaptive node selection CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single instance")
    run_parser.add_argument("--instance", required=True)
    run_parser.add_argument("--policy", default="boltzmann_adaptive")
    run_parser.add_argument("--portfolio-short-policy")
    run_parser.add_argument("--portfolio-long-policy")
    run_parser.add_argument("--portfolio-time-threshold-seconds", type=float)
    run_parser.add_argument("--instance-aware-short-small-policy")
    run_parser.add_argument("--instance-aware-short-large-policy")
    run_parser.add_argument("--instance-aware-long-policy")
    run_parser.add_argument("--instance-aware-time-threshold-seconds", type=float)
    run_parser.add_argument("--instance-aware-root-fractional-threshold", type=int)
    run_parser.add_argument("--multi-feature-short-low-complexity-policy")
    run_parser.add_argument("--multi-feature-short-high-complexity-policy")
    run_parser.add_argument("--multi-feature-long-policy")
    run_parser.add_argument("--multi-feature-time-threshold-seconds", type=float)
    run_parser.add_argument("--multi-feature-root-fractional-threshold", type=int)
    run_parser.add_argument("--multi-feature-root-fractional-ratio-threshold", type=float)
    run_parser.add_argument("--multi-feature-integer-variable-threshold", type=int)
    run_parser.add_argument("--multi-feature-root-solve-seconds-threshold", type=float)
    run_parser.add_argument("--multi-feature-complexity-votes-required", type=int)
    run_parser.add_argument("--learned-policy-model-path")
    run_parser.add_argument("--learned-policy-default")
    run_parser.add_argument(
        "--root-lp-proxy-mode",
        choices=["auto", "full", "sampled", "variable_only", "skip"],
    )
    run_parser.add_argument("--root-lp-proxy-constraint-sample-limit", type=int)
    run_parser.add_argument("--root-primal-probe-time-limit-seconds", type=float)
    run_parser.add_argument("--root-primal-probe-max-free-integer-vars", type=int)
    run_parser.add_argument("--root-primal-probe-plain-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--shortlist-size", type=int, default=32)
    run_parser.add_argument("--node-limit", type=int, default=300)
    run_parser.add_argument("--time-limit", type=float, default=60.0)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--solver-backend", choices=["cplex", "gurobi"], default="cplex")
    run_parser.add_argument("--tag", default="run")
    run_parser.add_argument("--output-root")
    run_parser.add_argument("--heuristic-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--heuristic-time-limit", type=float)
    run_parser.add_argument("--heuristic-every-n-selected", type=int)
    run_parser.add_argument("--heuristic-repair-max-free-integer-vars", type=int)
    run_parser.add_argument("--heuristic-diving-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--heuristic-diving-free-integer-vars", type=int)
    run_parser.add_argument("--heuristic-diving-stage-factor", type=int)
    run_parser.add_argument("--heuristic-diving-min-depth", type=int)
    run_parser.add_argument("--heuristic-diving-max-disagreement-vars", type=int)
    run_parser.add_argument("--heuristic-lp-guided-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--heuristic-lp-guided-free-integer-vars", type=int)
    run_parser.add_argument("--heuristic-lp-guided-stage-factor", type=int)
    run_parser.add_argument("--heuristic-lp-guided-min-depth", type=int)
    run_parser.add_argument("--heuristic-lp-guided-min-disagreement-vars", type=int)
    run_parser.add_argument("--heuristic-lp-guided-integral-tolerance", type=float)
    run_parser.add_argument("--heuristic-rins-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--heuristic-rins-time-limit", type=float)
    run_parser.add_argument("--heuristic-rins-every-n-selected", type=int)
    run_parser.add_argument("--heuristic-rins-free-integer-vars", type=int)
    run_parser.add_argument("--heuristic-rins-min-depth", type=int)
    run_parser.add_argument("--heuristic-rins-min-agreement-vars", type=int)
    run_parser.add_argument("--heuristic-rins-integral-tolerance", type=float)
    run_parser.add_argument("--heuristic-local-branching-enabled", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--heuristic-local-branching-radius", type=int)
    run_parser.add_argument("--heuristic-local-branching-max-binary-vars", type=int)
    run_parser.add_argument("--heuristic-local-branching-max-fractional-vars", type=int)
    run_parser.add_argument("--heuristic-local-branching-max-gap", type=float)
    run_parser.set_defaults(func=run_single)

    batch_parser = subparsers.add_parser("batch-run", help="Run experiment batch from YAML")
    batch_parser.add_argument("--config", required=True)
    batch_parser.set_defaults(func=run_batch)

    root_features_parser = subparsers.add_parser("extract-root-features", help="Extract auditable root LP features")
    root_features_parser.add_argument("--manifest", required=True)
    root_features_parser.add_argument("--output-prefix", required=True)
    root_features_parser.add_argument("--instance-id", action="append")
    root_features_parser.add_argument("--threads", type=int, default=1)
    root_features_parser.add_argument("--seed", type=int, default=0)
    root_features_parser.add_argument(
        "--lp-proxy-mode",
        choices=["full", "sampled", "variable_only", "skip"],
        default="full",
    )
    root_features_parser.add_argument("--lp-proxy-constraint-sample-limit", type=int)
    root_features_parser.add_argument("--primal-probe-time-limit-seconds", type=float, default=0.0)
    root_features_parser.add_argument("--primal-probe-max-free-integer-vars", type=int, default=12)
    root_features_parser.add_argument("--primal-probe-plain-enabled", action=argparse.BooleanOptionalAction, default=True)
    root_features_parser.set_defaults(func=extract_root_features_command)

    fit_selector_parser = subparsers.add_parser("fit-selector", help="Fit an auditable regret-tree selector")
    fit_selector_parser.add_argument("--features", required=True)
    fit_selector_parser.add_argument("--summary", action="append", required=True)
    fit_selector_parser.add_argument("--candidate-policy", action="append", required=True)
    fit_selector_parser.add_argument("--feature-name", action="append")
    fit_selector_parser.add_argument("--time-weight", type=float, default=0.05)
    fit_selector_parser.add_argument("--max-depth", type=int, default=2)
    fit_selector_parser.add_argument("--min-samples-leaf", type=int, default=1)
    fit_selector_parser.add_argument("--complexity-penalty", type=float, default=0.01)
    fit_selector_parser.add_argument("--output-model", required=True)
    fit_selector_parser.add_argument("--output-report")
    fit_selector_parser.add_argument("--output-dataset-prefix")
    fit_selector_parser.set_defaults(func=fit_selector_command)

    evaluate_selector_parser = subparsers.add_parser("evaluate-selector", help="Evaluate a fitted selector on audited summaries")
    evaluate_selector_parser.add_argument("--model", required=True)
    evaluate_selector_parser.add_argument("--features", required=True)
    evaluate_selector_parser.add_argument("--summary", action="append", required=True)
    evaluate_selector_parser.add_argument("--time-weight", type=float, default=0.05)
    evaluate_selector_parser.add_argument("--output-report", required=True)
    evaluate_selector_parser.add_argument("--output-assignments-prefix")
    evaluate_selector_parser.set_defaults(func=evaluate_selector_command)

    audit_runtime_parser = subparsers.add_parser(
        "audit-runtime-selector",
        help="Audit realized regret of a runtime selector summary against direct policy baselines",
    )
    audit_runtime_parser.add_argument("--runtime-summary", required=True)
    audit_runtime_parser.add_argument("--baseline-summary", action="append", required=True)
    audit_runtime_parser.add_argument("--candidate-policy", action="append")
    audit_runtime_parser.add_argument("--time-weight", type=float, default=0.05)
    audit_runtime_parser.add_argument("--output-report", required=True)
    audit_runtime_parser.add_argument("--output-rows-prefix")
    audit_runtime_parser.set_defaults(func=audit_runtime_selector_command)

    dataset_parser = subparsers.add_parser("verify-dataset-manifest", help="Verify dataset manifest hashes")
    dataset_parser.add_argument("--manifest", required=True)
    dataset_parser.set_defaults(func=verify_dataset)

    literature_parser = subparsers.add_parser("verify-literature-registry", help="Verify literature registry")
    literature_parser.add_argument("--registry", required=True)
    literature_parser.set_defaults(func=verify_literature)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
