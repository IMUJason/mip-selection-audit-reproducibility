from __future__ import annotations

import math
from statistics import pstdev

from .models import NodeState, ScoreConfig


def normalized_entropy(probabilities: list[float]) -> tuple[float, float]:
    filtered = [prob for prob in probabilities if prob > 0.0]
    if not filtered:
        return 0.0, 0.0
    entropy = -sum(prob * math.log(prob) for prob in filtered)
    if len(filtered) <= 1:
        return entropy, 0.0
    return entropy, entropy / math.log(len(filtered))


def softmax_from_scores(scores: dict[str, float], beta: float) -> dict[str, float]:
    if not scores:
        return {}
    scaled = {node_id: -beta * score for node_id, score in scores.items()}
    shift = max(scaled.values())
    weights = {node_id: math.exp(value - shift) for node_id, value in scaled.items()}
    total = sum(weights.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return {node_id: uniform for node_id in scores}
    return {node_id: weight / total for node_id, weight in weights.items()}


def compute_node_scores(
    nodes: list[NodeState],
    current_step: int,
    incumbent_objective: float | None,
    score_config: ScoreConfig,
) -> dict[str, float]:
    if not nodes:
        return {}

    bounds = [node.lp_objective for node in nodes if node.lp_objective is not None]
    if not bounds:
        return {node.node_id: 0.0 for node in nodes}

    reference_bound = max(bounds)
    if incumbent_objective is None:
        scale = max(abs(reference_bound), 1.0)
    else:
        scale = max(abs(incumbent_objective - reference_bound), 1.0)

    max_depth = max(node.depth for node in nodes) or 1
    ages = [node.age(current_step) for node in nodes]
    max_age = max(ages) or 1

    scores: dict[str, float] = {}
    for node in nodes:
        if node.lp_objective is None:
            scores[node.node_id] = float("inf")
            continue
        bound_term = (reference_bound - node.lp_objective) / max(scale, score_config.epsilon)
        depth_term = node.depth / max_depth
        age_term = node.age(current_step) / max_age
        scores[node.node_id] = bound_term + score_config.depth_weight * depth_term - score_config.age_weight * age_term
    return scores


def pool_dispersion(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def starvation_ratio(nodes: list[NodeState], current_step: int, age_ratio: float) -> float:
    if not nodes:
        return 0.0
    ages = [node.age(current_step) for node in nodes]
    max_age = max(ages)
    if max_age <= 0:
        return 0.0
    threshold = max_age * age_ratio
    return sum(age >= threshold for age in ages) / len(ages)


def relative_gap(incumbent: float | None, best_bound: float | None) -> float | None:
    if incumbent is None or best_bound is None:
        return None
    denominator = max(abs(incumbent), 1.0)
    return max(incumbent - best_bound, 0.0) / denominator
