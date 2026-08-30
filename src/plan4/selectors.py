from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .metrics import compute_node_scores, normalized_entropy, pool_dispersion, softmax_from_scores, starvation_ratio
from .models import NodeState, ScoreConfig, SelectionDiagnostics


def _weighted_sample_without_replacement(
    items: list[NodeState],
    probabilities: dict[str, float],
    k: int,
    rng: random.Random,
) -> list[NodeState]:
    remaining = list(items)
    selected: list[NodeState] = []
    limit = min(k, len(remaining))

    for _ in range(limit):
        weights = [probabilities.get(node.node_id, 0.0) for node in remaining]
        total = sum(weights)
        if total <= 0.0:
            chosen_index = rng.randrange(len(remaining))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            chosen_index = 0
            for index, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= threshold:
                    chosen_index = index
                    break
        chosen = remaining.pop(chosen_index)
        selected.append(chosen)
        remaining_total = sum(probabilities.get(node.node_id, 0.0) for node in remaining)
        if remaining_total > 0.0:
            probabilities = {node.node_id: probabilities.get(node.node_id, 0.0) / remaining_total for node in remaining}
        else:
            probabilities = {node.node_id: 1.0 / len(remaining) for node in remaining} if remaining else {}
    return selected


def _mix_with_uniform(probabilities: dict[str, float], epsilon_floor: float) -> dict[str, float]:
    if not probabilities:
        return {}
    epsilon_floor = min(max(epsilon_floor, 0.0), 1.0)
    if epsilon_floor == 0.0:
        return probabilities
    uniform = 1.0 / len(probabilities)
    return {
        node_id: (1.0 - epsilon_floor) * prob + epsilon_floor * uniform
        for node_id, prob in probabilities.items()
    }


@dataclass
class SelectorContext:
    current_step: int
    incumbent_objective: float | None
    beta: float
    batch_size: int
    elapsed_seconds: float = 0.0
    time_limit_seconds: float = 0.0
    pseudocosts: dict[str, dict[str, float]] | None = None


class NodeSelector:
    def __init__(
        self,
        policy: str,
        score_config: ScoreConfig,
        shortlist_size: int | None,
        batch_size: int,
        random_seed: int,
        starvation_age_ratio: float,
    ) -> None:
        self.policy = policy
        self.score_config = score_config
        self.shortlist_size = shortlist_size
        self.batch_size = batch_size
        self.random = random.Random(random_seed)
        self.starvation_age_ratio = starvation_age_ratio

    @staticmethod
    def _bound_value(node: NodeState) -> float:
        return node.lp_objective if node.lp_objective is not None else float("-inf")

    def _estimate_penalty(
        self,
        node: NodeState,
        pseudocosts: dict[str, dict[str, float]] | None,
    ) -> float:
        penalty = 0.0
        for var_name, value in node.fractional_variables.items():
            floor_value = math.floor(value)
            frac_down = max(value - floor_value, self.score_config.epsilon)
            frac_up = max(math.ceil(value) - value, self.score_config.epsilon)
            pseudo = (pseudocosts or {}).get(var_name, {})
            down_cost = pseudo.get("down_sum", 0.0) / pseudo.get("down_count", 1.0) if pseudo.get("down_count", 0.0) > 0.0 else frac_down
            up_cost = pseudo.get("up_sum", 0.0) / pseudo.get("up_count", 1.0) if pseudo.get("up_count", 0.0) > 0.0 else frac_up
            penalty += min(down_cost * frac_down, up_cost * frac_up)
        return penalty

    def _best_estimate_costs(self, nodes: list[NodeState], context: SelectorContext) -> dict[str, float]:
        base_scores = compute_node_scores(nodes, context.current_step, context.incumbent_objective, self.score_config)
        max_fractional = max((len(node.fractional_variables) for node in nodes), default=1) or 1
        total_visits = sum(node.visit_count for node in nodes)
        estimate_values = {
            node.node_id: (node.lp_objective or 0.0) + self._estimate_penalty(node, context.pseudocosts)
            for node in nodes
        }
        min_estimate = min(estimate_values.values(), default=0.0)
        max_estimate = max(estimate_values.values(), default=1.0)
        estimate_scale = max(max_estimate - min_estimate, 1.0)

        best_estimate: dict[str, float] = {}
        for node in nodes:
            base = base_scores[node.node_id]
            frac_ratio = len(node.fractional_variables) / max_fractional
            estimate_proxy = (max_estimate - estimate_values[node.node_id]) / estimate_scale
            visit_bonus = self.score_config.uct_weight * math.sqrt(math.log(total_visits + 2.0) / (node.visit_count + 1.0))
            best_estimate[node.node_id] = (
                (1.0 - self.score_config.estimate_weight) * base
                + self.score_config.estimate_weight * estimate_proxy
                + self.score_config.fractional_weight * frac_ratio
                - visit_bonus
            )
        return best_estimate

    def _plateau_rescue_candidates(
        self,
        nodes: list[NodeState],
        scores: dict[str, float],
    ) -> list[NodeState]:
        bounds = [self._bound_value(node) for node in nodes if node.lp_objective is not None]
        if not bounds:
            return []
        topk = max(1, min(self.score_config.plateau_rescue_topk, len(nodes)))
        minimum_pool = max(1, self.score_config.plateau_rescue_min_pool)
        reference_bound = max(bounds)
        band = max(abs(reference_bound), 1.0) * max(self.score_config.plateau_rescue_bound_band_ratio, 0.0)
        strong_bound_nodes = [
            node
            for node in nodes
            if node.lp_objective is not None and (reference_bound - self._bound_value(node)) <= band + self.score_config.epsilon
        ]
        if len(strong_bound_nodes) < minimum_pool:
            return []
        ordered_candidates = sorted(
            strong_bound_nodes,
            key=lambda node: (
                len(node.fractional_variables),
                -node.depth,
                scores[node.node_id],
                node.node_id,
            ),
        )
        return ordered_candidates[:topk]

    def _select_safeguarded_hybrid(
        self,
        nodes: list[NodeState],
        ordered: list[NodeState],
        shortlist: list[NodeState],
        shortlist_scores: dict[str, float],
        scores: dict[str, float],
        context: SelectorContext,
        plateau_aware: bool,
    ) -> tuple[list[NodeState], dict[str, float], str]:
        best_bound_node = max(nodes, key=lambda node: (self._bound_value(node), node.depth, -node.created_step))
        depth_rescue_candidates = ordered[: max(1, min(self.score_config.depth_rescue_topk, len(ordered)))]
        depth_rescue_node = max(depth_rescue_candidates, key=lambda node: (node.depth, -node.age(context.current_step), self._bound_value(node)))
        primal_rescue_candidates = ordered[: max(1, min(self.score_config.primal_rescue_topk, len(ordered)))]
        primal_rescue_node = min(
            primal_rescue_candidates,
            key=lambda node: (
                len(node.fractional_variables),
                scores[node.node_id],
                -node.depth,
                node.node_id,
            ),
        )
        plateau_rescue_candidates = self._plateau_rescue_candidates(nodes, scores) if plateau_aware else []
        plateau_rescue_node = plateau_rescue_candidates[0] if plateau_rescue_candidates else None

        if self.score_config.best_bound_freq > 0 and context.current_step % self.score_config.best_bound_freq == 0:
            return [best_bound_node], {best_bound_node.node_id: 1.0}, "best_bound_rescue"
        if (
            plateau_rescue_node is not None
            and self.score_config.plateau_rescue_freq > 0
            and context.current_step % self.score_config.plateau_rescue_freq == 0
        ):
            return [plateau_rescue_node], {plateau_rescue_node.node_id: 1.0}, "plateau_rescue"
        if (
            context.incumbent_objective is not None
            and self.score_config.primal_rescue_freq > 0
            and context.current_step % self.score_config.primal_rescue_freq == 0
        ):
            return [primal_rescue_node], {primal_rescue_node.node_id: 1.0}, "primal_rescue"
        if self.score_config.depth_rescue_freq > 0 and context.current_step % self.score_config.depth_rescue_freq == 0:
            return [depth_rescue_node], {depth_rescue_node.node_id: 1.0}, "depth_rescue"

        probabilities = softmax_from_scores(shortlist_scores, context.beta)
        probabilities = _mix_with_uniform(probabilities, self.score_config.epsilon_floor)
        selected = _weighted_sample_without_replacement(shortlist, probabilities.copy(), context.batch_size, self.random)
        return selected, probabilities, "softmax"

    def _select_phase_switch_hybrid(
        self,
        shortlist: list[NodeState],
        shortlist_scores: dict[str, float],
        context: SelectorContext,
        late_phase: bool,
    ) -> tuple[list[NodeState], dict[str, float], str]:
        if late_phase:
            selected = shortlist[: context.batch_size]
            probabilities = {node.node_id: (1.0 if index == 0 else 0.0) for index, node in enumerate(selected)}
            return selected, probabilities, "phase_late_best_estimate"
        probabilities = softmax_from_scores(shortlist_scores, context.beta)
        probabilities = _mix_with_uniform(probabilities, self.score_config.epsilon_floor)
        selected = _weighted_sample_without_replacement(shortlist, probabilities.copy(), context.batch_size, self.random)
        return selected, probabilities, "phase_early_softmax"

    def _select_ramped_estimate_hybrid(
        self,
        shortlist: list[NodeState],
        shortlist_scores: dict[str, float],
        context: SelectorContext,
        estimate_rescue_active: bool,
        late_phase: bool,
    ) -> tuple[list[NodeState], dict[str, float], str]:
        if estimate_rescue_active:
            selected = shortlist[: context.batch_size]
            probabilities = {node.node_id: (1.0 if index == 0 else 0.0) for index, node in enumerate(selected)}
            selection_mode = "ramped_late_estimate_rescue" if late_phase else "ramped_estimate_rescue"
            return selected, probabilities, selection_mode

        probabilities = softmax_from_scores(shortlist_scores, context.beta)
        probabilities = _mix_with_uniform(probabilities, self.score_config.epsilon_floor)
        selected = _weighted_sample_without_replacement(shortlist, probabilities.copy(), context.batch_size, self.random)
        return selected, probabilities, "ramped_softmax"

    def select(self, nodes: list[NodeState], context: SelectorContext) -> tuple[list[NodeState], SelectionDiagnostics]:
        if not nodes:
            diagnostics = SelectionDiagnostics(
                policy=self.policy,
                candidate_pool_size=0,
                shortlist_size=0,
                batch_size=0,
                beta_before=context.beta,
                beta_after=context.beta,
                entropy=0.0,
                normalized_entropy=0.0,
                dispersion=0.0,
                starvation_ratio=0.0,
                selected_node_ids=[],
                probabilities={},
                scores={},
                selection_mode="empty",
            )
            return [], diagnostics

        late_phase = False
        if self.policy == "phase_switch_hybrid":
            early_scores = compute_node_scores(nodes, context.current_step, context.incumbent_objective, self.score_config)
            late_scores = self._best_estimate_costs(nodes, context)
            time_ratio = context.elapsed_seconds / max(context.time_limit_seconds, self.score_config.epsilon)
            late_phase = (
                context.current_step >= max(self.score_config.phase_switch_min_step, 1)
                and time_ratio >= self.score_config.phase_switch_time_ratio
            )
            scores = late_scores if late_phase else early_scores
        elif self.policy == "ramped_estimate_hybrid":
            softmax_scores = compute_node_scores(nodes, context.current_step, context.incumbent_objective, self.score_config)
            estimate_scores = self._best_estimate_costs(nodes, context)
            time_ratio = context.elapsed_seconds / max(context.time_limit_seconds, self.score_config.epsilon)
            late_phase = time_ratio >= self.score_config.estimate_rescue_time_ratio
            rescue_freq = (
                self.score_config.estimate_rescue_late_freq if late_phase else self.score_config.estimate_rescue_freq
            )
            estimate_rescue_active = (
                rescue_freq > 0
                and context.current_step >= max(self.score_config.estimate_rescue_min_step, 1)
                and context.current_step % rescue_freq == 0
            )
            scores = estimate_scores if estimate_rescue_active else softmax_scores
        elif self.policy in {"best_estimate", "safeguarded_hybrid", "plateau_aware_hybrid"}:
            scores = self._best_estimate_costs(nodes, context)
        else:
            scores = compute_node_scores(nodes, context.current_step, context.incumbent_objective, self.score_config)
        finite_scores = [value for value in scores.values() if value != float("inf")]
        dispersion = pool_dispersion(finite_scores)
        starved = starvation_ratio(nodes, context.current_step, self.starvation_age_ratio)

        ordered = sorted(nodes, key=lambda node: (scores[node.node_id], -self._bound_value(node), -node.depth, node.node_id))
        shortlist = ordered[: self.shortlist_size] if self.shortlist_size else ordered
        shortlist_scores = {node.node_id: scores[node.node_id] for node in shortlist}
        selection_mode = self.policy

        if self.policy == "best_bound":
            selected = [max(nodes, key=lambda node: (self._bound_value(node), node.depth, -node.created_step))]
            probabilities = {selected[0].node_id: 1.0}
            selection_mode = "best_bound"
        elif self.policy == "depth_first":
            selected = [max(nodes, key=lambda node: (node.depth, -node.created_step, self._bound_value(node)))]
            probabilities = {selected[0].node_id: 1.0}
            selection_mode = "depth_first"
        elif self.policy == "hybrid_best_bound_depth":
            selected = sorted(nodes, key=lambda node: (-self._bound_value(node), -node.depth, node.node_id))[: context.batch_size]
            probabilities = {node.node_id: (1.0 if index == 0 else 0.0) for index, node in enumerate(selected)}
            selection_mode = "hybrid_best_bound_depth"
        elif self.policy == "random_uniform":
            selected = self.random.sample(nodes, k=min(context.batch_size, len(nodes)))
            uniform = 1.0 / len(nodes)
            probabilities = {node.node_id: uniform for node in nodes}
            selection_mode = "random_uniform"
        elif self.policy == "greedy_score":
            selected = shortlist[: context.batch_size]
            probabilities = {node.node_id: (1.0 if index == 0 else 0.0) for index, node in enumerate(selected)}
            selection_mode = "greedy_score"
        elif self.policy == "best_estimate":
            selected = shortlist[: context.batch_size]
            probabilities = {node.node_id: (1.0 if index == 0 else 0.0) for index, node in enumerate(selected)}
            selection_mode = "best_estimate"
        elif self.policy in {"boltzmann_fixed", "boltzmann_adaptive"}:
            probabilities = softmax_from_scores(shortlist_scores, context.beta)
            probabilities = _mix_with_uniform(probabilities, self.score_config.epsilon_floor)
            selected = _weighted_sample_without_replacement(shortlist, probabilities.copy(), context.batch_size, self.random)
            selection_mode = "softmax"
        elif self.policy == "safeguarded_hybrid":
            selected, probabilities, selection_mode = self._select_safeguarded_hybrid(
                nodes=nodes,
                ordered=ordered,
                shortlist=shortlist,
                shortlist_scores=shortlist_scores,
                scores=scores,
                context=context,
                plateau_aware=False,
            )
        elif self.policy == "plateau_aware_hybrid":
            selected, probabilities, selection_mode = self._select_safeguarded_hybrid(
                nodes=nodes,
                ordered=ordered,
                shortlist=shortlist,
                shortlist_scores=shortlist_scores,
                scores=scores,
                context=context,
                plateau_aware=True,
            )
        elif self.policy == "phase_switch_hybrid":
            selected, probabilities, selection_mode = self._select_phase_switch_hybrid(
                shortlist=shortlist,
                shortlist_scores=shortlist_scores,
                context=context,
                late_phase=late_phase,
            )
        elif self.policy == "ramped_estimate_hybrid":
            selected, probabilities, selection_mode = self._select_ramped_estimate_hybrid(
                shortlist=shortlist,
                shortlist_scores=shortlist_scores,
                context=context,
                estimate_rescue_active=estimate_rescue_active,
                late_phase=late_phase,
            )
        else:
            raise ValueError(f"Unsupported policy: {self.policy}")

        entropy, norm_entropy = normalized_entropy(list(probabilities.values()))
        diagnostics = SelectionDiagnostics(
            policy=self.policy,
            candidate_pool_size=len(nodes),
            shortlist_size=len(shortlist),
            batch_size=len(selected),
            beta_before=context.beta,
            beta_after=context.beta,
            entropy=entropy,
            normalized_entropy=norm_entropy,
            dispersion=dispersion,
            starvation_ratio=starved,
            selected_node_ids=[node.node_id for node in selected],
            probabilities=probabilities,
            scores=shortlist_scores
            if self.policy.startswith("boltzmann") or self.policy in {"greedy_score", "best_estimate", "safeguarded_hybrid", "plateau_aware_hybrid", "phase_switch_hybrid", "ramped_estimate_hybrid"}
            else {},
            selection_mode=selection_mode,
        )
        return selected, diagnostics
