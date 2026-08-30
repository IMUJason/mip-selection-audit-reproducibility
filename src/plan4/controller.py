from __future__ import annotations

import math

from .metrics import compute_node_scores, pool_dispersion, starvation_ratio
from .models import BetaConfig, NodeState, ScoreConfig


class AdaptiveBetaController:
    def __init__(self, beta_config: BetaConfig, score_config: ScoreConfig) -> None:
        self.beta_config = beta_config
        self.score_config = score_config

    def update(
        self,
        current_beta: float,
        active_nodes: list[NodeState],
        current_step: int,
        incumbent_objective: float | None,
        incumbent_improved: bool,
        steps_since_improvement: int,
    ) -> tuple[float, dict[str, float]]:
        scores = compute_node_scores(active_nodes, current_step, incumbent_objective, self.score_config)
        finite_scores = [value for value in scores.values() if math.isfinite(value)]
        dispersion = pool_dispersion(finite_scores)
        starved = starvation_ratio(active_nodes, current_step, self.beta_config.starvation_age_ratio)
        stagnation = min(steps_since_improvement / max(self.beta_config.stagnation_window, 1), 1.0)
        improvement = 1.0 if incumbent_improved else 0.0

        exponent = (
            self.beta_config.improve_gain * improvement
            - self.beta_config.stagnation_penalty * stagnation
            + self.beta_config.dispersion_gain * dispersion
            - self.beta_config.starvation_penalty * starved
        )
        next_beta = current_beta * math.exp(exponent)
        next_beta = min(max(next_beta, self.beta_config.min_beta), self.beta_config.max_beta)
        metrics = {
            "dispersion": dispersion,
            "starvation_ratio": starved,
            "stagnation_signal": stagnation,
            "improvement_signal": improvement,
        }
        return next_beta, metrics
