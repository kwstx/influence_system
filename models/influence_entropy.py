from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List, Sequence


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _normalised_entropy(weights: Sequence[float]) -> float:
    if not weights:
        return 0.0
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0.0:
        return 0.0
    probs = [max(0.0, float(w)) / total for w in weights if w > 0.0]
    if len(probs) <= 1:
        return 0.0
    raw_entropy = -sum(p * log(p) for p in probs)
    max_entropy = log(len(probs))
    if max_entropy <= 0.0:
        return 0.0
    return _clamp(raw_entropy / max_entropy)


@dataclass(frozen=True)
class InfluenceEntropySnapshot:
    entropy: float
    threshold: float
    concentration: float
    regularization_strength: float
    regularization_applied: bool
    dominant_agent_ids: List[str]
    underutilized_reliable_agent_ids: List[str]

    def to_dict(self) -> Dict[str, float | bool | List[str]]:
        return {
            "entropy": self.entropy,
            "threshold": self.threshold,
            "concentration": self.concentration,
            "regularization_strength": self.regularization_strength,
            "regularization_applied": self.regularization_applied,
            "dominant_agent_ids": self.dominant_agent_ids,
            "underutilized_reliable_agent_ids": self.underutilized_reliable_agent_ids,
        }


class InfluenceEntropy:
    """
    Measures concentration of trust distribution and regularizes influence
    weights when dominance concentration becomes too high.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.75,
        dominance_softness: float = 0.35,
        opportunity_boost_strength: float = 0.30,
    ):
        if not (0.0 < entropy_threshold <= 1.0):
            raise ValueError("entropy_threshold must be in (0, 1]")
        if not (0.0 <= dominance_softness <= 1.0):
            raise ValueError("dominance_softness must be in [0, 1]")
        if not (0.0 <= opportunity_boost_strength <= 1.0):
            raise ValueError("opportunity_boost_strength must be in [0, 1]")
        self.entropy_threshold = entropy_threshold
        self.dominance_softness = dominance_softness
        self.opportunity_boost_strength = opportunity_boost_strength

    def evaluate(
        self,
        agent_ids: Sequence[str],
        trust_coefficients: Sequence[float],
        base_weights: Sequence[float],
        reliability_scores: Sequence[float],
    ) -> tuple[List[float], InfluenceEntropySnapshot]:
        if not (
            len(agent_ids)
            == len(trust_coefficients)
            == len(base_weights)
            == len(reliability_scores)
        ):
            raise ValueError("All input sequences must have equal length")
        if not agent_ids:
            snapshot = InfluenceEntropySnapshot(
                entropy=0.0,
                threshold=self.entropy_threshold,
                concentration=1.0,
                regularization_strength=0.0,
                regularization_applied=False,
                dominant_agent_ids=[],
                underutilized_reliable_agent_ids=[],
            )
            return [], snapshot

        entropy = _normalised_entropy(trust_coefficients)
        concentration = 1.0 - entropy
        deficit = _clamp((self.entropy_threshold - entropy) / self.entropy_threshold)
        reg_strength = deficit * self.dominance_softness
        applied = reg_strength > 0.0

        n_agents = len(agent_ids)
        uniform = [1.0 / n_agents] * n_agents
        blended = [
            ((1.0 - reg_strength) * weight) + (reg_strength * uniform_weight)
            for weight, uniform_weight in zip(base_weights, uniform)
        ]

        reliability = [_clamp(score) for score in reliability_scores]
        utilization = [_clamp(weight) for weight in blended]
        opportunity_scores = [
            rel * (1.0 - use)
            for rel, use in zip(reliability, utilization)
        ]
        opportunity_total = sum(opportunity_scores)
        if opportunity_total > 0.0:
            opportunity_distribution = [
                score / opportunity_total for score in opportunity_scores
            ]
        else:
            opportunity_distribution = uniform

        boost_strength = reg_strength * self.opportunity_boost_strength
        final_weights = [
            ((1.0 - boost_strength) * weight) + (boost_strength * opp_weight)
            for weight, opp_weight in zip(blended, opportunity_distribution)
        ]
        total_final = sum(final_weights)
        if total_final > 0.0:
            final_weights = [w / total_final for w in final_weights]
        else:
            final_weights = uniform

        dominant = [
            aid
            for aid, w in zip(agent_ids, base_weights)
            if w > (1.0 / n_agents)
        ]
        underutilized_reliable = [
            aid
            for aid, rel, util in zip(agent_ids, reliability, utilization)
            if rel >= 0.6 and util < (1.0 / n_agents)
        ]

        snapshot = InfluenceEntropySnapshot(
            entropy=entropy,
            threshold=self.entropy_threshold,
            concentration=concentration,
            regularization_strength=reg_strength,
            regularization_applied=applied,
            dominant_agent_ids=dominant,
            underutilized_reliable_agent_ids=underutilized_reliable,
        )
        return final_weights, snapshot
