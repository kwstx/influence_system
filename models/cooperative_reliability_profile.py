from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import exp, log
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional
import uuid

from models.real_world_calibration import CalibrationRecord


def _bounded_average(values: Iterable[float]) -> float:
    seq = [max(0.0, min(1.0, float(v))) for v in values]
    if not seq:
        return 0.0
    return mean(seq)


def _slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    return mean(diffs)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class TrustWeightingParameters:
    version: str = "1.0.0"
    predictive_weight: float = 0.26
    marginal_weight: float = 0.19
    synergy_weight: float = 0.19
    stability_weight: float = 0.20
    persistence_weight: float = 0.16
    predictive_exponent: float = 1.1
    marginal_exponent: float = 1.0
    synergy_exponent: float = 1.0
    stability_exponent: float = 1.1
    persistence_exponent: float = 0.95
    interaction_gain: float = 0.18
    second_order_gain: float = 0.12
    logistic_steepness: float = 4.0
    logistic_center: float = 0.53
    epsilon: float = 1e-6

    def normalized_weights(self) -> Dict[str, float]:
        weights = {
            "predictive": max(0.0, float(self.predictive_weight)),
            "marginal": max(0.0, float(self.marginal_weight)),
            "synergy": max(0.0, float(self.synergy_weight)),
            "stability": max(0.0, float(self.stability_weight)),
            "persistence": max(0.0, float(self.persistence_weight)),
        }
        total = sum(weights.values())
        if total <= 0.0:
            return {k: 0.2 for k in weights}
        return {k: v / total for k, v in weights.items()}

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "version": self.version,
            "predictive_weight": self.predictive_weight,
            "marginal_weight": self.marginal_weight,
            "synergy_weight": self.synergy_weight,
            "stability_weight": self.stability_weight,
            "persistence_weight": self.persistence_weight,
            "predictive_exponent": self.predictive_exponent,
            "marginal_exponent": self.marginal_exponent,
            "synergy_exponent": self.synergy_exponent,
            "stability_exponent": self.stability_exponent,
            "persistence_exponent": self.persistence_exponent,
            "interaction_gain": self.interaction_gain,
            "second_order_gain": self.second_order_gain,
            "logistic_steepness": self.logistic_steepness,
            "logistic_center": self.logistic_center,
            "epsilon": self.epsilon,
        }


class TrustWeightingFunction:
    """
    Computes a continuous trust coefficient from multi-dimensional cooperative metrics
    using non-linear interactions, multiplicative coupling, and logistic normalization.
    """

    def __init__(self, parameters: Optional[TrustWeightingParameters] = None) -> None:
        self.parameters = parameters or TrustWeightingParameters()

    @property
    def version(self) -> str:
        return self.parameters.version

    def compute(
        self,
        predictive_accuracy_index: float,
        marginal_cooperative_influence: float,
        synergy_density_contribution: float,
        cooperative_stability_score: float,
        long_term_impact_persistence: float,
    ) -> float:
        p = _clamp01(predictive_accuracy_index)
        m = _clamp01(marginal_cooperative_influence)
        s = _clamp01(synergy_density_contribution)
        c = _clamp01(cooperative_stability_score)
        l = _clamp01(long_term_impact_persistence)

        params = self.parameters
        weights = params.normalized_weights()
        eps = max(float(params.epsilon), 1e-12)

        weighted_log_sum = (
            weights["predictive"] * log(max(eps, p ** max(0.1, params.predictive_exponent)))
            + weights["marginal"] * log(max(eps, m ** max(0.1, params.marginal_exponent)))
            + weights["synergy"] * log(max(eps, s ** max(0.1, params.synergy_exponent)))
            + weights["stability"] * log(max(eps, c ** max(0.1, params.stability_exponent)))
            + weights["persistence"] * log(max(eps, l ** max(0.1, params.persistence_exponent)))
        )
        multiplicative_core = exp(weighted_log_sum)

        pairwise_interaction = (
            (p * c) + (m * s) + (s * l) + (p * l) + (m * c)
        ) / 5.0
        higher_order_interaction = p * m * s * c * l

        raw = (
            multiplicative_core
            + float(params.interaction_gain) * pairwise_interaction
            + float(params.second_order_gain) * higher_order_interaction
        )
        steepness = max(0.1, float(params.logistic_steepness))
        center = _clamp01(float(params.logistic_center))
        trust = 1.0 / (1.0 + exp(-steepness * (raw - center)))
        return _clamp01(trust)


@dataclass(frozen=True)
class CooperativeReliabilitySnapshot:
    profile_id: str
    agent_id: str
    generated_at: datetime
    calibration_consistency: float
    synergy_density_participation: float
    marginal_cooperative_influence_consistency: float
    collaborative_stability: float
    collective_outcome_reliability: float
    trend: str
    long_term_impact_persistence: float = 0.5
    trust_weighting_version: str = "1.0.0"
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "agent_id": self.agent_id,
            "generated_at": self.generated_at.isoformat(),
            "metrics": {
                "calibration_consistency": self.calibration_consistency,
                "synergy_density_participation": self.synergy_density_participation,
                "marginal_cooperative_influence_consistency": self.marginal_cooperative_influence_consistency,
                "collaborative_stability": self.collaborative_stability,
                "long_term_impact_persistence": self.long_term_impact_persistence,
                "trust_weighting_version": self.trust_weighting_version,
                "collective_outcome_reliability": self.collective_outcome_reliability,
            },
            "trend": self.trend,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CooperativeReliabilityProfile:
    agent_id: str
    latest: CooperativeReliabilitySnapshot
    history: List[CooperativeReliabilitySnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "latest": self.latest.to_dict(),
            "history": [snapshot.to_dict() for snapshot in self.history],
        }


class CooperativeReliabilityProfileGenerator:
    """
    Builds cooperative reliability profiles focused on collective-outcome improvement.
    Persists reliability evolution as ordered snapshots for each agent.
    """

    def __init__(
        self,
        history_window: int = 365,
        trust_weighting_function: Optional[TrustWeightingFunction] = None,
    ) -> None:
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.history_window = history_window
        self.trust_weighting_function = trust_weighting_function or TrustWeightingFunction()
        self._history_by_agent: Dict[str, Deque[CooperativeReliabilitySnapshot]] = defaultdict(
            lambda: deque(maxlen=self.history_window)
        )

    def generate_profile(
        self,
        agent_id: str,
        calibration_history: Optional[List[CalibrationRecord]],
        synergy_density_participation: float,
        marginal_cooperative_influence_consistency: float,
        collaborative_stability: float,
        long_term_impact_persistence: float = 0.5,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> CooperativeReliabilityProfile:
        calibration_consistency = self._calibration_consistency(calibration_history or [])
        clamped_synergy = _clamp01(synergy_density_participation)
        clamped_marginal = _clamp01(marginal_cooperative_influence_consistency)
        clamped_stability = _clamp01(collaborative_stability)
        clamped_persistence = _clamp01(long_term_impact_persistence)
        collective_outcome_reliability = self.trust_weighting_function.compute(
            predictive_accuracy_index=calibration_consistency,
            marginal_cooperative_influence=clamped_marginal,
            synergy_density_contribution=clamped_synergy,
            cooperative_stability_score=clamped_stability,
            long_term_impact_persistence=clamped_persistence,
        )

        prior_scores = [
            snapshot.collective_outcome_reliability
            for snapshot in self._history_by_agent.get(agent_id, deque())
        ]
        trend = self._classify_trend(prior_scores + [collective_outcome_reliability])

        snapshot = CooperativeReliabilitySnapshot(
            profile_id=str(uuid.uuid4()),
            agent_id=agent_id,
            generated_at=datetime.now(UTC),
            calibration_consistency=calibration_consistency,
            synergy_density_participation=clamped_synergy,
            marginal_cooperative_influence_consistency=clamped_marginal,
            collaborative_stability=clamped_stability,
            long_term_impact_persistence=clamped_persistence,
            trust_weighting_version=self.trust_weighting_function.version,
            collective_outcome_reliability=collective_outcome_reliability,
            trend=trend,
            evidence={
                **dict(evidence or {}),
                "trust_weighting": self.trust_weighting_function.parameters.to_dict(),
            },
        )

        self._history_by_agent[agent_id].append(snapshot)
        history = list(self._history_by_agent[agent_id])
        return CooperativeReliabilityProfile(agent_id=agent_id, latest=snapshot, history=history)

    def get_reliability_evolution(self, agent_id: str) -> List[Dict[str, Any]]:
        return [snapshot.to_dict() for snapshot in self._history_by_agent.get(agent_id, deque())]

    def _calibration_consistency(self, calibration_history: List[CalibrationRecord]) -> float:
        if not calibration_history:
            return 0.0

        per_record_scores: List[float] = []
        for record in calibration_history:
            horizon_scores: List[float] = []
            for metrics in record.per_horizon.values():
                aggregate_error = (
                    metrics.magnitude_deviation
                    + metrics.timing_deviation_hours
                    + metrics.synergy_assumption_error
                ) / 3.0
                horizon_scores.append(1.0 / (1.0 + aggregate_error))
            if horizon_scores:
                per_record_scores.append(mean(horizon_scores))
        return _bounded_average(per_record_scores)

    def _classify_trend(self, reliability_scores: List[float]) -> str:
        score_slope = _slope(reliability_scores[-5:])
        if score_slope > 0.01:
            return "improving"
        if score_slope < -0.01:
            return "declining"
        return "stable"
