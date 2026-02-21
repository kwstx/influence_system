from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import fsum
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from models.cooperative_reliability_profile import (
    CooperativeReliabilityProfile,
    CooperativeReliabilityProfileGenerator,
    TrustWeightingFunction,
)
from models.drift_detection import DriftDetector
from models.influence_entropy import InfluenceEntropy
from models.influence_projection import InfluenceProjector
from models.influence_signal import InfluenceSignal
from models.real_world_calibration import CalibrationRecord, RealWorldCalibration


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise(weights: Iterable[float]) -> List[float]:
    values = [max(0.0, float(w)) for w in weights]
    total = fsum(values)
    if total <= 0.0:
        if not values:
            return []
        equal = 1.0 / len(values)
        return [equal for _ in values]
    return [value / total for value in values]


def _error_response(status_code: int, message: str) -> Dict[str, Any]:
    return {
        "status_code": status_code,
        "body": {"error": {"code": status_code, "message": message}},
    }


def _tensor_representation(
    axis_names: Sequence[str],
    axis_labels: Sequence[Sequence[str]],
    values: Any,
) -> Dict[str, Any]:
    return {
        "type": "multi_dimensional_tensor",
        "axes": [
            {"name": name, "labels": list(labels)}
            for name, labels in zip(axis_names, axis_labels)
        ],
        "values": values,
    }


@dataclass
class AgentGovernanceState:
    agent_id: str
    trust_dimensions: Dict[str, float]
    recent_signals: List[InfluenceSignal] = field(default_factory=list)
    cohort_trust: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GovernanceAPI:
    """
    Governance-facing API handlers for trust and influence introspection.

    Endpoint paths:
    - /governance/agents/{agent_id}/trust-vector
    - /governance/agents/{agent_id}/influence-distribution
    - /governance/agents/{agent_id}/entropy-adjusted-weight
    - /governance/agents/{agent_id}/reliability-curve
    - /governance/agents/{agent_id}/calibration-history
    - /governance/agents/{agent_id}/drift-status
    """

    _TRUST_VECTOR_RE = re.compile(r"^/governance/agents/([^/]+)/trust-vector$")
    _INFLUENCE_RE = re.compile(r"^/governance/agents/([^/]+)/influence-distribution$")
    _ENTROPY_RE = re.compile(r"^/governance/agents/([^/]+)/entropy-adjusted-weight$")
    _RELIABILITY_RE = re.compile(r"^/governance/agents/([^/]+)/reliability-curve$")
    _CALIBRATION_RE = re.compile(r"^/governance/agents/([^/]+)/calibration-history$")
    _DRIFT_RE = re.compile(r"^/governance/agents/([^/]+)/drift-status$")

    def __init__(
        self,
        trust_weighting_function: Optional[TrustWeightingFunction] = None,
        projector: Optional[InfluenceProjector] = None,
        entropy: Optional[InfluenceEntropy] = None,
        drift_detector: Optional[DriftDetector] = None,
        calibration: Optional[RealWorldCalibration] = None,
    ) -> None:
        self._trust_weighting = trust_weighting_function or TrustWeightingFunction()
        self._reliability_generator = CooperativeReliabilityProfileGenerator(
            trust_weighting_function=self._trust_weighting
        )
        self._projector = projector or InfluenceProjector()
        self._entropy = entropy or InfluenceEntropy()
        self._drift = drift_detector or DriftDetector()
        self._calibration = calibration or RealWorldCalibration()
        self._agents: Dict[str, AgentGovernanceState] = {}

    def upsert_agent(
        self,
        agent_id: str,
        trust_dimensions: Mapping[str, float],
        recent_signals: Optional[List[InfluenceSignal]] = None,
        cohort_trust: Optional[Mapping[str, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        dims = {
            "predictive_accuracy_index": _clamp01(
                trust_dimensions.get("predictive_accuracy_index", 0.0)
            ),
            "marginal_cooperative_influence": _clamp01(
                trust_dimensions.get("marginal_cooperative_influence", 0.0)
            ),
            "synergy_density_contribution": _clamp01(
                trust_dimensions.get("synergy_density_contribution", 0.0)
            ),
            "cooperative_stability_score": _clamp01(
                trust_dimensions.get("cooperative_stability_score", 0.0)
            ),
            "long_term_impact_persistence": _clamp01(
                trust_dimensions.get("long_term_impact_persistence", 0.0)
            ),
        }
        state = AgentGovernanceState(
            agent_id=agent_id,
            trust_dimensions=dims,
            recent_signals=list(recent_signals or []),
            cohort_trust=dict(cohort_trust or {}),
            metadata=dict(metadata or {}),
        )
        self._agents[agent_id] = state
        trust = self._compute_trust_coefficient(dims)
        self._drift.register_agent(agent_id, initial_trust=trust)

    def record_calibration_event(
        self,
        agent_id: str,
        predicted_by_horizon: Mapping[str, Mapping[str, float]],
        realized_by_horizon: Mapping[str, Mapping[str, float]],
        metadata_by_horizon: Optional[Mapping[str, Dict[str, Any]]] = None,
    ) -> CalibrationRecord:
        return self._calibration.record_calibration(
            agent_id=agent_id,
            predicted_by_horizon=predicted_by_horizon,
            realized_by_horizon=realized_by_horizon,
            metadata_by_horizon=metadata_by_horizon,
        )

    def record_drift_observation(
        self, agent_id: str, projected_influence: float, realized_impact: float
    ) -> None:
        self._drift.record_observation(
            agent_id=agent_id,
            projected_influence=projected_influence,
            realised_impact=realized_impact,
        )

    def handle_request(self, method: str, raw_path: str) -> Dict[str, Any]:
        if method.upper() != "GET":
            return _error_response(405, "Only GET is supported")

        parsed = urlparse(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query)

        for regex, handler, supports_horizon in (
            (self._TRUST_VECTOR_RE, self.get_trust_vector, False),
            (self._INFLUENCE_RE, self.get_influence_distribution, False),
            (self._ENTROPY_RE, self.get_entropy_adjusted_weight, False),
            (self._RELIABILITY_RE, self.get_reliability_curve, True),
            (self._CALIBRATION_RE, self.get_calibration_history, False),
            (self._DRIFT_RE, self.get_drift_status, False),
        ):
            match = regex.match(path)
            if not match:
                continue
            agent_id = match.group(1)
            if supports_horizon:
                horizon = query.get("horizon", [None])[0]
                body = handler(agent_id=agent_id, horizon=horizon)
            else:
                body = handler(agent_id=agent_id)
            return {"status_code": 200, "body": body}

        return _error_response(404, f"Unknown path: {path}")

    def get_trust_vector(self, agent_id: str) -> Dict[str, Any]:
        state = self._require_agent(agent_id)
        dims = state.trust_dimensions
        trust = self._compute_trust_coefficient(dims)

        dimension_names = [
            "predictive_accuracy_index",
            "marginal_cooperative_influence",
            "synergy_density_contribution",
            "cooperative_stability_score",
            "long_term_impact_persistence",
            "trust_coefficient",
        ]
        values = [dims[name] for name in dimension_names[:-1]] + [trust]

        traces = []
        for name in dimension_names[:-1]:
            traces.append(
                {
                    "dimension": name,
                    "inputs": {"raw_value": dims[name]},
                    "transform": "clamp_to_unit_interval",
                    "formula": "max(0, min(1, raw_value))",
                    "output": dims[name],
                }
            )
        traces.append(
            {
                "dimension": "trust_coefficient",
                "inputs": {
                    "predictive_accuracy_index": dims["predictive_accuracy_index"],
                    "marginal_cooperative_influence": dims["marginal_cooperative_influence"],
                    "synergy_density_contribution": dims["synergy_density_contribution"],
                    "cooperative_stability_score": dims["cooperative_stability_score"],
                    "long_term_impact_persistence": dims["long_term_impact_persistence"],
                    "weights": self._trust_weighting.parameters.normalized_weights(),
                },
                "transform": "nonlinear_multiplicative_trust_weighting",
                "formula": "TrustWeightingFunction.compute(...)",
                "output": trust,
            }
        )

        return {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "representation": _tensor_representation(
                axis_names=["dimension"],
                axis_labels=[dimension_names],
                values=values,
            ),
            "causal_trace": {
                "model_version": self._trust_weighting.version,
                "dimension_traces": traces,
            },
        }

    def get_influence_distribution(self, agent_id: str) -> Dict[str, Any]:
        state = self._require_agent(agent_id)
        profile = self._build_reliability_profile(agent_id)
        trust = self._compute_trust_coefficient(state.trust_dimensions)
        projection = self._projector.project(
            agent_id=agent_id,
            reliability_profile=profile,
            recent_signals=state.recent_signals,
            trust_coefficient=trust,
        )

        return {
            "agent_id": agent_id,
            "timestamp": projection.timestamp.isoformat(),
            "representation": _tensor_representation(
                axis_names=["statistic", "bound"],
                axis_labels=[["projection", "confidence"], ["lower", "mean", "upper"]],
                values=[
                    [
                        projection.lower_bound,
                        projection.mean_projection,
                        projection.upper_bound,
                    ],
                    [
                        projection.confidence_score,
                        projection.confidence_score,
                        projection.confidence_score,
                    ],
                ],
            ),
            "causal_trace": {
                "factors": projection.metadata.get("factors", {}),
                "weights": projection.metadata.get("weights", {}),
                "propagation_scale": projection.metadata.get("propagation_scale"),
                "memory_layer": projection.metadata.get("memory_layer", {}),
            },
        }

    def get_entropy_adjusted_weight(self, agent_id: str) -> Dict[str, Any]:
        state = self._require_agent(agent_id)
        cohort_trust = dict(state.cohort_trust)
        cohort_trust[agent_id] = self._compute_trust_coefficient(state.trust_dimensions)

        agent_ids = list(cohort_trust.keys())
        trust_values = [cohort_trust[aid] for aid in agent_ids]
        base_weights = _normalise(trust_values)
        reliability_scores = [
            self._build_reliability_profile(aid).latest.collective_outcome_reliability
            if aid in self._agents
            else cohort_trust[aid]
            for aid in agent_ids
        ]

        adjusted_weights, entropy_snapshot = self._entropy.evaluate(
            agent_ids=agent_ids,
            trust_coefficients=trust_values,
            base_weights=base_weights,
            reliability_scores=reliability_scores,
        )

        idx = agent_ids.index(agent_id)
        per_agent_rows = [
            [base_weights[i], adjusted_weights[i], reliability_scores[i], trust_values[i]]
            for i in range(len(agent_ids))
        ]

        return {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "representation": _tensor_representation(
                axis_names=["agent_id", "metric"],
                axis_labels=[
                    agent_ids,
                    ["base_weight", "entropy_adjusted_weight", "reliability", "trust_coefficient"],
                ],
                values=per_agent_rows,
            ),
            "selected_agent": {
                "base_weight": base_weights[idx],
                "entropy_adjusted_weight": adjusted_weights[idx],
                "reliability": reliability_scores[idx],
                "trust_coefficient": trust_values[idx],
            },
            "causal_trace": {
                "entropy_snapshot": entropy_snapshot.to_dict(),
                "formula": "InfluenceEntropy.evaluate(agent_ids, trust_coefficients, base_weights, reliability_scores)",
            },
        }

    def get_reliability_curve(
        self, agent_id: str, horizon: Optional[str] = None
    ) -> Dict[str, Any]:
        self._require_agent(agent_id)
        curve = self._calibration.get_temporal_predictive_reliability_curve(
            agent_id=agent_id, horizon=horizon
        )

        timestamps = [point["timestamp"] for point in curve]
        values = [
            [
                point["reliability_score"],
                point["magnitude_deviation"],
                point["timing_deviation_hours"],
                point["synergy_assumption_error"],
            ]
            for point in curve
        ]

        return {
            "agent_id": agent_id,
            "horizon_filter": horizon,
            "representation": _tensor_representation(
                axis_names=["time_index", "metric"],
                axis_labels=[
                    [str(i) for i in range(len(curve))],
                    [
                        "reliability_score",
                        "magnitude_deviation",
                        "timing_deviation_hours",
                        "synergy_assumption_error",
                    ],
                ],
                values=values,
            ),
            "timeline": timestamps,
            "causal_trace": {
                "formula": "reliability_score = 1 / (1 + mean(magnitude_deviation, timing_deviation_hours, synergy_assumption_error))",
                "source": "RealWorldCalibration.get_temporal_predictive_reliability_curve",
            },
        }

    def get_calibration_history(self, agent_id: str) -> Dict[str, Any]:
        self._require_agent(agent_id)
        history = self._calibration.get_agent_history(agent_id)
        record_ids = [record.calibration_id for record in history]
        horizons = sorted(
            {
                horizon
                for record in history
                for horizon in record.per_horizon.keys()
            }
        )
        values = []
        for record in history:
            row = []
            for horizon in horizons:
                metrics = record.per_horizon.get(horizon)
                if metrics is None:
                    row.append([0.0, 0.0, 0.0])
                else:
                    row.append(
                        [
                            metrics.magnitude_deviation,
                            metrics.timing_deviation_hours,
                            metrics.synergy_assumption_error,
                        ]
                    )
            values.append(row)

        return {
            "agent_id": agent_id,
            "records": [record.to_dict() for record in history],
            "representation": {
                "type": "multi_dimensional_tensor",
                "axes": [
                    {"name": "calibration_record", "labels": record_ids},
                    {"name": "horizon", "labels": horizons},
                    {
                        "name": "metric",
                        "labels": [
                            "magnitude_deviation",
                            "timing_deviation_hours",
                            "synergy_assumption_error",
                        ],
                    },
                ],
                "values": values,
            },
            "causal_trace": {
                "formula": {
                    "magnitude_deviation": "euclidean_norm(realized_vector - predicted_vector)",
                    "timing_deviation_hours": "mean(abs(realized_time - predicted_time))",
                    "synergy_assumption_error": "euclidean_norm(realized_synergy - predicted_synergy)",
                },
                "source": "RealWorldCalibration.record_calibration",
            },
        }

    def get_drift_status(self, agent_id: str) -> Dict[str, Any]:
        self._require_agent(agent_id)
        summary = self._drift.get_deviation_summary(agent_id)
        state = self._drift.get_state(agent_id)

        return {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "representation": _tensor_representation(
                axis_names=["state_component", "metric"],
                axis_labels=[
                    ["trust", "deviation", "lifecycle"],
                    ["current", "baseline", "ema", "mean", "max", "above_ticks", "below_ticks"],
                ],
                values=[
                    [
                        summary["current_trust"],
                        summary["original_trust"],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        summary["ema_deviation"],
                        summary["mean_deviation"],
                        summary["max_deviation"],
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        float(summary["consecutive_above_threshold"]),
                        float(summary["consecutive_below_threshold"]),
                    ],
                ],
            ),
            "status": {
                "is_decaying": summary["is_decaying"],
                "observation_count": summary["observation_count"],
                "decay_ticks": state.decay_ticks,
                "recovery_ticks": state.recovery_ticks,
                "trust_ratio": summary["trust_ratio"],
            },
            "causal_trace": {
                "parameters": summary["parameters"],
                "ema_deviation": summary["ema_deviation"],
                "state_transitions": {
                    "enter_decay_when": "consecutive_above_threshold >= sustained_ticks_to_decay",
                    "enter_recovery_when": "consecutive_below_threshold >= sustained_ticks_to_recover",
                },
            },
        }

    def _require_agent(self, agent_id: str) -> AgentGovernanceState:
        state = self._agents.get(agent_id)
        if state is None:
            raise KeyError(f"Agent '{agent_id}' is not registered")
        return state

    def _compute_trust_coefficient(self, dimensions: Mapping[str, float]) -> float:
        return self._trust_weighting.compute(
            predictive_accuracy_index=dimensions["predictive_accuracy_index"],
            marginal_cooperative_influence=dimensions["marginal_cooperative_influence"],
            synergy_density_contribution=dimensions["synergy_density_contribution"],
            cooperative_stability_score=dimensions["cooperative_stability_score"],
            long_term_impact_persistence=dimensions["long_term_impact_persistence"],
        )

    def _build_reliability_profile(self, agent_id: str) -> CooperativeReliabilityProfile:
        state = self._require_agent(agent_id)
        calibration_history = self._calibration.get_agent_history(agent_id)
        return self._reliability_generator.generate_profile(
            agent_id=agent_id,
            calibration_history=calibration_history,
            synergy_density_participation=state.trust_dimensions["synergy_density_contribution"],
            marginal_cooperative_influence_consistency=state.trust_dimensions[
                "marginal_cooperative_influence"
            ],
            collaborative_stability=state.trust_dimensions["cooperative_stability_score"],
            long_term_impact_persistence=state.trust_dimensions["long_term_impact_persistence"],
            evidence={"source": "governance_api"},
        )
