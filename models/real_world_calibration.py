from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional
import uuid


Vector = Mapping[str, float]


def _euclidean_norm(values: Iterable[float]) -> float:
    return sqrt(sum(v * v for v in values))


def _vector_delta(predicted: Vector, realized: Vector) -> Dict[str, float]:
    keys = set(predicted.keys()) | set(realized.keys())
    return {k: float(realized.get(k, 0.0)) - float(predicted.get(k, 0.0)) for k in keys}


def _timing_deviation(
    predicted_event_times: Optional[Mapping[str, datetime]],
    realized_event_times: Optional[Mapping[str, datetime]],
) -> float:
    if not predicted_event_times or not realized_event_times:
        return 0.0

    shared = set(predicted_event_times.keys()) & set(realized_event_times.keys())
    if not shared:
        return 0.0

    deviations = [
        abs(
            (realized_event_times[event] - predicted_event_times[event]).total_seconds()
            / 3600.0
        )
        for event in shared
    ]
    return mean(deviations)


def _synergy_assumption_error(
    predicted_synergy: Optional[Vector],
    realized_synergy: Optional[Vector],
) -> float:
    if not predicted_synergy and not realized_synergy:
        return 0.0
    predicted_synergy = predicted_synergy or {}
    realized_synergy = realized_synergy or {}
    return _euclidean_norm(_vector_delta(predicted_synergy, realized_synergy).values())


@dataclass(frozen=True)
class HorizonCalibrationMetrics:
    horizon: str
    magnitude_deviation: float
    timing_deviation_hours: float
    synergy_assumption_error: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "magnitude_deviation": self.magnitude_deviation,
            "timing_deviation_hours": self.timing_deviation_hours,
            "synergy_assumption_error": self.synergy_assumption_error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CalibrationRecord:
    agent_id: str
    calibration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    per_horizon: Dict[str, HorizonCalibrationMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "calibration_id": self.calibration_id,
            "created_at": self.created_at.isoformat(),
            "per_horizon": {
                horizon: metrics.to_dict() for horizon, metrics in self.per_horizon.items()
            },
        }


class RealWorldCalibration:
    """
    Tracks rolling calibration histories by agent and computes temporal reliability curves.
    """

    def __init__(self, rolling_window_size: int = 200) -> None:
        if rolling_window_size < 1:
            raise ValueError("rolling_window_size must be >= 1")
        self.rolling_window_size = rolling_window_size
        self._history_by_agent: Dict[str, Deque[CalibrationRecord]] = defaultdict(
            lambda: deque(maxlen=self.rolling_window_size)
        )

    def record_calibration(
        self,
        agent_id: str,
        predicted_by_horizon: Mapping[str, Vector],
        realized_by_horizon: Mapping[str, Vector],
        predicted_event_times_by_horizon: Optional[Mapping[str, Mapping[str, datetime]]] = None,
        realized_event_times_by_horizon: Optional[Mapping[str, Mapping[str, datetime]]] = None,
        predicted_synergy_by_horizon: Optional[Mapping[str, Vector]] = None,
        realized_synergy_by_horizon: Optional[Mapping[str, Vector]] = None,
        metadata_by_horizon: Optional[Mapping[str, Dict[str, Any]]] = None,
    ) -> CalibrationRecord:
        horizons = set(predicted_by_horizon.keys()) | set(realized_by_horizon.keys())
        per_horizon: Dict[str, HorizonCalibrationMetrics] = {}

        for horizon in horizons:
            predicted = predicted_by_horizon.get(horizon, {})
            realized = realized_by_horizon.get(horizon, {})
            magnitude_deviation = _euclidean_norm(_vector_delta(predicted, realized).values())

            timing_dev = _timing_deviation(
                (predicted_event_times_by_horizon or {}).get(horizon),
                (realized_event_times_by_horizon or {}).get(horizon),
            )
            synergy_error = _synergy_assumption_error(
                (predicted_synergy_by_horizon or {}).get(horizon),
                (realized_synergy_by_horizon or {}).get(horizon),
            )

            per_horizon[horizon] = HorizonCalibrationMetrics(
                horizon=horizon,
                magnitude_deviation=magnitude_deviation,
                timing_deviation_hours=timing_dev,
                synergy_assumption_error=synergy_error,
                metadata=dict((metadata_by_horizon or {}).get(horizon, {})),
            )

        record = CalibrationRecord(agent_id=agent_id, per_horizon=per_horizon)
        self._history_by_agent[agent_id].append(record)
        return record

    def get_agent_history(self, agent_id: str) -> List[CalibrationRecord]:
        return list(self._history_by_agent.get(agent_id, []))

    def get_temporal_predictive_reliability_curve(
        self,
        agent_id: str,
        horizon: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        history = self._history_by_agent.get(agent_id, deque())
        curve: List[Dict[str, Any]] = []

        for record in history:
            horizons = [horizon] if horizon else list(record.per_horizon.keys())
            for hz in horizons:
                metrics = record.per_horizon.get(hz)
                if not metrics:
                    continue
                aggregate_error = (
                    metrics.magnitude_deviation
                    + metrics.timing_deviation_hours
                    + metrics.synergy_assumption_error
                ) / 3.0
                reliability = 1.0 / (1.0 + aggregate_error)
                curve.append(
                    {
                        "agent_id": agent_id,
                        "calibration_id": record.calibration_id,
                        "horizon": hz,
                        "timestamp": metrics.timestamp.isoformat(),
                        "reliability_score": reliability,
                        "magnitude_deviation": metrics.magnitude_deviation,
                        "timing_deviation_hours": metrics.timing_deviation_hours,
                        "synergy_assumption_error": metrics.synergy_assumption_error,
                    }
                )
        return curve

    def summarize_reliability_by_horizon(self, agent_id: str) -> Dict[str, Dict[str, float]]:
        curve = self.get_temporal_predictive_reliability_curve(agent_id=agent_id)
        grouped: Dict[str, List[float]] = defaultdict(list)
        for point in curve:
            grouped[point["horizon"]].append(float(point["reliability_score"]))

        summary: Dict[str, Dict[str, float]] = {}
        for horizon, values in grouped.items():
            summary[horizon] = {
                "mean_reliability": mean(values),
                "latest_reliability": values[-1],
                "samples": float(len(values)),
            }
        return summary

