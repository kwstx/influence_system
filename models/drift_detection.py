"""
Drift Detection Subsystem.

Identifies divergence between projected influence and realised downstream
impact over time.  When sustained deviation exceeds a configurable tolerance
threshold the agent's trust coefficient is smoothly decayed via an
exponential attenuation curve.  Decay is **reversible**: once new validated
observations demonstrate re-alignment, trust is gradually restored along a
sigmoid-gated recovery trajectory.

Key design decisions
--------------------
* **Exponential Moving Average (EMA)** for deviation smoothing – this gives
  recent observations more weight while retaining memory of the past, which
  prevents a single outlier from triggering decay.
* **Sustained deviation window** – decay only begins after a configurable
  number of consecutive above-threshold observations, protecting against
  transient spikes.
* **Smooth exponential decay** – trust is attenuated multiplicatively by a
  per-tick decay factor, producing a gradual curve rather than a cliff-edge.
* **Sigmoid-gated recovery** – when deviation drops back below threshold the
  trust coefficient is restored via a logistic curve anchored at the current
  value, ensuring recovery is equally smooth and monotonic.
* **Floor and ceiling** – trust never decays below a configurable minimum
  (keeping the agent in the system) and never exceeds 1.0.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log
from statistics import mean
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftObservation:
    """A single projected-vs-realised comparison at one point in time."""

    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    projected_influence: float = 0.0
    realised_impact: float = 0.0

    @property
    def absolute_deviation(self) -> float:
        return abs(self.projected_influence - self.realised_impact)

    @property
    def signed_deviation(self) -> float:
        """Positive means projection was optimistic, negative means pessimistic."""
        return self.projected_influence - self.realised_impact

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "projected_influence": self.projected_influence,
            "realised_impact": self.realised_impact,
            "absolute_deviation": self.absolute_deviation,
            "signed_deviation": self.signed_deviation,
        }


@dataclass
class TrustDecayState:
    """Mutable state tracking the trust decay/recovery lifecycle for one agent."""

    agent_id: str
    original_trust: float = 1.0
    current_trust: float = 1.0
    ema_deviation: float = 0.0
    consecutive_above_threshold: int = 0
    consecutive_below_threshold: int = 0
    is_decaying: bool = False
    decay_ticks: int = 0
    recovery_ticks: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "original_trust": self.original_trust,
            "current_trust": self.current_trust,
            "ema_deviation": self.ema_deviation,
            "consecutive_above_threshold": self.consecutive_above_threshold,
            "consecutive_below_threshold": self.consecutive_below_threshold,
            "is_decaying": self.is_decaying,
            "decay_ticks": self.decay_ticks,
            "recovery_ticks": self.recovery_ticks,
        }


@dataclass(frozen=True)
class DriftDetectionParameters:
    """Tunable knobs for the drift detector."""

    # -- Deviation smoothing --
    ema_alpha: float = 0.3
    """Weight for the latest observation in the EMA. Higher = more responsive."""

    # -- Threshold and patience --
    deviation_tolerance: float = 0.15
    """Absolute deviation (EMA-smoothed) above which drift is flagged."""

    sustained_ticks_to_decay: int = 3
    """Number of consecutive above-threshold ticks before decay kicks in."""

    sustained_ticks_to_recover: int = 2
    """Number of consecutive below-threshold ticks before recovery begins."""

    # -- Decay curve --
    decay_rate: float = 0.05
    """Per-tick multiplicative decay: trust *= (1 - decay_rate)."""

    min_trust: float = 0.1
    """Floor – trust never drops below this value."""

    # -- Recovery curve --
    recovery_rate: float = 0.08
    """Per-tick additive recovery step (scaled by sigmoid gate)."""

    max_trust: float = 1.0
    """Ceiling – trust never exceeds this value."""

    # -- Rolling observation window --
    window_size: int = 50
    """Maximum number of observations retained per agent."""

    def validate(self) -> None:
        """Raise ValueError if any parameter is out of valid range."""
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.deviation_tolerance < 0.0:
            raise ValueError("deviation_tolerance must be >= 0")
        if self.sustained_ticks_to_decay < 1:
            raise ValueError("sustained_ticks_to_decay must be >= 1")
        if self.sustained_ticks_to_recover < 1:
            raise ValueError("sustained_ticks_to_recover must be >= 1")
        if not 0.0 < self.decay_rate < 1.0:
            raise ValueError("decay_rate must be in (0, 1)")
        if not 0.0 <= self.min_trust < self.max_trust <= 1.0:
            raise ValueError("min_trust must be in [0, max_trust) and max_trust in (min_trust, 1]")
        if self.recovery_rate <= 0.0:
            raise ValueError("recovery_rate must be > 0")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Identifies divergence between projected influence and realised downstream
    impact, and adjusts an agent's trust coefficient accordingly.

    Usage
    -----
    1. Create a ``DriftDetector`` with desired parameters.
    2. For each agent, call :meth:`register_agent` with the agent's current
       trust coefficient.
    3. As new projected/realised pairs arrive, call :meth:`record_observation`.
       The method returns an updated ``TrustDecayState`` reflecting any trust
       adjustment.
    4. Read the current trust via :meth:`get_current_trust` or the full state
       via :meth:`get_state`.

    The detector is **stateful** – it maintains per-agent rolling windows,
    EMA deviation, and decay/recovery lifecycle state.
    """

    def __init__(self, parameters: Optional[DriftDetectionParameters] = None) -> None:
        self.params = parameters or DriftDetectionParameters()
        self.params.validate()
        self._states: Dict[str, TrustDecayState] = {}
        self._observations: Dict[str, Deque[DriftObservation]] = defaultdict(
            lambda: deque(maxlen=self.params.window_size)
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, initial_trust: float = 1.0) -> TrustDecayState:
        """Register an agent with its current trust coefficient."""
        clamped = _clamp(initial_trust, self.params.min_trust, self.params.max_trust)
        state = TrustDecayState(
            agent_id=agent_id,
            original_trust=clamped,
            current_trust=clamped,
        )
        self._states[agent_id] = state
        return state

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------

    def record_observation(
        self,
        agent_id: str,
        projected_influence: float,
        realised_impact: float,
        timestamp: Optional[datetime] = None,
    ) -> TrustDecayState:
        """
        Record a new projected-vs-realised pair and update the agent's trust.

        If the agent has not been registered, it is auto-registered with
        trust = 1.0.

        Returns the updated :class:`TrustDecayState`.
        """
        if agent_id not in self._states:
            self.register_agent(agent_id)

        obs = DriftObservation(
            agent_id=agent_id,
            projected_influence=projected_influence,
            realised_impact=realised_impact,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._observations[agent_id].append(obs)
        state = self._states[agent_id]

        # -- Update EMA deviation --
        alpha = self.params.ema_alpha
        state.ema_deviation = (
            alpha * obs.absolute_deviation + (1.0 - alpha) * state.ema_deviation
        )

        # -- Threshold check --
        above = state.ema_deviation > self.params.deviation_tolerance

        if above:
            state.consecutive_above_threshold += 1
            state.consecutive_below_threshold = 0
        else:
            state.consecutive_below_threshold += 1
            state.consecutive_above_threshold = 0

        # -- Transition: enter decay --
        if (
            not state.is_decaying
            and state.consecutive_above_threshold >= self.params.sustained_ticks_to_decay
        ):
            state.is_decaying = True
            state.decay_ticks = 0
            state.recovery_ticks = 0

        # -- Transition: leave decay (begin recovery) --
        if (
            state.is_decaying
            and state.consecutive_below_threshold >= self.params.sustained_ticks_to_recover
        ):
            state.is_decaying = False
            state.recovery_ticks = 0

        # -- Apply trust adjustment --
        if state.is_decaying:
            state.decay_ticks += 1
            self._apply_decay(state)
        elif state.current_trust < state.original_trust:
            state.recovery_ticks += 1
            self._apply_recovery(state)

        # -- Record snapshot --
        state.history.append(state.snapshot())
        return state

    # ------------------------------------------------------------------
    # Decay & recovery helpers
    # ------------------------------------------------------------------

    def _apply_decay(self, state: TrustDecayState) -> None:
        """Smooth exponential decay: trust *= (1 - decay_rate), floored at min_trust."""
        new_trust = state.current_trust * (1.0 - self.params.decay_rate)
        state.current_trust = max(self.params.min_trust, new_trust)

    def _apply_recovery(self, state: TrustDecayState) -> None:
        """
        Sigmoid-gated additive recovery.

        The recovery step is scaled by a logistic function of the number of
        recovery ticks so that initial recovery is gentle (avoiding sudden
        trust jumps) and then accelerates as sustained alignment is confirmed.

        gate(t) = 1 / (1 + exp(-0.5 * (t - 4)))

        At t=0 the gate ≈ 0.12, at t=4 it is 0.5, at t=8 it is ≈ 0.88.
        """
        t = state.recovery_ticks
        gate = 1.0 / (1.0 + exp(-0.5 * (t - 4)))
        step = self.params.recovery_rate * gate
        new_trust = state.current_trust + step
        state.current_trust = min(self.params.max_trust, min(state.original_trust, new_trust))

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_current_trust(self, agent_id: str) -> float:
        """Return the agent's current (possibly decayed) trust coefficient."""
        state = self._states.get(agent_id)
        if state is None:
            raise KeyError(f"Agent '{agent_id}' is not registered")
        return state.current_trust

    def get_state(self, agent_id: str) -> TrustDecayState:
        """Return the full mutable state object for an agent."""
        state = self._states.get(agent_id)
        if state is None:
            raise KeyError(f"Agent '{agent_id}' is not registered")
        return state

    def get_observations(self, agent_id: str) -> List[DriftObservation]:
        """Return the rolling observation window for an agent."""
        return list(self._observations.get(agent_id, []))

    def get_deviation_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Return a summary of the agent's deviation history.

        Includes EMA deviation, raw mean/max deviation over the window, and
        the current trust state.
        """
        state = self._states.get(agent_id)
        if state is None:
            raise KeyError(f"Agent '{agent_id}' is not registered")

        obs = list(self._observations.get(agent_id, []))
        deviations = [o.absolute_deviation for o in obs]

        return {
            "agent_id": agent_id,
            "observation_count": len(obs),
            "ema_deviation": state.ema_deviation,
            "mean_deviation": mean(deviations) if deviations else 0.0,
            "max_deviation": max(deviations) if deviations else 0.0,
            "current_trust": state.current_trust,
            "original_trust": state.original_trust,
            "trust_ratio": state.current_trust / state.original_trust if state.original_trust > 0 else 0.0,
            "is_decaying": state.is_decaying,
            "consecutive_above_threshold": state.consecutive_above_threshold,
            "consecutive_below_threshold": state.consecutive_below_threshold,
            "parameters": {
                "deviation_tolerance": self.params.deviation_tolerance,
                "sustained_ticks_to_decay": self.params.sustained_ticks_to_decay,
                "sustained_ticks_to_recover": self.params.sustained_ticks_to_recover,
                "decay_rate": self.params.decay_rate,
                "recovery_rate": self.params.recovery_rate,
            },
        }

    def get_all_agent_ids(self) -> List[str]:
        """Return the IDs of all registered agents."""
        return list(self._states.keys())

    def reset_agent(self, agent_id: str, new_trust: Optional[float] = None) -> TrustDecayState:
        """
        Reset an agent's drift state, optionally with a new baseline trust.

        This clears the observation window, EMA, and decay/recovery state.
        """
        state = self._states.get(agent_id)
        if state is None:
            raise KeyError(f"Agent '{agent_id}' is not registered")

        trust = _clamp(
            new_trust if new_trust is not None else state.original_trust,
            self.params.min_trust,
            self.params.max_trust,
        )
        state.original_trust = trust
        state.current_trust = trust
        state.ema_deviation = 0.0
        state.consecutive_above_threshold = 0
        state.consecutive_below_threshold = 0
        state.is_decaying = False
        state.decay_ticks = 0
        state.recovery_ticks = 0
        state.history.clear()

        # Clear observation window
        if agent_id in self._observations:
            self._observations[agent_id].clear()

        return state
