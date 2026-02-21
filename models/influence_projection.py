from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log, sqrt
from statistics import mean
from typing import Dict, List, Optional, Any, Tuple
import uuid

from models.influence_signal import InfluenceSignal
from models.cooperative_reliability_profile import CooperativeReliabilityProfile, CooperativeReliabilitySnapshot
from models.influence_entropy import InfluenceEntropy


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))

@dataclass(frozen=True)
class InfluenceProjectionDistribution:
    """
    Represents a projected influence distribution with uncertainty bounds.
    """
    mean_projection: float
    lower_bound: float
    upper_bound: float
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_projection": self.mean_projection,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class InfluenceProjector:
    """
    Forecasts an agent's expected future impact contribution.

    When a *trust_coefficient* is supplied to :meth:`project`, it directly
    scales the causal propagation weights used in the projection.  A higher
    trust coefficient amplifies every weighted factor proportionally, giving
    high-trust agents greater influence over shared projections.  The
    coefficient acts as a *propagation intensity modifier* – it does **not**
    bypass or override multi-agent consensus constraints (those are enforced
    by :class:`CollaborativeProjectionAggregator`).
    """

    # Default propagation scaling bounds – prevents trust from zeroing out
    # or exploding an individual projection beyond interpretable limits.
    _MIN_PROPAGATION_SCALE: float = 0.1
    _MAX_PROPAGATION_SCALE: float = 2.0
    _MEMORY_DECAY_HALF_LIFE_STEPS: float = 6.0
    _MEMORY_DECAY_FLOOR: float = 0.2
    _REINFORCEMENT_DECAY_HALF_LIFE_STEPS: float = 3.0
    _COMPOUND_REINFORCEMENT_GAIN: float = 0.45
    _MAX_COMPOUND_MULTIPLIER: float = 2.2

    def __init__(
        self,
        weight_reliability: float = 0.4,
        weight_synergy: float = 0.3,
        weight_memory: float = 0.2,
        weight_slope: float = 0.1,
    ):
        self.weights = {
            "reliability": weight_reliability,
            "synergy": weight_synergy,
            "memory": weight_memory,
            "slope": weight_slope,
        }

    def project(
        self,
        agent_id: str,
        reliability_profile: CooperativeReliabilityProfile,
        recent_signals: List[InfluenceSignal],
        trust_coefficient: Optional[float] = None,
    ) -> InfluenceProjectionDistribution:
        """
        Computes the projected influence distribution.

        Parameters
        ----------
        trust_coefficient : float | None
            When provided (0-1 range, but any positive value is accepted),
            the trust coefficient is mapped to a *propagation scale factor*
            that multiplicatively adjusts the causal weight contributions.
            A trust of 1.0 maps to a scale of 1.0 (baseline); values above
            or below shift the scale proportionally, clamped to
            ``[_MIN_PROPAGATION_SCALE, _MAX_PROPAGATION_SCALE]``.
        """
        if not recent_signals:
            return self._neutral_projection()

        latest_snapshot = reliability_profile.latest

        # ------------------------------------------------------------------
        # Compute the propagation scale factor from the trust coefficient.
        # The mapping is: scale = 2 * trust  (so trust=0.5 → scale=1.0).
        # This means an agent with perfect trust (1.0) gets double the
        # propagation weight, while low-trust agents are attenuated.
        # ------------------------------------------------------------------
        if trust_coefficient is not None:
            raw_scale = 2.0 * max(0.0, float(trust_coefficient))
            propagation_scale = _clamp(
                raw_scale,
                self._MIN_PROPAGATION_SCALE,
                self._MAX_PROPAGATION_SCALE,
            )
        else:
            propagation_scale = 1.0  # legacy neutral

        # 1. Historical Reliability
        reliability = latest_snapshot.collective_outcome_reliability

        # 2. Synergy Signature Participation Strength
        synergy_strength = latest_snapshot.synergy_density_participation

        # 3. Temporal Impact Memory
        memory_values = [
            s.long_term_temporal_impact_weight.value for s in recent_signals
        ]
        avg_memory = mean(memory_values) if memory_values else 0.0

        # 4. Calibration Trend Slope
        history_scores = [
            s.collective_outcome_reliability
            for s in reliability_profile.history
        ]
        slope = self._calculate_slope(history_scores)
        clamped_slope = _clamp(slope, -0.5, 0.5)

        # ------------------------------------------------------------------
        # Weighted base projection – each factor is scaled by the trust-
        # derived propagation intensity so that high-trust agents produce
        # larger projections that carry more weight during aggregation.
        # ------------------------------------------------------------------
        base_projection = propagation_scale * (
            (self.weights["reliability"] * reliability)
            + (self.weights["synergy"] * synergy_strength)
            + (self.weights["memory"] * avg_memory)
        )

        # Slope influence is also scaled
        mean_proj = base_projection + propagation_scale * (
            self.weights["slope"] * clamped_slope
        )

        mean_proj = _clamp(mean_proj)

        # ------------------------------------------------------------------
        # Uncertainty bounds
        # ------------------------------------------------------------------
        uncertainty_base = 1.0 - reliability
        if len(memory_values) > 1:
            variance = mean([(x - avg_memory) ** 2 for x in memory_values])
            uncertainty_base = (uncertainty_base + sqrt(variance)) / 2.0

        bound_width = 0.2 * uncertainty_base + 0.05
        lower = max(0.0, mean_proj - bound_width)
        upper = min(1.0, mean_proj + bound_width)
        confidence = 1.0 - uncertainty_base

        return InfluenceProjectionDistribution(
            mean_projection=mean_proj,
            lower_bound=lower,
            upper_bound=upper,
            confidence_score=confidence,
            metadata={
                "agent_id": agent_id,
                "trust_coefficient": trust_coefficient,
                "propagation_scale": propagation_scale,
                "factors": {
                    "reliability": reliability,
                    "synergy": synergy_strength,
                    "memory": avg_memory,
                    "slope": slope,
                },
                "weights": self.weights,
            },
        )

    def _calculate_slope(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        return mean(diffs)

    def _neutral_projection(self) -> InfluenceProjectionDistribution:
        return InfluenceProjectionDistribution(
            mean_projection=0.5,
            lower_bound=0.25,
            upper_bound=0.75,
            confidence_score=0.0,
            metadata={"status": "insufficient_data"},
        )


# ---------------------------------------------------------------------------
# Collaborative multi-agent aggregation with consensus constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentProjectionEntry:
    """Bundles an individual projection with the originating agent's trust."""

    agent_id: str
    trust_coefficient: float
    projection: InfluenceProjectionDistribution


class CollaborativeProjectionAggregator:
    """
    Merges projections from multiple agents into a single shared projection
    for a collaborative task.

    **Trust-weighted influence**: each agent's projection mean is weighted by
    its trust coefficient so that high-trust agents have proportionally
    greater pull on the consensus result.

    **Consensus constraints**: to prevent any single agent – regardless of
    trust – from dominating the shared projection:

    * ``max_trust_share``: no agent's *normalized* weight can exceed this
      fraction of the total weight.  Excess weight is redistributed evenly
      across the remaining agents.
    * ``min_agents_for_consensus``: aggregation requires at least this many
      contributing agents; otherwise the result is flagged as
      *below_consensus_threshold*.

    Trust therefore modifies propagation intensity (via the weights) without
    overriding the multi-agent consensus guarantee.
    """

    def __init__(
        self,
        max_trust_share: float = 0.4,
        min_agents_for_consensus: int = 2,
        entropy_threshold: float = 0.75,
        entropy_regularization_strength: float = 0.35,
        opportunity_boost_strength: float = 0.30,
    ):
        if not 0.0 < max_trust_share <= 1.0:
            raise ValueError("max_trust_share must be in (0, 1]")
        if min_agents_for_consensus < 1:
            raise ValueError("min_agents_for_consensus must be >= 1")
        self.max_trust_share = max_trust_share
        self.min_agents_for_consensus = min_agents_for_consensus
        self._influence_entropy = InfluenceEntropy(
            entropy_threshold=entropy_threshold,
            dominance_softness=entropy_regularization_strength,
            opportunity_boost_strength=opportunity_boost_strength,
        )

    # ------------------------------------------------------------------
    # Weight normalisation with consensus cap
    # ------------------------------------------------------------------

    def _compute_capped_weights(
        self, entries: List[AgentProjectionEntry]
    ) -> List[float]:
        """Return normalised weights with the consensus cap applied."""
        raw = [max(0.0, float(e.trust_coefficient)) for e in entries]
        total = sum(raw)
        if total <= 0.0:
            # Equal fallback when all trusts are zero
            n = len(entries)
            return [1.0 / n] * n

        normalised = [w / total for w in raw]

        # Iteratively cap weights that exceed max_trust_share and
        # redistribute the excess to uncapped agents.
        capped = list(normalised)
        for _ in range(len(entries)):
            excess = 0.0
            uncapped_count = 0
            for i, w in enumerate(capped):
                if w > self.max_trust_share:
                    excess += w - self.max_trust_share
                    capped[i] = self.max_trust_share
                else:
                    uncapped_count += 1
            if excess <= 0.0 or uncapped_count == 0:
                break
            redistribution = excess / uncapped_count
            for i in range(len(capped)):
                if capped[i] < self.max_trust_share:
                    capped[i] += redistribution

        # Final renormalisation to ensure sum == 1.0
        cap_total = sum(capped)
        if cap_total > 0.0:
            capped = [w / cap_total for w in capped]
        return capped

    # ------------------------------------------------------------------
    # Public aggregation interface
    # ------------------------------------------------------------------

    def aggregate(
        self, entries: List[AgentProjectionEntry]
    ) -> InfluenceProjectionDistribution:
        """Produce a consensus-constrained, trust-weighted shared projection."""
        if not entries:
            return InfluenceProjectionDistribution(
                mean_projection=0.5,
                lower_bound=0.25,
                upper_bound=0.75,
                confidence_score=0.0,
                metadata={"status": "no_entries"},
            )

        below_consensus = len(entries) < self.min_agents_for_consensus
        base_weights = self._compute_capped_weights(entries)

        weights, entropy_snapshot = self._influence_entropy.evaluate(
            agent_ids=[entry.agent_id for entry in entries],
            trust_coefficients=[entry.trust_coefficient for entry in entries],
            base_weights=base_weights,
            reliability_scores=[
                entry.projection.confidence_score for entry in entries
            ],
        )

        # Trust-weighted mean projection
        w_mean = sum(
            w * e.projection.mean_projection for w, e in zip(weights, entries)
        )

        # Trust-weighted bounds (conservative: widest plausible range)
        w_lower = sum(
            w * e.projection.lower_bound for w, e in zip(weights, entries)
        )
        w_upper = sum(
            w * e.projection.upper_bound for w, e in zip(weights, entries)
        )

        # Trust-weighted confidence
        w_confidence = sum(
            w * e.projection.confidence_score for w, e in zip(weights, entries)
        )

        # Build per-agent contribution detail for auditability
        contributions = []
        for w, e in zip(weights, entries):
            contributions.append(
                {
                    "agent_id": e.agent_id,
                    "trust_coefficient": e.trust_coefficient,
                    "normalised_weight": round(w, 6),
                    "individual_mean": e.projection.mean_projection,
                }
            )

        return InfluenceProjectionDistribution(
            mean_projection=_clamp(w_mean),
            lower_bound=_clamp(w_lower),
            upper_bound=_clamp(w_upper),
            confidence_score=_clamp(w_confidence),
            metadata={
                "aggregation": "trust_weighted_consensus",
                "agent_count": len(entries),
                "max_trust_share": self.max_trust_share,
                "below_consensus_threshold": below_consensus,
                "entropy_regularization": entropy_snapshot.to_dict(),
                "contributions": contributions,
            },
        )
