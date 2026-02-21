"""
Trust-Aware Task Formation Engine.

Biases agent selection toward combinations with high projected synergy
density and stable trust interaction history.  Uses Shannon-entropy
constraints on team composition to maintain diversity and prevent
over-centralization of high-trust agents.  Clustering heuristics are
continuously updated based on realized cooperative outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from math import exp, log, sqrt
from statistics import mean, stdev
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import uuid


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _shannon_entropy(weights: List[float]) -> float:
    """Compute normalised Shannon entropy over a probability distribution.

    Returns a value in [0, 1] where 1 means maximum diversity (uniform)
    and 0 means all weight concentrated on a single element.
    """
    if not weights:
        return 0.0
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    probs = [w / total for w in weights if w > 0]
    if len(probs) <= 1:
        return 0.0
    raw = -sum(p * log(p) for p in probs)
    max_entropy = log(len(probs))
    if max_entropy <= 0.0:
        return 0.0
    return raw / max_entropy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentTrustRecord:
    """Snapshot of an agent's trust-relevant metrics for task formation."""

    agent_id: str
    trust_coefficient: float          # [0, 1]
    synergy_density: float            # [0, 1]  average synergy with prior partners
    cooperative_stability: float      # [0, 1]  variance-based stability
    calibration_accuracy: float       # [0, 1]
    long_term_persistence: float      # [0, 1]
    interaction_count: int = 0        # total cooperative interactions recorded
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseSynergyScore:
    """Projected synergy density between two agents."""

    agent_a: str
    agent_b: str
    synergy_density: float            # [0, 1]
    trust_stability: float            # [0, 1]  stability of mutual trust history
    interaction_history_depth: int    # number of past joint interactions
    combined_score: float             # weighted combination


@dataclass(frozen=True)
class TeamCandidate:
    """A scored candidate team configuration."""

    team_id: str
    agent_ids: FrozenSet[str]
    mean_synergy_density: float
    trust_stability_score: float
    entropy_score: float              # diversity measure
    composite_score: float            # final weighted score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeRecord:
    """Realised cooperative outcome for heuristic refinement."""

    team_id: str
    agent_ids: FrozenSet[str]
    predicted_synergy: float
    realised_synergy: float
    predicted_stability: float
    realised_stability: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class TrustAwareTaskFormationEngine:
    """Assembles candidate teams biased toward high projected synergy density
    and stable trust interaction history.

    **Entropy constraints** ensure that teams are not over-populated with
    the same high-trust agents across many tasks, maintaining structural
    diversity.

    **Heuristic updates** recalibrate the internal weight profile used for
    scoring whenever realised outcome records are fed back into the engine.

    Parameters
    ----------
    weight_synergy : float
        Importance of projected synergy density in composite scoring.
    weight_stability : float
        Importance of trust interaction stability.
    weight_entropy : float
        Importance of team diversity (entropy).
    min_entropy : float
        Minimum normalised entropy required for a team to be considered
        viable.  Teams below this threshold are discarded.
    max_agent_reuse_share : float
        Upper-bound on how often a single agent can appear across
        recommended teams (prevents over-centralisation).
    learning_rate : float
        How aggressively outcome feedback shifts internal weights.
    """

    def __init__(
        self,
        weight_synergy: float = 0.45,
        weight_stability: float = 0.30,
        weight_entropy: float = 0.25,
        min_entropy: float = 0.3,
        max_agent_reuse_share: float = 0.5,
        learning_rate: float = 0.1,
    ):
        if not (0.0 < min_entropy <= 1.0):
            raise ValueError("min_entropy must be in (0, 1]")
        if not (0.0 < max_agent_reuse_share <= 1.0):
            raise ValueError("max_agent_reuse_share must be in (0, 1]")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")

        self._w_synergy = weight_synergy
        self._w_stability = weight_stability
        self._w_entropy = weight_entropy
        self._min_entropy = min_entropy
        self._max_agent_reuse_share = max_agent_reuse_share
        self._learning_rate = learning_rate

        # Outcome history for heuristic recalibration
        self._outcome_history: List[OutcomeRecord] = []
        # Pairwise synergy cache (updated by outcomes)
        self._synergy_adjustments: Dict[FrozenSet[str], float] = {}

    # ------------------------------------------------------------------
    # Properties for introspection / testing
    # ------------------------------------------------------------------

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "synergy": self._w_synergy,
            "stability": self._w_stability,
            "entropy": self._w_entropy,
        }

    @property
    def min_entropy(self) -> float:
        return self._min_entropy

    @property
    def max_agent_reuse_share(self) -> float:
        return self._max_agent_reuse_share

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def outcome_count(self) -> int:
        return len(self._outcome_history)

    # ------------------------------------------------------------------
    # Pairwise synergy scoring
    # ------------------------------------------------------------------

    def compute_pairwise_synergy(
        self,
        agent_a: AgentTrustRecord,
        agent_b: AgentTrustRecord,
        joint_interaction_count: int = 0,
    ) -> PairwiseSynergyScore:
        """Compute projected synergy density between two agents.

        The synergy density is the geometric mean of both agents' individual
        synergy densities, amplified by mutual trust and calibration overlap.
        Trust stability is derived from the harmonic mean of cooperative
        stability scores, penalised when interaction depth is shallow.
        """
        # Geometric mean of individual synergy densities
        raw_synergy = sqrt(
            max(0.0, agent_a.synergy_density)
            * max(0.0, agent_b.synergy_density)
        )

        # Trust-calibration overlap amplifier
        trust_overlap = (agent_a.trust_coefficient + agent_b.trust_coefficient) / 2.0
        calibration_overlap = (
            agent_a.calibration_accuracy + agent_b.calibration_accuracy
        ) / 2.0
        amplifier = 1.0 + 0.3 * trust_overlap * calibration_overlap

        synergy_density = _clamp(raw_synergy * amplifier)

        # Apply learned adjustment from past outcomes
        pair_key = frozenset([agent_a.agent_id, agent_b.agent_id])
        adjustment = self._synergy_adjustments.get(pair_key, 0.0)
        synergy_density = _clamp(synergy_density + adjustment)

        # Trust interaction stability (harmonic mean, depth-penalised)
        sa = max(1e-9, agent_a.cooperative_stability)
        sb = max(1e-9, agent_b.cooperative_stability)
        harmonic = 2.0 * sa * sb / (sa + sb)

        # Depth discount: shallow interaction history → reduced confidence
        depth_factor = 1.0 - exp(-0.3 * joint_interaction_count)
        trust_stability = _clamp(harmonic * (0.5 + 0.5 * depth_factor))

        # Combined score
        combined = (
            self._w_synergy * synergy_density
            + self._w_stability * trust_stability
        ) / (self._w_synergy + self._w_stability)

        return PairwiseSynergyScore(
            agent_a=agent_a.agent_id,
            agent_b=agent_b.agent_id,
            synergy_density=synergy_density,
            trust_stability=trust_stability,
            interaction_history_depth=joint_interaction_count,
            combined_score=_clamp(combined),
        )

    # ------------------------------------------------------------------
    # Team scoring
    # ------------------------------------------------------------------

    def score_team(
        self,
        agents: List[AgentTrustRecord],
        pairwise_interactions: Optional[Dict[FrozenSet[str], int]] = None,
    ) -> TeamCandidate:
        """Score a specific team of agents.

        Returns a ``TeamCandidate`` with synergy, stability, entropy and
        composite scores.  The team may be rejected (low composite) if its
        entropy falls below ``min_entropy``.
        """
        if len(agents) < 2:
            raise ValueError("A team requires at least 2 agents")

        pairwise_interactions = pairwise_interactions or {}

        # Compute all pairwise synergy scores
        pairwise_scores: List[PairwiseSynergyScore] = []
        for a, b in combinations(agents, 2):
            pair_key = frozenset([a.agent_id, b.agent_id])
            depth = pairwise_interactions.get(pair_key, 0)
            pairwise_scores.append(
                self.compute_pairwise_synergy(a, b, depth)
            )

        mean_synergy = mean(ps.synergy_density for ps in pairwise_scores)
        mean_stability = mean(ps.trust_stability for ps in pairwise_scores)

        # Entropy of trust coefficients within the team
        trust_weights = [max(0.0, a.trust_coefficient) for a in agents]
        entropy = _shannon_entropy(trust_weights)

        # Entropy penalty: if below threshold, the composite score is
        # heavily penalised to discourage homogeneous high-trust teams.
        entropy_penalty = 1.0
        if entropy < self._min_entropy:
            entropy_penalty = entropy / self._min_entropy  # soft ramp

        composite = entropy_penalty * (
            self._w_synergy * mean_synergy
            + self._w_stability * mean_stability
            + self._w_entropy * entropy
        )

        agent_ids = frozenset(a.agent_id for a in agents)

        return TeamCandidate(
            team_id=str(uuid.uuid4()),
            agent_ids=agent_ids,
            mean_synergy_density=round(mean_synergy, 6),
            trust_stability_score=round(mean_stability, 6),
            entropy_score=round(entropy, 6),
            composite_score=round(_clamp(composite), 6),
            metadata={
                "pairwise_count": len(pairwise_scores),
                "entropy_penalty_applied": entropy_penalty < 1.0,
                "weights": self.weights,
            },
        )

    # ------------------------------------------------------------------
    # Team recommendation (enumeration + filtering)
    # ------------------------------------------------------------------

    def recommend_teams(
        self,
        agents: List[AgentTrustRecord],
        team_size: int,
        top_k: int = 5,
        pairwise_interactions: Optional[Dict[FrozenSet[str], int]] = None,
    ) -> List[TeamCandidate]:
        """Enumerate all possible teams of ``team_size`` from the agent pool
        and return the top-k by composite score, subject to entropy and
        reuse constraints.

        For large pools this is combinatorially expensive; in production a
        heuristic pre-filter or sampling strategy should be used.
        """
        if team_size < 2:
            raise ValueError("team_size must be >= 2")
        if team_size > len(agents):
            raise ValueError("team_size cannot exceed the number of agents")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        candidates: List[TeamCandidate] = []
        for combo in combinations(agents, team_size):
            candidate = self.score_team(
                list(combo), pairwise_interactions
            )
            # Entropy gate: discard teams below diversity threshold
            if candidate.entropy_score >= self._min_entropy:
                candidates.append(candidate)

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # Agent-reuse constraint: greedily select top-k while ensuring no
        # agent appears in more than max_agent_reuse_share of selected teams.
        selected: List[TeamCandidate] = []
        agent_counts: Dict[str, int] = {}
        max_appearances = max(1, int(top_k * self._max_agent_reuse_share))

        for candidate in candidates:
            if len(selected) >= top_k:
                break
            # Check reuse limits
            if all(
                agent_counts.get(aid, 0) < max_appearances
                for aid in candidate.agent_ids
            ):
                selected.append(candidate)
                for aid in candidate.agent_ids:
                    agent_counts[aid] = agent_counts.get(aid, 0) + 1

        return selected

    # ------------------------------------------------------------------
    # Outcome feedback & heuristic recalibration
    # ------------------------------------------------------------------

    def record_outcome(self, outcome: OutcomeRecord) -> None:
        """Feed a realised cooperative outcome back into the engine.

        The engine uses the prediction-vs-realisation delta to:
        1. Adjust pairwise synergy bias terms for the team's agent pairs.
        2. Shift the internal weight profile toward dimensions that better
           predicted the realised outcome.
        """
        self._outcome_history.append(outcome)

        # 1. Pairwise synergy bias adjustment
        synergy_error = outcome.realised_synergy - outcome.predicted_synergy
        for pair in combinations(sorted(outcome.agent_ids), 2):
            pair_key = frozenset(pair)
            current = self._synergy_adjustments.get(pair_key, 0.0)
            # Exponential moving average toward the error
            updated = current + self._learning_rate * (synergy_error - current)
            self._synergy_adjustments[pair_key] = _clamp(updated, -0.5, 0.5)

        # 2. Weight recalibration based on which dimension was more
        #    predictive of the realised outcome.
        stability_error = abs(
            outcome.realised_stability - outcome.predicted_stability
        )
        synergy_abs_error = abs(synergy_error)

        # If synergy prediction was more accurate, increase synergy weight
        # relative to stability (and vice versa). Entropy weight is kept
        # stable as a structural constraint.
        if synergy_abs_error + stability_error > 0:
            # Lower error means higher accuracy; invert for weight signal
            synergy_accuracy = 1.0 / (1.0 + synergy_abs_error)
            stability_accuracy = 1.0 / (1.0 + stability_error)
            total_accuracy = synergy_accuracy + stability_accuracy

            target_syn = synergy_accuracy / total_accuracy
            target_stab = stability_accuracy / total_accuracy

            # Blend toward target allocation (keeping entropy weight fixed)
            non_entropy_budget = 1.0 - self._w_entropy
            self._w_synergy += self._learning_rate * (
                target_syn * non_entropy_budget - self._w_synergy
            )
            self._w_stability += self._learning_rate * (
                target_stab * non_entropy_budget - self._w_stability
            )

            # Renormalise to keep total = 1.0
            total = self._w_synergy + self._w_stability + self._w_entropy
            if total > 0:
                self._w_synergy /= total
                self._w_stability /= total
                self._w_entropy /= total

    def get_synergy_adjustment(self, agent_a: str, agent_b: str) -> float:
        """Return the learned synergy bias for a specific agent pair."""
        return self._synergy_adjustments.get(
            frozenset([agent_a, agent_b]), 0.0
        )

    def get_weight_history_snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the engine's current state."""
        return {
            "weights": self.weights,
            "outcome_count": self.outcome_count,
            "synergy_adjustments": {
                str(sorted(k)): round(v, 6)
                for k, v in self._synergy_adjustments.items()
            },
            "min_entropy": self._min_entropy,
            "max_agent_reuse_share": self._max_agent_reuse_share,
            "learning_rate": self._learning_rate,
        }
