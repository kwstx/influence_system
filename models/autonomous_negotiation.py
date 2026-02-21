from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union
import uuid

from models.cooperative_reliability_profile import CooperativeReliabilityProfile


ScalarProposal = float
VectorProposal = Mapping[str, float]
Proposal = Union[ScalarProposal, VectorProposal]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_vector(proposal: Proposal) -> bool:
    return isinstance(proposal, Mapping)


def _to_vector(proposal: Proposal) -> Dict[str, float]:
    if isinstance(proposal, Mapping):
        return {str(k): float(v) for k, v in proposal.items()}
    return {"__scalar__": float(proposal)}


def _from_vector(vector: Mapping[str, float], template: Proposal) -> Proposal:
    if isinstance(template, Mapping):
        return {k: _clamp01(float(v)) for k, v in vector.items()}
    return _clamp01(float(vector.get("__scalar__", 0.0)))


def _vector_union_keys(vectors: List[Mapping[str, float]]) -> List[str]:
    keys: set[str] = set()
    for v in vectors:
        keys |= set(v.keys())
    return sorted(keys)


def _weighted_mean_vector(vectors: List[Mapping[str, float]], weights: List[float]) -> Dict[str, float]:
    if not vectors:
        return {"__scalar__": 0.0}
    keys = _vector_union_keys(vectors)
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(w * float(v.get(k, 0.0)) for w, v in zip(weights, vectors))
    return out


def _cap_and_normalize(weights: List[float], max_share: float) -> Tuple[List[float], List[float]]:
    """
    Returns (normalized_pre_cap, normalized_post_cap).
    """
    raw = [max(0.0, float(w)) for w in weights]
    total = sum(raw)
    if total <= 0.0:
        n = max(1, len(raw))
        equal = [1.0 / n] * n
        return equal, equal

    normalized = [w / total for w in raw]
    if not 0.0 < max_share <= 1.0:
        return normalized, normalized

    capped = list(normalized)
    for _ in range(len(capped)):
        excess = 0.0
        uncapped = 0
        for i, w in enumerate(capped):
            if w > max_share:
                excess += w - max_share
                capped[i] = max_share
            else:
                uncapped += 1
        if excess <= 0.0 or uncapped == 0:
            break
        redistribution = excess / uncapped
        for i in range(len(capped)):
            if capped[i] < max_share:
                capped[i] += redistribution

    cap_total = sum(capped)
    if cap_total > 0.0:
        capped = [w / cap_total for w in capped]
    return normalized, capped


def _mean_abs_step(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    return mean(diffs)


def calibration_stability(profile: CooperativeReliabilityProfile, window: int = 5) -> float:
    """
    Computes a calibration stability score in [0, 1].

    - Uses the agent's *calibration_consistency* time series.
    - Penalizes volatility (mean absolute step between snapshots).
    - Blends current calibration consistency with stability-of-calibration.
    """
    history = list(profile.history[-max(1, int(window)):])
    if not history:
        return 0.0

    consistencies = [_clamp01(s.calibration_consistency) for s in history]
    volatility = _clamp01(_mean_abs_step(consistencies))
    stability = _clamp01(1.0 - volatility)
    current = consistencies[-1]
    return _clamp01(0.5 * current + 0.5 * stability)


@dataclass(frozen=True)
class TrustWeightedNegotiationParameters:
    version: str = "1.0.0"
    max_influence_share: float = 0.4
    convergence_rate: float = 0.35
    max_delta_per_round: float = 0.25
    max_rounds: int = 25
    tolerance: float = 1e-4
    epsilon: float = 1e-9


@dataclass(frozen=True)
class NegotiationAuditEvent:
    audit_event_id: str
    created_at: datetime
    round_index: int
    agent_id: str
    trust_reliability: float
    calibration_stability: float
    raw_weight: float
    normalized_weight_pre_cap: float
    normalized_weight_post_cap: float
    cap_applied: bool
    consensus: Dict[str, float]
    proposal_before: Dict[str, float]
    proposal_after: Dict[str, float]
    delta: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_event_id": self.audit_event_id,
            "created_at": self.created_at.isoformat(),
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "trust_reliability": self.trust_reliability,
            "calibration_stability": self.calibration_stability,
            "raw_weight": self.raw_weight,
            "normalized_weight_pre_cap": self.normalized_weight_pre_cap,
            "normalized_weight_post_cap": self.normalized_weight_post_cap,
            "cap_applied": self.cap_applied,
            "consensus": dict(self.consensus),
            "proposal_before": dict(self.proposal_before),
            "proposal_after": dict(self.proposal_after),
            "delta": dict(self.delta),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrustWeightedNegotiationResult:
    settlement: Proposal
    rounds_executed: int
    converged: bool
    weights: Dict[str, float]
    audit_log: List[NegotiationAuditEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "settlement": self.settlement,
            "rounds_executed": self.rounds_executed,
            "converged": self.converged,
            "weights": dict(self.weights),
            "audit_log": [e.to_dict() for e in self.audit_log],
            "metadata": dict(self.metadata),
        }


class TrustWeightedNegotiator:
    """
    Trust-weighted cooperative bargaining with bounded influence.

    Each agent provides an initial proposal (scalar or vector). The negotiator:
    - Computes per-agent convergence weights from:
        (a) trust reliability (collective_outcome_reliability)
        (b) calibration stability (stability of calibration_consistency)
    - Uses those weights to compute a trust-weighted consensus proposal.
    - Iteratively moves each agent's proposal toward consensus.
    - Enforces a cap on any agent's normalized influence share.
    - Logs every trust-weighted adjustment for auditability.
    """

    def __init__(self, parameters: Optional[TrustWeightedNegotiationParameters] = None) -> None:
        self.parameters = parameters or TrustWeightedNegotiationParameters()

    def _compute_components(
        self,
        profile: Optional[CooperativeReliabilityProfile],
    ) -> Tuple[float, float, float]:
        if profile is None:
            return 0.0, 0.0, 0.0
        trust_rel = _clamp01(profile.latest.collective_outcome_reliability)
        calib_stab = calibration_stability(profile)
        raw_weight = max(float(self.parameters.epsilon), trust_rel * calib_stab)
        return trust_rel, calib_stab, raw_weight

    def negotiate(
        self,
        proposals_by_agent: Mapping[str, Proposal],
        profiles_by_agent: Mapping[str, CooperativeReliabilityProfile],
    ) -> TrustWeightedNegotiationResult:
        if not proposals_by_agent:
            return TrustWeightedNegotiationResult(
                settlement=0.0,
                rounds_executed=0,
                converged=True,
                weights={},
                audit_log=[],
                metadata={"status": "no_proposals"},
            )

        params = self.parameters
        if params.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if not 0.0 < params.convergence_rate <= 1.0:
            raise ValueError("convergence_rate must be in (0, 1]")
        if params.max_delta_per_round <= 0.0:
            raise ValueError("max_delta_per_round must be > 0")

        agent_ids = list(proposals_by_agent.keys())
        template = next(iter(proposals_by_agent.values()))
        proposals: MutableMapping[str, Dict[str, float]] = {
            agent_id: _to_vector(proposals_by_agent[agent_id]) for agent_id in agent_ids
        }

        trust_reliability: Dict[str, float] = {}
        calib_stability: Dict[str, float] = {}
        raw_weights: List[float] = []
        for agent_id in agent_ids:
            tr, cs, rw = self._compute_components(profiles_by_agent.get(agent_id))
            trust_reliability[agent_id] = tr
            calib_stability[agent_id] = cs
            raw_weights.append(rw)

        pre_cap, post_cap = _cap_and_normalize(raw_weights, params.max_influence_share)
        weights = {agent_id: float(w) for agent_id, w in zip(agent_ids, post_cap)}

        audit_log: List[NegotiationAuditEvent] = []
        converged = False
        settlement_vec: Dict[str, float] = _weighted_mean_vector(
            [proposals[a] for a in agent_ids], post_cap
        )

        for round_index in range(params.max_rounds):
            pre_cap, post_cap = _cap_and_normalize(raw_weights, params.max_influence_share)
            settlement_vec = _weighted_mean_vector(
                [proposals[a] for a in agent_ids], post_cap
            )

            max_residual = 0.0
            for i, agent_id in enumerate(agent_ids):
                before = dict(proposals[agent_id])
                after: Dict[str, float] = {}
                delta: Dict[str, float] = {}

                for k, consensus_value in settlement_vec.items():
                    current_value = float(before.get(k, 0.0))
                    step = float(params.convergence_rate) * (float(consensus_value) - current_value)
                    if step > params.max_delta_per_round:
                        step = params.max_delta_per_round
                    elif step < -params.max_delta_per_round:
                        step = -params.max_delta_per_round

                    new_value = _clamp01(current_value + step)
                    after[k] = new_value
                    delta[k] = new_value - current_value
                    max_residual = max(max_residual, abs(float(consensus_value) - new_value))

                proposals[agent_id] = after

                audit_log.append(
                    NegotiationAuditEvent(
                        audit_event_id=str(uuid.uuid4()),
                        created_at=datetime.now(UTC),
                        round_index=round_index,
                        agent_id=agent_id,
                        trust_reliability=trust_reliability[agent_id],
                        calibration_stability=calib_stability[agent_id],
                        raw_weight=float(raw_weights[i]),
                        normalized_weight_pre_cap=float(pre_cap[i]),
                        normalized_weight_post_cap=float(post_cap[i]),
                        cap_applied=abs(float(pre_cap[i]) - float(post_cap[i])) > 1e-12,
                        consensus=dict(settlement_vec),
                        proposal_before=before,
                        proposal_after=after,
                        delta=delta,
                        metadata={
                            "parameters_version": params.version,
                            "max_influence_share": params.max_influence_share,
                            "convergence_rate": params.convergence_rate,
                            "max_delta_per_round": params.max_delta_per_round,
                        },
                    )
                )

            if max_residual <= float(params.tolerance):
                converged = True
                break

        settlement = _from_vector(settlement_vec, template)
        return TrustWeightedNegotiationResult(
            settlement=settlement,
            rounds_executed=(round_index + 1),
            converged=converged,
            weights=weights,
            audit_log=audit_log,
            metadata={
                "parameters": {
                    "version": params.version,
                    "max_influence_share": params.max_influence_share,
                    "convergence_rate": params.convergence_rate,
                    "max_delta_per_round": params.max_delta_per_round,
                    "max_rounds": params.max_rounds,
                    "tolerance": params.tolerance,
                },
                "agent_count": len(agent_ids),
                "proposal_type": "vector" if _is_vector(template) else "scalar",
            },
        )

