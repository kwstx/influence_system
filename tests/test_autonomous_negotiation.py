from datetime import UTC, datetime, timedelta

from models.autonomous_negotiation import (
    TrustWeightedNegotiationParameters,
    TrustWeightedNegotiator,
    calibration_stability,
)
from models.cooperative_reliability_profile import (
    CooperativeReliabilityProfile,
    CooperativeReliabilitySnapshot,
)


def _snapshot(
    agent_id: str,
    calibration_consistency: float,
    collective_outcome_reliability: float,
    generated_at: datetime,
) -> CooperativeReliabilitySnapshot:
    return CooperativeReliabilitySnapshot(
        profile_id=f"snap-{agent_id}-{generated_at.timestamp()}",
        agent_id=agent_id,
        generated_at=generated_at,
        calibration_consistency=calibration_consistency,
        synergy_density_participation=0.5,
        marginal_cooperative_influence_consistency=0.5,
        collaborative_stability=0.5,
        collective_outcome_reliability=collective_outcome_reliability,
        trend="stable",
    )


def _profile(agent_id: str, snapshots: list[CooperativeReliabilitySnapshot]) -> CooperativeReliabilityProfile:
    return CooperativeReliabilityProfile(agent_id=agent_id, latest=snapshots[-1], history=list(snapshots))


def test_calibration_stability_penalizes_volatility():
    now = datetime.now(UTC)
    stable = _profile(
        "A",
        [
            _snapshot("A", 0.8, 0.8, now - timedelta(days=3)),
            _snapshot("A", 0.8, 0.8, now - timedelta(days=2)),
            _snapshot("A", 0.8, 0.8, now - timedelta(days=1)),
        ],
    )
    volatile = _profile(
        "B",
        [
            _snapshot("B", 0.2, 0.8, now - timedelta(days=3)),
            _snapshot("B", 0.9, 0.8, now - timedelta(days=2)),
            _snapshot("B", 0.1, 0.8, now - timedelta(days=1)),
        ],
    )

    assert calibration_stability(stable) > calibration_stability(volatile)


def test_trust_weighted_negotiation_caps_influence_and_logs():
    now = datetime.now(UTC)

    profiles = {
        "dominant": _profile(
            "dominant",
            [_snapshot("dominant", 0.9, 1.0, now - timedelta(hours=1)), _snapshot("dominant", 0.9, 1.0, now)],
        ),
        "peer1": _profile(
            "peer1",
            [_snapshot("peer1", 0.6, 0.2, now - timedelta(hours=1)), _snapshot("peer1", 0.6, 0.2, now)],
        ),
        "peer2": _profile(
            "peer2",
            [_snapshot("peer2", 0.6, 0.2, now - timedelta(hours=1)), _snapshot("peer2", 0.6, 0.2, now)],
        ),
    }

    negotiator = TrustWeightedNegotiator(
        TrustWeightedNegotiationParameters(
            max_influence_share=0.5,
            convergence_rate=0.5,
            max_delta_per_round=1.0,
            max_rounds=10,
            tolerance=1e-6,
        )
    )

    result = negotiator.negotiate(
        proposals_by_agent={"dominant": 1.0, "peer1": 0.0, "peer2": 0.0},
        profiles_by_agent=profiles,
    )

    assert result.weights["dominant"] <= 0.5 + 1e-9
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    assert len(result.audit_log) == result.rounds_executed * 3
    assert any(e.agent_id == "dominant" and e.cap_applied for e in result.audit_log)
    assert all(0.0 <= float(e.normalized_weight_post_cap) <= 1.0 for e in result.audit_log)


def test_settlement_matches_capped_weighted_mean_for_scalar_proposals():
    now = datetime.now(UTC)
    profiles = {
        "A": _profile("A", [_snapshot("A", 0.9, 1.0, now)]),
        "B": _profile("B", [_snapshot("B", 0.9, 0.1, now)]),
        "C": _profile("C", [_snapshot("C", 0.9, 0.1, now)]),
    }

    negotiator = TrustWeightedNegotiator(
        TrustWeightedNegotiationParameters(
            max_influence_share=0.5,
            convergence_rate=0.35,
            max_delta_per_round=0.5,
            max_rounds=25,
            tolerance=1e-4,
        )
    )

    result = negotiator.negotiate(
        proposals_by_agent={"A": 1.0, "B": 0.0, "C": 0.0},
        profiles_by_agent=profiles,
    )

    # With a max share of 0.5, A cannot pull the settlement above 0.5
    assert 0.0 <= float(result.settlement) <= 0.5 + 1e-3
    assert float(result.settlement) > 0.3

