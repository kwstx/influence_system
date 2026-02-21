from datetime import UTC, datetime, timedelta

from models.cooperative_reliability_profile import (
    CooperativeReliabilityProfileGenerator,
    TrustWeightingFunction,
    TrustWeightingParameters,
)
from models.real_world_calibration import RealWorldCalibration


def _seed_calibration_history(agent_id: str):
    calibration = RealWorldCalibration()
    now = datetime.now(UTC)
    calibration.record_calibration(
        agent_id=agent_id,
        predicted_by_horizon={"short": {"collective_gain": 1.0}},
        realized_by_horizon={"short": {"collective_gain": 0.9}},
        predicted_event_times_by_horizon={"short": {"rally": now}},
        realized_event_times_by_horizon={"short": {"rally": now + timedelta(minutes=30)}},
        predicted_synergy_by_horizon={"short": {"team_a_team_b": 0.8}},
        realized_synergy_by_horizon={"short": {"team_a_team_b": 0.75}},
    )
    calibration.record_calibration(
        agent_id=agent_id,
        predicted_by_horizon={"long": {"collective_gain": 2.0}},
        realized_by_horizon={"long": {"collective_gain": 2.1}},
        predicted_event_times_by_horizon={"long": {"handoff": now}},
        realized_event_times_by_horizon={"long": {"handoff": now + timedelta(hours=1)}},
        predicted_synergy_by_horizon={"long": {"team_a_team_c": 0.6}},
        realized_synergy_by_horizon={"long": {"team_a_team_c": 0.65}},
    )
    return calibration.get_agent_history(agent_id)


def test_generates_structured_profile_from_collective_metrics():
    agent_id = "agent_collective_1"
    history = _seed_calibration_history(agent_id)
    generator = CooperativeReliabilityProfileGenerator(history_window=10)

    profile = generator.generate_profile(
        agent_id=agent_id,
        calibration_history=history,
        synergy_density_participation=0.8,
        marginal_cooperative_influence_consistency=0.7,
        collaborative_stability=0.9,
        evidence={"cohort_id": "alpha"},
    )

    assert profile.agent_id == agent_id
    assert profile.latest.collective_outcome_reliability > 0.0
    assert 0.0 <= profile.latest.calibration_consistency <= 1.0
    assert profile.latest.evidence["cohort_id"] == "alpha"
    assert len(profile.history) == 1


def test_persists_reliability_evolution_over_time():
    agent_id = "agent_collective_2"
    history = _seed_calibration_history(agent_id)
    generator = CooperativeReliabilityProfileGenerator(history_window=10)

    profile_v1 = generator.generate_profile(
        agent_id=agent_id,
        calibration_history=history,
        synergy_density_participation=0.4,
        marginal_cooperative_influence_consistency=0.5,
        collaborative_stability=0.5,
    )
    profile_v2 = generator.generate_profile(
        agent_id=agent_id,
        calibration_history=history,
        synergy_density_participation=0.8,
        marginal_cooperative_influence_consistency=0.85,
        collaborative_stability=0.9,
    )

    evolution = generator.get_reliability_evolution(agent_id)
    assert len(evolution) == 2
    assert profile_v2.latest.collective_outcome_reliability >= profile_v1.latest.collective_outcome_reliability
    assert evolution[-1]["metrics"]["collective_outcome_reliability"] == profile_v2.latest.collective_outcome_reliability
    assert profile_v2.latest.trend in {"stable", "improving", "declining"}


def test_clamps_metric_inputs_to_collective_range():
    generator = CooperativeReliabilityProfileGenerator()
    profile = generator.generate_profile(
        agent_id="agent_collective_3",
        calibration_history=[],
        synergy_density_participation=2.2,
        marginal_cooperative_influence_consistency=-0.5,
        collaborative_stability=1.4,
    )

    assert profile.latest.synergy_density_participation == 1.0
    assert profile.latest.marginal_cooperative_influence_consistency == 0.0
    assert profile.latest.collaborative_stability == 1.0


def test_trust_weighting_function_is_continuous_and_versioned():
    fn = TrustWeightingFunction(
        TrustWeightingParameters(
            version="2.1.0",
            interaction_gain=0.2,
            second_order_gain=0.15,
        )
    )

    low = fn.compute(
        predictive_accuracy_index=0.35,
        marginal_cooperative_influence=0.35,
        synergy_density_contribution=0.35,
        cooperative_stability_score=0.35,
        long_term_impact_persistence=0.35,
    )
    high = fn.compute(
        predictive_accuracy_index=0.75,
        marginal_cooperative_influence=0.75,
        synergy_density_contribution=0.75,
        cooperative_stability_score=0.75,
        long_term_impact_persistence=0.75,
    )
    tiny_step = fn.compute(
        predictive_accuracy_index=0.351,
        marginal_cooperative_influence=0.35,
        synergy_density_contribution=0.35,
        cooperative_stability_score=0.35,
        long_term_impact_persistence=0.35,
    )

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low
    assert abs(tiny_step - low) < 0.01
    assert fn.version == "2.1.0"


def test_profile_uses_parameterized_trust_weighting():
    params = TrustWeightingParameters(
        version="3.0.0",
        predictive_weight=0.05,
        marginal_weight=0.05,
        synergy_weight=0.05,
        stability_weight=0.05,
        persistence_weight=0.8,
    )
    generator = CooperativeReliabilityProfileGenerator(
        trust_weighting_function=TrustWeightingFunction(params)
    )
    history = _seed_calibration_history("agent_collective_param")

    low_persistence = generator.generate_profile(
        agent_id="agent_collective_param",
        calibration_history=history,
        synergy_density_participation=0.8,
        marginal_cooperative_influence_consistency=0.8,
        collaborative_stability=0.8,
        long_term_impact_persistence=0.1,
    )
    high_persistence = generator.generate_profile(
        agent_id="agent_collective_param",
        calibration_history=history,
        synergy_density_participation=0.8,
        marginal_cooperative_influence_consistency=0.8,
        collaborative_stability=0.8,
        long_term_impact_persistence=0.9,
    )

    assert (
        high_persistence.latest.collective_outcome_reliability
        > low_persistence.latest.collective_outcome_reliability
    )
    assert high_persistence.latest.trust_weighting_version == "3.0.0"
    assert (
        high_persistence.latest.evidence["trust_weighting"]["version"]
        == "3.0.0"
    )
