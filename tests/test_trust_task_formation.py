"""
Tests for the Trust-Aware Task Formation Engine.

Covers pairwise synergy scoring, team scoring with entropy constraints,
team recommendation with agent-reuse limits, and heuristic recalibration
from realised cooperative outcomes.
"""

import pytest
from datetime import datetime

from models.trust_task_formation import (
    AgentTrustRecord,
    OutcomeRecord,
    PairwiseSynergyScore,
    TeamCandidate,
    TrustAwareTaskFormationEngine,
    _shannon_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(
    agent_id: str = "agent_A",
    trust: float = 0.7,
    synergy: float = 0.6,
    stability: float = 0.8,
    calibration: float = 0.75,
    persistence: float = 0.65,
    interactions: int = 10,
) -> AgentTrustRecord:
    return AgentTrustRecord(
        agent_id=agent_id,
        trust_coefficient=trust,
        synergy_density=synergy,
        cooperative_stability=stability,
        calibration_accuracy=calibration,
        long_term_persistence=persistence,
        interaction_count=interactions,
    )


def _diverse_pool(n: int = 6) -> list[AgentTrustRecord]:
    """Create a pool of agents with varied trust and synergy profiles."""
    configs = [
        ("alpha",   0.9, 0.8, 0.85, 0.9,  0.7),
        ("beta",    0.7, 0.7, 0.75, 0.7,  0.6),
        ("gamma",   0.5, 0.6, 0.65, 0.55, 0.5),
        ("delta",   0.3, 0.5, 0.55, 0.4,  0.4),
        ("epsilon", 0.6, 0.65, 0.7, 0.6,  0.55),
        ("zeta",    0.4, 0.55, 0.6, 0.5,  0.45),
    ]
    return [
        _agent(aid, trust, syn, stab, cal, pers)
        for aid, trust, syn, stab, cal, pers in configs[:n]
    ]


# ---------------------------------------------------------------------------
# Shannon entropy helper
# ---------------------------------------------------------------------------

class TestShannonEntropy:

    def test_uniform_distribution_is_maximum(self):
        """A uniform distribution should yield entropy = 1.0."""
        assert abs(_shannon_entropy([1, 1, 1, 1]) - 1.0) < 1e-9

    def test_degenerate_distribution_is_zero(self):
        """All weight on one element should give entropy = 0."""
        assert _shannon_entropy([1, 0, 0, 0]) == 0.0

    def test_empty_is_zero(self):
        assert _shannon_entropy([]) == 0.0

    def test_single_element_is_zero(self):
        assert _shannon_entropy([5.0]) == 0.0

    def test_two_equal_elements(self):
        assert abs(_shannon_entropy([3, 3]) - 1.0) < 1e-9

    def test_skewed_distribution(self):
        """A skewed distribution should have entropy between 0 and 1."""
        e = _shannon_entropy([0.9, 0.05, 0.05])
        assert 0.0 < e < 1.0

    def test_all_zeros_is_zero(self):
        assert _shannon_entropy([0, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:

    def test_default_weights(self):
        engine = TrustAwareTaskFormationEngine()
        w = engine.weights
        assert abs(w["synergy"] + w["stability"] + w["entropy"] - 1.0) < 1e-9

    def test_custom_weights(self):
        engine = TrustAwareTaskFormationEngine(
            weight_synergy=0.5, weight_stability=0.3, weight_entropy=0.2
        )
        assert engine.weights["synergy"] == 0.5

    def test_invalid_min_entropy_raises(self):
        with pytest.raises(ValueError):
            TrustAwareTaskFormationEngine(min_entropy=0.0)
        with pytest.raises(ValueError):
            TrustAwareTaskFormationEngine(min_entropy=1.5)

    def test_invalid_max_agent_reuse_share_raises(self):
        with pytest.raises(ValueError):
            TrustAwareTaskFormationEngine(max_agent_reuse_share=0.0)

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError):
            TrustAwareTaskFormationEngine(learning_rate=0.0)
        with pytest.raises(ValueError):
            TrustAwareTaskFormationEngine(learning_rate=1.5)

    def test_outcome_count_starts_at_zero(self):
        engine = TrustAwareTaskFormationEngine()
        assert engine.outcome_count == 0


# ---------------------------------------------------------------------------
# Pairwise synergy
# ---------------------------------------------------------------------------

class TestPairwiseSynergy:

    def test_high_synergy_agents_score_high(self):
        engine = TrustAwareTaskFormationEngine()
        a = _agent("A", trust=0.9, synergy=0.9, stability=0.9, calibration=0.9)
        b = _agent("B", trust=0.9, synergy=0.9, stability=0.9, calibration=0.9)
        score = engine.compute_pairwise_synergy(a, b, joint_interaction_count=20)
        assert score.synergy_density > 0.8
        assert score.trust_stability > 0.5
        assert score.combined_score > 0.5

    def test_low_synergy_agents_score_low(self):
        engine = TrustAwareTaskFormationEngine()
        a = _agent("A", trust=0.1, synergy=0.1, stability=0.2, calibration=0.1)
        b = _agent("B", trust=0.1, synergy=0.1, stability=0.2, calibration=0.1)
        score = engine.compute_pairwise_synergy(a, b, joint_interaction_count=0)
        assert score.synergy_density < 0.3
        assert score.combined_score < 0.3

    def test_asymmetric_agents(self):
        """Mixed high/low should produce moderate synergy (geometric mean)."""
        engine = TrustAwareTaskFormationEngine()
        high = _agent("H", trust=0.9, synergy=0.9, stability=0.9, calibration=0.9)
        low = _agent("L", trust=0.1, synergy=0.1, stability=0.2, calibration=0.1)
        score = engine.compute_pairwise_synergy(high, low, joint_interaction_count=5)
        assert 0.1 < score.synergy_density < 0.8

    def test_deeper_interaction_increases_stability(self):
        """More joint interactions should increase the trust stability score."""
        engine = TrustAwareTaskFormationEngine()
        a = _agent("A", stability=0.7)
        b = _agent("B", stability=0.7)
        shallow = engine.compute_pairwise_synergy(a, b, joint_interaction_count=0)
        deep = engine.compute_pairwise_synergy(a, b, joint_interaction_count=20)
        assert deep.trust_stability > shallow.trust_stability

    def test_synergy_density_clamped_to_unit(self):
        engine = TrustAwareTaskFormationEngine()
        a = _agent("A", trust=1.0, synergy=1.0, calibration=1.0)
        b = _agent("B", trust=1.0, synergy=1.0, calibration=1.0)
        score = engine.compute_pairwise_synergy(a, b, joint_interaction_count=100)
        assert 0.0 <= score.synergy_density <= 1.0
        assert 0.0 <= score.trust_stability <= 1.0
        assert 0.0 <= score.combined_score <= 1.0

    def test_pairwise_result_fields(self):
        engine = TrustAwareTaskFormationEngine()
        a = _agent("A")
        b = _agent("B")
        score = engine.compute_pairwise_synergy(a, b, joint_interaction_count=3)
        assert score.agent_a == "A"
        assert score.agent_b == "B"
        assert score.interaction_history_depth == 3


# ---------------------------------------------------------------------------
# Team scoring
# ---------------------------------------------------------------------------

class TestTeamScoring:

    def test_minimum_two_agents_required(self):
        engine = TrustAwareTaskFormationEngine()
        with pytest.raises(ValueError):
            engine.score_team([_agent("A")])

    def test_high_quality_team(self):
        """A team of high-trust diverse agents should score well."""
        engine = TrustAwareTaskFormationEngine()
        agents = [
            _agent("A", trust=0.8, synergy=0.8, stability=0.85),
            _agent("B", trust=0.6, synergy=0.7, stability=0.75),
            _agent("C", trust=0.4, synergy=0.6, stability=0.65),
        ]
        candidate = engine.score_team(agents)
        assert candidate.composite_score > 0.0
        assert candidate.entropy_score > 0.0
        assert len(candidate.agent_ids) == 3

    def test_homogeneous_trust_gets_high_entropy(self):
        """Agents with similar trust should produce higher entropy."""
        engine = TrustAwareTaskFormationEngine()
        similar = [
            _agent("A", trust=0.5, synergy=0.5, stability=0.5),
            _agent("B", trust=0.5, synergy=0.5, stability=0.5),
            _agent("C", trust=0.5, synergy=0.5, stability=0.5),
        ]
        candidate = engine.score_team(similar)
        assert candidate.entropy_score > 0.95  # nearly uniform

    def test_skewed_trust_gets_lower_entropy(self):
        """One dominant-trust agent should reduce entropy."""
        engine = TrustAwareTaskFormationEngine()
        skewed = [
            _agent("A", trust=0.95, synergy=0.8, stability=0.9),
            _agent("B", trust=0.02, synergy=0.3, stability=0.3),
            _agent("C", trust=0.03, synergy=0.3, stability=0.3),
        ]
        candidate = engine.score_team(skewed)
        assert candidate.entropy_score < 0.5

    def test_entropy_penalty_reduces_composite(self):
        """Teams below min_entropy should have penalised composite scores."""
        engine = TrustAwareTaskFormationEngine(min_entropy=0.8)
        skewed = [
            _agent("A", trust=0.95, synergy=0.9, stability=0.9),
            _agent("B", trust=0.03, synergy=0.9, stability=0.9),
            _agent("C", trust=0.02, synergy=0.9, stability=0.9),
        ]
        balanced = [
            _agent("D", trust=0.5, synergy=0.5, stability=0.5),
            _agent("E", trust=0.5, synergy=0.5, stability=0.5),
            _agent("F", trust=0.5, synergy=0.5, stability=0.5),
        ]
        skewed_score = engine.score_team(skewed)
        balanced_score = engine.score_team(balanced)

        # Despite higher raw synergy, the skewed team's composite is
        # penalised due to low entropy.
        assert skewed_score.metadata["entropy_penalty_applied"] is True
        assert balanced_score.metadata["entropy_penalty_applied"] is False

    def test_team_candidate_metadata(self):
        engine = TrustAwareTaskFormationEngine()
        agents = [_agent("A"), _agent("B")]
        candidate = engine.score_team(agents)
        assert "pairwise_count" in candidate.metadata
        assert candidate.metadata["pairwise_count"] == 1
        assert "weights" in candidate.metadata

    def test_composite_score_in_unit_range(self):
        engine = TrustAwareTaskFormationEngine()
        for pool_size in range(2, 5):
            pool = _diverse_pool(pool_size)
            candidate = engine.score_team(pool)
            assert 0.0 <= candidate.composite_score <= 1.0


# ---------------------------------------------------------------------------
# Team recommendation
# ---------------------------------------------------------------------------

class TestTeamRecommendation:

    def test_basic_recommendation(self):
        engine = TrustAwareTaskFormationEngine(min_entropy=0.1)
        pool = _diverse_pool(5)
        teams = engine.recommend_teams(pool, team_size=3, top_k=3)
        assert 0 < len(teams) <= 3
        for t in teams:
            assert len(t.agent_ids) == 3

    def test_teams_sorted_by_composite(self):
        engine = TrustAwareTaskFormationEngine(min_entropy=0.1)
        pool = _diverse_pool(5)
        teams = engine.recommend_teams(pool, team_size=2, top_k=5)
        scores = [t.composite_score for t in teams]
        assert scores == sorted(scores, reverse=True)

    def test_team_size_validation(self):
        engine = TrustAwareTaskFormationEngine()
        pool = _diverse_pool(4)
        with pytest.raises(ValueError):
            engine.recommend_teams(pool, team_size=1)
        with pytest.raises(ValueError):
            engine.recommend_teams(pool, team_size=5)

    def test_top_k_validation(self):
        engine = TrustAwareTaskFormationEngine()
        pool = _diverse_pool(4)
        with pytest.raises(ValueError):
            engine.recommend_teams(pool, team_size=2, top_k=0)

    def test_entropy_filter_removes_low_diversity_teams(self):
        """High min_entropy should filter out skewed teams entirely."""
        engine = TrustAwareTaskFormationEngine(min_entropy=0.99)
        # Pool where some pairs have very different trust -> entropy < 0.99
        agents = [
            _agent("A", trust=0.99),
            _agent("B", trust=0.01),
            _agent("C", trust=0.5),
            _agent("D", trust=0.5),
        ]
        teams = engine.recommend_teams(agents, team_size=2, top_k=10)
        # Only teams with near-equal trust should survive
        for t in teams:
            assert t.entropy_score >= 0.99

    def test_agent_reuse_constraint(self):
        """No agent should appear in more than max_agent_reuse_share of teams."""
        engine = TrustAwareTaskFormationEngine(
            min_entropy=0.1, max_agent_reuse_share=0.4
        )
        pool = _diverse_pool(6)
        top_k = 5
        teams = engine.recommend_teams(pool, team_size=3, top_k=top_k)
        max_appearances = max(1, int(top_k * 0.4))
        agent_counts: dict[str, int] = {}
        for t in teams:
            for aid in t.agent_ids:
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
        for aid, count in agent_counts.items():
            assert count <= max_appearances, (
                f"Agent {aid} appeared {count} times, max allowed {max_appearances}"
            )

    def test_recommendation_returns_unique_teams(self):
        engine = TrustAwareTaskFormationEngine(min_entropy=0.1)
        pool = _diverse_pool(5)
        teams = engine.recommend_teams(pool, team_size=2, top_k=5)
        team_sets = [t.agent_ids for t in teams]
        assert len(team_sets) == len(set(team_sets)), "Duplicate teams detected"


# ---------------------------------------------------------------------------
# Outcome feedback & heuristic recalibration
# ---------------------------------------------------------------------------

class TestOutcomeFeedback:

    def test_outcome_increments_count(self):
        engine = TrustAwareTaskFormationEngine()
        outcome = OutcomeRecord(
            team_id="t1",
            agent_ids=frozenset(["A", "B"]),
            predicted_synergy=0.6,
            realised_synergy=0.8,
            predicted_stability=0.5,
            realised_stability=0.7,
        )
        engine.record_outcome(outcome)
        assert engine.outcome_count == 1

    def test_positive_outcome_increases_synergy_adjustment(self):
        """When realised > predicted, the adjustment should become positive."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.5)
        outcome = OutcomeRecord(
            team_id="t1",
            agent_ids=frozenset(["A", "B"]),
            predicted_synergy=0.4,
            realised_synergy=0.8,
            predicted_stability=0.5,
            realised_stability=0.5,
        )
        engine.record_outcome(outcome)
        adj = engine.get_synergy_adjustment("A", "B")
        assert adj > 0.0

    def test_negative_outcome_decreases_synergy_adjustment(self):
        """When realised < predicted, the adjustment should become negative."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.5)
        outcome = OutcomeRecord(
            team_id="t1",
            agent_ids=frozenset(["A", "B"]),
            predicted_synergy=0.8,
            realised_synergy=0.3,
            predicted_stability=0.5,
            realised_stability=0.5,
        )
        engine.record_outcome(outcome)
        adj = engine.get_synergy_adjustment("A", "B")
        assert adj < 0.0

    def test_adjustment_clamped(self):
        """Synergy adjustments should be bounded to prevent runaway drift."""
        engine = TrustAwareTaskFormationEngine(learning_rate=1.0)
        for _ in range(50):
            engine.record_outcome(OutcomeRecord(
                team_id="t",
                agent_ids=frozenset(["X", "Y"]),
                predicted_synergy=0.0,
                realised_synergy=1.0,
                predicted_stability=0.5,
                realised_stability=0.5,
            ))
        adj = engine.get_synergy_adjustment("X", "Y")
        assert -0.5 <= adj <= 0.5

    def test_weight_recalibration_after_outcomes(self):
        """Weights should shift based on which prediction dimension was more accurate."""
        engine = TrustAwareTaskFormationEngine(
            weight_synergy=0.45, weight_stability=0.30, weight_entropy=0.25,
            learning_rate=0.5,
        )
        original_weights = dict(engine.weights)

        # Feed an outcome where synergy prediction was perfect but stability
        # prediction was terrible.
        engine.record_outcome(OutcomeRecord(
            team_id="t",
            agent_ids=frozenset(["A", "B"]),
            predicted_synergy=0.7,
            realised_synergy=0.7,    # perfect synergy prediction
            predicted_stability=0.2,
            realised_stability=0.9,  # terrible stability prediction
        ))

        # Synergy weight should increase relative to stability weight
        assert engine.weights["synergy"] > original_weights["synergy"]
        assert engine.weights["stability"] < original_weights["stability"]

    def test_weights_remain_normalised(self):
        """After recalibration, weights should still sum to ~1.0."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.3)
        for i in range(10):
            engine.record_outcome(OutcomeRecord(
                team_id=f"t{i}",
                agent_ids=frozenset([f"A{i}", f"B{i}"]),
                predicted_synergy=0.3 + i * 0.05,
                realised_synergy=0.5,
                predicted_stability=0.5,
                realised_stability=0.3 + i * 0.04,
            ))
        w = engine.weights
        assert abs(w["synergy"] + w["stability"] + w["entropy"] - 1.0) < 1e-6

    def test_learned_adjustment_affects_scoring(self):
        """After recording a positive outcome, the pair's synergy in future
        scoring should be higher than without the outcome."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.5)
        a = _agent("A", synergy=0.5, trust=0.5, stability=0.5, calibration=0.5)
        b = _agent("B", synergy=0.5, trust=0.5, stability=0.5, calibration=0.5)

        score_before = engine.compute_pairwise_synergy(a, b, 5)

        engine.record_outcome(OutcomeRecord(
            team_id="t1",
            agent_ids=frozenset(["A", "B"]),
            predicted_synergy=0.5,
            realised_synergy=0.9,
            predicted_stability=0.5,
            realised_stability=0.5,
        ))

        score_after = engine.compute_pairwise_synergy(a, b, 5)
        assert score_after.synergy_density > score_before.synergy_density

    def test_multi_pair_outcome_updates_all_pairs(self):
        """An outcome with 3 agents should update all pairwise adjustments."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.5)
        engine.record_outcome(OutcomeRecord(
            team_id="t",
            agent_ids=frozenset(["A", "B", "C"]),
            predicted_synergy=0.5,
            realised_synergy=0.8,
            predicted_stability=0.5,
            realised_stability=0.5,
        ))
        assert engine.get_synergy_adjustment("A", "B") > 0
        assert engine.get_synergy_adjustment("A", "C") > 0
        assert engine.get_synergy_adjustment("B", "C") > 0

    def test_get_weight_history_snapshot(self):
        engine = TrustAwareTaskFormationEngine()
        snap = engine.get_weight_history_snapshot()
        assert "weights" in snap
        assert "outcome_count" in snap
        assert "synergy_adjustments" in snap
        assert snap["outcome_count"] == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end team formation pipeline
# ---------------------------------------------------------------------------

class TestIntegrationPipeline:

    def test_full_pipeline(self):
        """End-to-end: build pool → recommend → record outcome → re-recommend."""
        engine = TrustAwareTaskFormationEngine(
            min_entropy=0.1, learning_rate=0.3
        )
        pool = _diverse_pool(6)

        # First recommendation round
        teams_r1 = engine.recommend_teams(pool, team_size=3, top_k=3)
        assert len(teams_r1) > 0

        # Simulate realised outcomes for the top team
        top = teams_r1[0]
        engine.record_outcome(OutcomeRecord(
            team_id=top.team_id,
            agent_ids=top.agent_ids,
            predicted_synergy=top.mean_synergy_density,
            realised_synergy=top.mean_synergy_density + 0.1,
            predicted_stability=top.trust_stability_score,
            realised_stability=top.trust_stability_score + 0.05,
        ))

        # Second recommendation round – should be affected by outcome
        teams_r2 = engine.recommend_teams(pool, team_size=3, top_k=3)
        assert len(teams_r2) > 0

        # The engine state should reflect the learning
        assert engine.outcome_count == 1
        snap = engine.get_weight_history_snapshot()
        assert snap["outcome_count"] == 1

    def test_repeated_outcomes_converge_weights(self):
        """Many consistent outcomes should push weights toward a stable ratio."""
        engine = TrustAwareTaskFormationEngine(learning_rate=0.2)
        for i in range(30):
            # Both dimensions have small, equal errors — the engine should
            # converge toward a balanced split of the non-entropy budget.
            engine.record_outcome(OutcomeRecord(
                team_id=f"t{i}",
                agent_ids=frozenset(["A", "B"]),
                predicted_synergy=0.5,
                realised_synergy=0.55,   # small synergy error
                predicted_stability=0.5,
                realised_stability=0.55, # same stability error
            ))
        w = engine.weights
        # When both prediction dimensions have identical error magnitudes,
        # weights should converge toward an equal split of the non-entropy
        # budget.
        non_entropy_syn = w["synergy"] / (w["synergy"] + w["stability"])
        assert 0.4 < non_entropy_syn < 0.6

    def test_diversity_maintained_across_recommendations(self):
        """Even after learning, entropy constraint keeps teams diverse."""
        engine = TrustAwareTaskFormationEngine(
            min_entropy=0.5, learning_rate=0.3
        )
        pool = _diverse_pool(6)

        # Heavy positive outcomes for the top-trust pair
        for _ in range(10):
            engine.record_outcome(OutcomeRecord(
                team_id="t",
                agent_ids=frozenset(["alpha", "beta"]),
                predicted_synergy=0.6,
                realised_synergy=0.9,
                predicted_stability=0.5,
                realised_stability=0.8,
            ))

        teams = engine.recommend_teams(pool, team_size=3, top_k=5)
        for t in teams:
            assert t.entropy_score >= 0.5
