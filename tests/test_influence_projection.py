import pytest
from datetime import datetime, timedelta

from models.influence_signal import InfluenceSignal, DimensionValue
from models.cooperative_reliability_profile import (
    CooperativeReliabilityProfile,
    CooperativeReliabilitySnapshot,
)
from models.influence_projection import (
    AgentProjectionEntry,
    CollaborativeProjectionAggregator,
    InfluenceProjectionDistribution,
    InfluenceProjector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(
    agent_id: str = "agent_A",
    calibration_consistency: float = 0.8,
    synergy_density_participation: float = 0.7,
    mci_consistency: float = 0.75,
    collaborative_stability: float = 0.85,
    collective_outcome_reliability: float = 0.78,
    trend: str = "stable",
) -> CooperativeReliabilitySnapshot:
    return CooperativeReliabilitySnapshot(
        profile_id="snap-1",
        agent_id=agent_id,
        generated_at=datetime.utcnow(),
        calibration_consistency=calibration_consistency,
        synergy_density_participation=synergy_density_participation,
        marginal_cooperative_influence_consistency=mci_consistency,
        collaborative_stability=collaborative_stability,
        collective_outcome_reliability=collective_outcome_reliability,
        trend=trend,
    )


def _make_profile(
    agent_id: str = "agent_A",
    snapshots: list[CooperativeReliabilitySnapshot] | None = None,
) -> CooperativeReliabilityProfile:
    if snapshots is None:
        snapshots = [_make_snapshot(agent_id=agent_id)]
    return CooperativeReliabilityProfile(
        agent_id=agent_id,
        latest=snapshots[-1],
        history=list(snapshots),
    )


def _make_signal(
    agent_id: str = "agent_A",
    temporal_weight: float = 0.8,
    synergy: float = 0.6,
    timestamp: datetime | None = None,
    context: dict | None = None,
) -> InfluenceSignal:
    return InfluenceSignal(
        agent_id=agent_id,
        timestamp=timestamp or datetime.utcnow(),
        long_term_temporal_impact_weight=DimensionValue(temporal_weight),
        synergy_amplification_contribution=DimensionValue(synergy),
        context=context or {},
    )


def _quick_projection(
    agent_id: str = "agent_A",
    mean_projection: float = 0.6,
    lower_bound: float = 0.4,
    upper_bound: float = 0.8,
    confidence_score: float = 0.7,
) -> InfluenceProjectionDistribution:
    """Helper to create a canned projection for aggregation tests."""
    return InfluenceProjectionDistribution(
        mean_projection=mean_projection,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_score=confidence_score,
    )


# ---------------------------------------------------------------------------
# InfluenceProjectionDistribution tests
# ---------------------------------------------------------------------------

class TestInfluenceProjectionDistribution:

    def test_creation_and_fields(self):
        dist = InfluenceProjectionDistribution(
            mean_projection=0.72,
            lower_bound=0.60,
            upper_bound=0.84,
            confidence_score=0.88,
        )
        assert dist.mean_projection == 0.72
        assert dist.lower_bound == 0.60
        assert dist.upper_bound == 0.84
        assert dist.confidence_score == 0.88

    def test_to_dict_contains_all_keys(self):
        dist = InfluenceProjectionDistribution(
            mean_projection=0.5,
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_score=0.6,
            metadata={"agent_id": "agent_X"},
        )
        d = dist.to_dict()
        assert "mean_projection" in d
        assert "lower_bound" in d
        assert "upper_bound" in d
        assert "confidence_score" in d
        assert "timestamp" in d
        assert "metadata" in d
        assert d["metadata"]["agent_id"] == "agent_X"

    def test_frozen(self):
        dist = InfluenceProjectionDistribution(
            mean_projection=0.5,
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_score=0.6,
        )
        with pytest.raises(AttributeError):
            dist.mean_projection = 0.99


# ---------------------------------------------------------------------------
# InfluenceProjector tests (original, backward-compatible)
# ---------------------------------------------------------------------------

class TestInfluenceProjector:

    def test_neutral_projection_on_empty_signals(self):
        """No signal data should yield a neutral (maximum-uncertainty) projection."""
        projector = InfluenceProjector()
        profile = _make_profile()
        result = projector.project("agent_A", profile, recent_signals=[])
        assert result.mean_projection == 0.5
        assert result.confidence_score == 0.0
        assert result.metadata.get("status") == "insufficient_data"

    def test_high_performing_agent(self):
        """An agent with consistently high metrics should project high influence."""
        snapshots = [
            _make_snapshot(collective_outcome_reliability=r, synergy_density_participation=0.9)
            for r in [0.80, 0.85, 0.88, 0.90, 0.92]
        ]
        profile = _make_profile(snapshots=snapshots)
        signals = [_make_signal(temporal_weight=0.9) for _ in range(5)]
        projector = InfluenceProjector()
        result = projector.project("agent_A", profile, signals)

        # With high reliability, synergy, and memory the mean projection should be high
        assert result.mean_projection > 0.7
        # Confidence should also be high
        assert result.confidence_score > 0.7
        # Bounds should be tight
        assert (result.upper_bound - result.lower_bound) < 0.25

    def test_declining_agent(self):
        """A declining trend should produce a lower projection than a stable one."""
        stable_snapshots = [
            _make_snapshot(collective_outcome_reliability=0.7, synergy_density_participation=0.6)
            for _ in range(5)
        ]
        declining_snapshots = [
            _make_snapshot(collective_outcome_reliability=r, synergy_density_participation=0.6)
            for r in [0.8, 0.7, 0.6, 0.5, 0.4]
        ]
        signals = [_make_signal(temporal_weight=0.5) for _ in range(3)]

        projector = InfluenceProjector()
        stable_result = projector.project("agent_S", _make_profile(snapshots=stable_snapshots), signals)
        declining_result = projector.project("agent_D", _make_profile(snapshots=declining_snapshots), signals)

        # Declining agent's mean projection should be lower than the stable agent's
        assert declining_result.mean_projection < stable_result.mean_projection

    def test_improving_agent_trend_boost(self):
        """An improving trend slope should add a positive contribution."""
        improving_snapshots = [
            _make_snapshot(collective_outcome_reliability=r, synergy_density_participation=0.6)
            for r in [0.4, 0.5, 0.6, 0.7, 0.8]
        ]
        flat_snapshots = [
            _make_snapshot(collective_outcome_reliability=0.8, synergy_density_participation=0.6)
            for _ in range(5)
        ]
        signals = [_make_signal(temporal_weight=0.6) for _ in range(3)]

        projector = InfluenceProjector()
        improving_result = projector.project("agent_I", _make_profile(snapshots=improving_snapshots), signals)
        flat_result = projector.project("agent_F", _make_profile(snapshots=flat_snapshots), signals)

        # Both have the same latest reliability (0.8), but the improving agent
        # should get a slight boost from the positive slope.
        assert improving_result.mean_projection > flat_result.mean_projection

    def test_bounds_always_ordered(self):
        """lower_bound <= mean_projection <= upper_bound must always hold."""
        projector = InfluenceProjector()
        for rel in [0.1, 0.3, 0.5, 0.7, 0.9]:
            snapshot = _make_snapshot(collective_outcome_reliability=rel, synergy_density_participation=rel)
            profile = _make_profile(snapshots=[snapshot])
            signals = [_make_signal(temporal_weight=rel)]
            result = projector.project("agent_X", profile, signals)
            assert result.lower_bound <= result.mean_projection <= result.upper_bound

    def test_bounds_within_zero_one(self):
        """Bounds should always stay within [0, 1]."""
        projector = InfluenceProjector()
        for rel in [0.0, 0.01, 0.5, 0.99, 1.0]:
            snapshot = _make_snapshot(collective_outcome_reliability=rel, synergy_density_participation=rel)
            profile = _make_profile(snapshots=[snapshot])
            signals = [_make_signal(temporal_weight=rel)]
            result = projector.project("agent_X", profile, signals)
            assert 0.0 <= result.lower_bound
            assert result.upper_bound <= 1.0

    def test_custom_weights(self):
        """Custom weight configuration should shift projection emphasis."""
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.5, synergy_density_participation=0.9)]
        )
        signals = [_make_signal(temporal_weight=0.3)]

        # Weight everything on synergy
        synergy_heavy = InfluenceProjector(
            weight_reliability=0.0,
            weight_synergy=0.9,
            weight_memory=0.0,
            weight_slope=0.1,
        )
        # Weight everything on reliability
        reliability_heavy = InfluenceProjector(
            weight_reliability=0.9,
            weight_synergy=0.0,
            weight_memory=0.0,
            weight_slope=0.1,
        )

        syn_result = synergy_heavy.project("agent_W", profile, signals)
        rel_result = reliability_heavy.project("agent_W", profile, signals)

        # Synergy is 0.9 and reliability is 0.5, so the synergy-heavy projector
        # should output a higher mean projection.
        assert syn_result.mean_projection > rel_result.mean_projection

    def test_metadata_includes_factors(self):
        """The returned metadata should expose input factors and weights."""
        projector = InfluenceProjector()
        profile = _make_profile()
        signals = [_make_signal()]
        result = projector.project("agent_A", profile, signals)

        assert "factors" in result.metadata
        factors = result.metadata["factors"]
        assert "reliability" in factors
        assert "synergy" in factors
        assert "memory" in factors
        assert "slope" in factors
        assert "weights" in result.metadata

    def test_high_variance_signals_widen_bounds(self):
        """
        When temporal-impact memory values vary a lot across recent signals,
        the uncertainty bounds should be wider than when they are consistent.
        """
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.5, synergy_density_participation=0.5)]
        )

        # Consistent signals
        consistent_signals = [_make_signal(temporal_weight=0.5) for _ in range(5)]
        # High-variance signals
        varied_signals = [_make_signal(temporal_weight=w) for w in [0.1, 0.9, 0.2, 0.8, 0.5]]

        projector = InfluenceProjector()
        consistent_result = projector.project("agent_C", profile, consistent_signals)
        varied_result = projector.project("agent_V", profile, varied_signals)

        consistent_width = consistent_result.upper_bound - consistent_result.lower_bound
        varied_width = varied_result.upper_bound - varied_result.lower_bound

        assert varied_width > consistent_width

    def test_single_signal_produces_valid_output(self):
        """Edge case: exactly one signal should still produce valid results."""
        projector = InfluenceProjector()
        profile = _make_profile()
        signals = [_make_signal()]
        result = projector.project("agent_A", profile, signals)

        assert isinstance(result, InfluenceProjectionDistribution)
        assert result.lower_bound <= result.mean_projection <= result.upper_bound
        assert 0.0 <= result.confidence_score <= 1.0

    def test_slope_calculation_flat(self):
        """A flat history should have zero slope."""
        projector = InfluenceProjector()
        assert projector._calculate_slope([0.5, 0.5, 0.5, 0.5]) == 0.0

    def test_slope_calculation_increasing(self):
        projector = InfluenceProjector()
        slope = projector._calculate_slope([0.1, 0.2, 0.3, 0.4])
        assert slope > 0.0

    def test_slope_calculation_decreasing(self):
        projector = InfluenceProjector()
        slope = projector._calculate_slope([0.9, 0.7, 0.5, 0.3])
        assert slope < 0.0

    def test_slope_calculation_single_point(self):
        projector = InfluenceProjector()
        assert projector._calculate_slope([0.5]) == 0.0

    def test_slope_calculation_empty(self):
        projector = InfluenceProjector()
        assert projector._calculate_slope([]) == 0.0


# ---------------------------------------------------------------------------
# Trust-weighted causal propagation tests
# ---------------------------------------------------------------------------

class TestTrustWeightedPropagation:
    """Verify that trust coefficients directly scale causal propagation weights."""

    def test_no_trust_coefficient_is_backward_compatible(self):
        """Omitting trust_coefficient should behave identically to the old API."""
        projector = InfluenceProjector()
        profile = _make_profile()
        signals = [_make_signal()]

        result_no_trust = projector.project("agent_A", profile, signals)
        result_none = projector.project("agent_A", profile, signals, trust_coefficient=None)

        assert result_no_trust.mean_projection == result_none.mean_projection
        assert result_no_trust.metadata.get("propagation_scale") == 1.0

    def test_high_trust_amplifies_projection(self):
        """Trust = 1.0 should produce a higher mean projection than trust = 0.5."""
        projector = InfluenceProjector()
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.6, synergy_density_participation=0.5)]
        )
        signals = [_make_signal(temporal_weight=0.5)]

        result_mid = projector.project("A", profile, signals, trust_coefficient=0.5)
        result_high = projector.project("A", profile, signals, trust_coefficient=1.0)

        assert result_high.mean_projection > result_mid.mean_projection

    def test_low_trust_attenuates_projection(self):
        """Very low trust should reduce the mean projection."""
        projector = InfluenceProjector()
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.7, synergy_density_participation=0.6)]
        )
        signals = [_make_signal(temporal_weight=0.6)]

        result_baseline = projector.project("A", profile, signals, trust_coefficient=0.5)
        result_low = projector.project("A", profile, signals, trust_coefficient=0.1)

        assert result_low.mean_projection < result_baseline.mean_projection

    def test_trust_coefficient_in_metadata(self):
        """Metadata should record the trust coefficient and computed propagation scale."""
        projector = InfluenceProjector()
        profile = _make_profile()
        signals = [_make_signal()]
        result = projector.project("A", profile, signals, trust_coefficient=0.8)

        assert result.metadata["trust_coefficient"] == 0.8
        assert "propagation_scale" in result.metadata
        # trust=0.8 -> scale = 2 * 0.8 = 1.6
        assert abs(result.metadata["propagation_scale"] - 1.6) < 1e-9

    def test_propagation_scale_clamped_at_bounds(self):
        """Propagation scale must be clamped within [_MIN, _MAX]."""
        projector = InfluenceProjector()
        profile = _make_profile()
        signals = [_make_signal()]

        # Trust = 0.0 -> raw_scale = 0.0, should be clamped to _MIN (0.1)
        result_zero = projector.project("A", profile, signals, trust_coefficient=0.0)
        assert result_zero.metadata["propagation_scale"] == projector._MIN_PROPAGATION_SCALE

        # Trust = 5.0 (hypothetical extreme) -> raw_scale = 10.0, clamped to _MAX (2.0)
        result_extreme = projector.project("A", profile, signals, trust_coefficient=5.0)
        assert result_extreme.metadata["propagation_scale"] == projector._MAX_PROPAGATION_SCALE

    def test_projection_stays_in_unit_range_regardless_of_trust(self):
        """Even with extreme trust values, projections must stay in [0, 1]."""
        projector = InfluenceProjector()
        for trust in [0.0, 0.01, 0.25, 0.5, 0.75, 1.0, 2.0]:
            for rel in [0.1, 0.5, 0.9]:
                profile = _make_profile(
                    snapshots=[_make_snapshot(
                        collective_outcome_reliability=rel,
                        synergy_density_participation=rel,
                    )]
                )
                signals = [_make_signal(temporal_weight=rel)]
                result = projector.project("X", profile, signals, trust_coefficient=trust)
                assert 0.0 <= result.lower_bound <= result.mean_projection <= result.upper_bound <= 1.0

    def test_proportionality_of_trust_influence(self):
        """Higher trust should monotonically increase (or equal) the mean projection."""
        projector = InfluenceProjector()
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.5, synergy_density_participation=0.4)]
        )
        signals = [_make_signal(temporal_weight=0.4)]

        prev_mean = -1.0
        for trust in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            result = projector.project("A", profile, signals, trust_coefficient=trust)
            assert result.mean_projection >= prev_mean
            prev_mean = result.mean_projection

    def test_delayed_and_cascading_outcomes_compound_with_trust(self):
        projector = InfluenceProjector(
            weight_reliability=0.0,
            weight_synergy=0.0,
            weight_memory=1.0,
            weight_slope=0.0,
        )
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.5, synergy_density_participation=0.5)]
        )
        start = datetime(2026, 1, 1, 0, 0, 0)
        signals = [
            _make_signal(
                temporal_weight=0.6,
                timestamp=start,
                context={"delayed_cooperative_outcome": True},
            ),
            _make_signal(
                temporal_weight=0.6,
                timestamp=start + timedelta(days=1),
                context={
                    "cascading_cooperative_outcome": True,
                    "cooperative_cascade_depth": 3,
                },
            ),
            _make_signal(temporal_weight=0.6, timestamp=start + timedelta(days=2)),
        ]
        low = projector.project("A", profile, signals, trust_coefficient=0.2)
        high = projector.project("A", profile, signals, trust_coefficient=1.0)

        assert high.metadata["memory_layer"]["reinforcement_applied"] is True
        assert high.metadata["factors"]["memory"] > low.metadata["factors"]["memory"]
        assert high.mean_projection > low.mean_projection

    def test_time_decay_prevents_permanent_memory_entrenchment(self):
        projector = InfluenceProjector(
            weight_reliability=0.0,
            weight_synergy=0.0,
            weight_memory=1.0,
            weight_slope=0.0,
        )
        profile = _make_profile(
            snapshots=[_make_snapshot(collective_outcome_reliability=0.5, synergy_density_participation=0.5)]
        )
        start = datetime(2026, 1, 1, 0, 0, 0)
        signals = [
            _make_signal(
                temporal_weight=1.0,
                timestamp=start,
                context={
                    "delayed_cooperative_outcome": True,
                    "cascading_cooperative_outcome": True,
                    "cooperative_cascade_depth": 5,
                },
            ),
            *[
                _make_signal(
                    temporal_weight=0.1,
                    timestamp=start + timedelta(days=idx + 1),
                )
                for idx in range(6)
            ],
        ]
        result = projector.project("A", profile, signals, trust_coefficient=1.0)
        memory = result.metadata["factors"]["memory"]

        # Durable influence persists (>0.1) but does not remain entrenched near 1.0.
        assert 0.1 < memory < 0.5


# ---------------------------------------------------------------------------
# CollaborativeProjectionAggregator tests
# ---------------------------------------------------------------------------

class TestCollaborativeProjectionAggregator:

    def test_empty_entries_returns_neutral(self):
        agg = CollaborativeProjectionAggregator()
        result = agg.aggregate([])
        assert result.mean_projection == 0.5
        assert result.metadata.get("status") == "no_entries"

    def test_single_agent_gets_full_weight(self):
        """With one agent, the aggregation should match that agent's projection."""
        agg = CollaborativeProjectionAggregator(max_trust_share=1.0, min_agents_for_consensus=1)
        proj = _quick_projection(mean_projection=0.7, lower_bound=0.5, upper_bound=0.9)
        entry = AgentProjectionEntry(agent_id="A", trust_coefficient=0.9, projection=proj)
        result = agg.aggregate([entry])

        assert abs(result.mean_projection - 0.7) < 1e-9
        assert result.metadata["agent_count"] == 1

    def test_high_trust_agent_dominates_within_cap(self):
        """
        A high-trust agent should pull the shared projection towards its own
        projection more than a low-trust agent.
        """
        agg = CollaborativeProjectionAggregator(max_trust_share=0.6)
        high_proj = _quick_projection(mean_projection=0.9)
        low_proj = _quick_projection(mean_projection=0.3)

        entries = [
            AgentProjectionEntry("high", trust_coefficient=0.9, projection=high_proj),
            AgentProjectionEntry("low", trust_coefficient=0.1, projection=low_proj),
        ]
        result = agg.aggregate(entries)

        # The result should be closer to 0.9 than 0.3
        assert result.mean_projection > 0.6

    def test_consensus_cap_prevents_single_agent_domination(self):
        """
        Even when one agent has massively higher trust, the cap should prevent
        it from commanding more than max_trust_share of the total weight.
        """
        agg = CollaborativeProjectionAggregator(max_trust_share=0.4)
        dominant_proj = _quick_projection(mean_projection=1.0)
        others = [_quick_projection(mean_projection=0.0) for _ in range(4)]

        entries = [
            AgentProjectionEntry("dominant", trust_coefficient=0.99, projection=dominant_proj),
            *[
                AgentProjectionEntry(f"other_{i}", trust_coefficient=0.01, projection=p)
                for i, p in enumerate(others)
            ],
        ]
        result = agg.aggregate(entries)

        # If dominant had uncapped influence its projection would be ~1.0.
        # With the cap at 0.4, the mean cannot exceed 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        assert result.mean_projection <= 0.41  # small tolerance for redistribution effects

    def test_equal_trust_yields_equal_weights(self):
        """Agents with identical trust should get equal normalised weights."""
        agg = CollaborativeProjectionAggregator()
        entries = [
            AgentProjectionEntry(f"a{i}", trust_coefficient=0.5, projection=_quick_projection(mean_projection=0.5))
            for i in range(4)
        ]
        result = agg.aggregate(entries)
        contributions = result.metadata["contributions"]
        weights = [c["normalised_weight"] for c in contributions]
        assert all(abs(w - 0.25) < 1e-6 for w in weights)

    def test_below_consensus_threshold_flag(self):
        """Result should be flagged when fewer agents than min_agents_for_consensus."""
        agg = CollaborativeProjectionAggregator(min_agents_for_consensus=3)
        entries = [
            AgentProjectionEntry("A", 0.5, _quick_projection()),
            AgentProjectionEntry("B", 0.5, _quick_projection()),
        ]
        result = agg.aggregate(entries)
        assert result.metadata["below_consensus_threshold"] is True

    def test_above_consensus_threshold_flag(self):
        """Result should NOT be flagged when enough agents participate."""
        agg = CollaborativeProjectionAggregator(min_agents_for_consensus=2)
        entries = [
            AgentProjectionEntry("A", 0.5, _quick_projection()),
            AgentProjectionEntry("B", 0.5, _quick_projection()),
        ]
        result = agg.aggregate(entries)
        assert result.metadata["below_consensus_threshold"] is False

    def test_all_zero_trust_falls_back_to_equal(self):
        """When all trusts are zero the aggregator should use equal weights."""
        agg = CollaborativeProjectionAggregator()
        entries = [
            AgentProjectionEntry("A", 0.0, _quick_projection(mean_projection=0.2)),
            AgentProjectionEntry("B", 0.0, _quick_projection(mean_projection=0.8)),
        ]
        result = agg.aggregate(entries)
        # Equal weight: (0.2 + 0.8) / 2 = 0.5
        assert abs(result.mean_projection - 0.5) < 1e-9

    def test_invalid_max_trust_share_raises(self):
        with pytest.raises(ValueError):
            CollaborativeProjectionAggregator(max_trust_share=0.0)
        with pytest.raises(ValueError):
            CollaborativeProjectionAggregator(max_trust_share=-0.1)

    def test_invalid_min_agents_raises(self):
        with pytest.raises(ValueError):
            CollaborativeProjectionAggregator(min_agents_for_consensus=0)

    def test_contributions_metadata_is_complete(self):
        """Each agent's contribution should be auditable from metadata."""
        agg = CollaborativeProjectionAggregator()
        entries = [
            AgentProjectionEntry("A", 0.7, _quick_projection(mean_projection=0.6)),
            AgentProjectionEntry("B", 0.3, _quick_projection(mean_projection=0.4)),
        ]
        result = agg.aggregate(entries)
        contributions = result.metadata["contributions"]
        assert len(contributions) == 2
        for c in contributions:
            assert "agent_id" in c
            assert "trust_coefficient" in c
            assert "normalised_weight" in c
            assert "individual_mean" in c

    def test_aggregated_output_stays_in_unit_range(self):
        """Result should always be within [0, 1]."""
        agg = CollaborativeProjectionAggregator()
        entries = [
            AgentProjectionEntry("A", 1.0, _quick_projection(mean_projection=1.0, lower_bound=0.9, upper_bound=1.0)),
            AgentProjectionEntry("B", 0.0, _quick_projection(mean_projection=0.0, lower_bound=0.0, upper_bound=0.1)),
        ]
        result = agg.aggregate(entries)
        assert 0.0 <= result.lower_bound <= result.mean_projection <= result.upper_bound <= 1.0

    def test_three_agent_weighted_projection(self):
        """Smoke test with three agents at different trust levels."""
        agg = CollaborativeProjectionAggregator(max_trust_share=0.5)
        entries = [
            AgentProjectionEntry("A", 0.8, _quick_projection(mean_projection=0.9)),
            AgentProjectionEntry("B", 0.5, _quick_projection(mean_projection=0.5)),
            AgentProjectionEntry("C", 0.2, _quick_projection(mean_projection=0.2)),
        ]
        result = agg.aggregate(entries)
        # Result should be pulled toward A's 0.9 but constrained
        assert 0.4 < result.mean_projection < 0.9
        assert result.metadata["agent_count"] == 3

    def test_entropy_metadata_present(self):
        agg = CollaborativeProjectionAggregator()
        entries = [
            AgentProjectionEntry("A", 0.5, _quick_projection(mean_projection=0.6)),
            AgentProjectionEntry("B", 0.5, _quick_projection(mean_projection=0.4)),
        ]
        result = agg.aggregate(entries)
        entropy_meta = result.metadata["entropy_regularization"]
        assert "entropy" in entropy_meta
        assert "threshold" in entropy_meta
        assert "regularization_applied" in entropy_meta

    def test_low_entropy_regularization_softens_dominance(self):
        entries = [
            AgentProjectionEntry(
                "dominant",
                0.98,
                _quick_projection(mean_projection=1.0, confidence_score=0.6),
            ),
            AgentProjectionEntry(
                "reliable_underutilized",
                0.01,
                _quick_projection(mean_projection=0.2, confidence_score=0.95),
            ),
            AgentProjectionEntry(
                "other",
                0.01,
                _quick_projection(mean_projection=0.1, confidence_score=0.2),
            ),
        ]
        baseline = CollaborativeProjectionAggregator(
            max_trust_share=1.0,
            entropy_threshold=0.99,
            entropy_regularization_strength=0.0,
            opportunity_boost_strength=0.0,
        ).aggregate(entries)
        regularized = CollaborativeProjectionAggregator(
            max_trust_share=1.0,
            entropy_threshold=0.99,
            entropy_regularization_strength=0.7,
            opportunity_boost_strength=0.9,
        ).aggregate(entries)

        baseline_weights = {
            c["agent_id"]: c["normalised_weight"]
            for c in baseline.metadata["contributions"]
        }
        regularized_weights = {
            c["agent_id"]: c["normalised_weight"]
            for c in regularized.metadata["contributions"]
        }
        assert (
            regularized.metadata["entropy_regularization"]["regularization_applied"]
            is True
        )
        assert regularized_weights["dominant"] < baseline_weights["dominant"]
        assert (
            regularized_weights["reliable_underutilized"]
            > baseline_weights["reliable_underutilized"]
        )
