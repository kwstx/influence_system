import pytest
from datetime import datetime

from models.influence_signal import InfluenceSignal, DimensionValue
from models.cooperative_reliability_profile import (
    CooperativeReliabilityProfile,
    CooperativeReliabilitySnapshot,
)
from models.influence_projection import (
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
) -> InfluenceSignal:
    return InfluenceSignal(
        agent_id=agent_id,
        long_term_temporal_impact_weight=DimensionValue(temporal_weight),
        synergy_amplification_contribution=DimensionValue(synergy),
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
# InfluenceProjector tests
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
