"""
Tests for the DriftDetection subsystem.

Covers:
- DriftObservation creation and properties
- DriftDetectionParameters validation
- DriftDetector registration, observation ingestion, decay, and recovery
- Smooth decay curve properties (monotonic, bounded)
- Reversible recovery after re-alignment
- EMA smoothing behaviour
- Edge cases (zero deviation, extreme values, auto-registration)
"""

import pytest
from datetime import datetime

from models.drift_detection import (
    DriftObservation,
    DriftDetectionParameters,
    DriftDetector,
    TrustDecayState,
    _clamp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_constant_deviation(
    detector: DriftDetector,
    agent_id: str,
    projected: float,
    realised: float,
    n: int,
) -> TrustDecayState:
    """Feed *n* identical observations and return the final state."""
    state = None
    for _ in range(n):
        state = detector.record_observation(agent_id, projected, realised)
    return state


# ---------------------------------------------------------------------------
# DriftObservation tests
# ---------------------------------------------------------------------------

class TestDriftObservation:

    def test_absolute_deviation(self):
        obs = DriftObservation(agent_id="A", projected_influence=0.8, realised_impact=0.5)
        assert abs(obs.absolute_deviation - 0.3) < 1e-9

    def test_signed_deviation_optimistic(self):
        obs = DriftObservation(agent_id="A", projected_influence=0.9, realised_impact=0.4)
        assert obs.signed_deviation > 0  # projection was optimistic

    def test_signed_deviation_pessimistic(self):
        obs = DriftObservation(agent_id="A", projected_influence=0.3, realised_impact=0.7)
        assert obs.signed_deviation < 0  # projection was pessimistic

    def test_zero_deviation(self):
        obs = DriftObservation(agent_id="A", projected_influence=0.5, realised_impact=0.5)
        assert obs.absolute_deviation == 0.0
        assert obs.signed_deviation == 0.0

    def test_to_dict_contains_all_keys(self):
        obs = DriftObservation(agent_id="X", projected_influence=0.6, realised_impact=0.4)
        d = obs.to_dict()
        for key in [
            "observation_id", "agent_id", "timestamp",
            "projected_influence", "realised_impact",
            "absolute_deviation", "signed_deviation",
        ]:
            assert key in d
        assert d["agent_id"] == "X"

    def test_frozen(self):
        obs = DriftObservation(agent_id="A", projected_influence=0.5, realised_impact=0.5)
        with pytest.raises(AttributeError):
            obs.projected_influence = 0.99


# ---------------------------------------------------------------------------
# DriftDetectionParameters tests
# ---------------------------------------------------------------------------

class TestDriftDetectionParameters:

    def test_default_parameters_are_valid(self):
        params = DriftDetectionParameters()
        params.validate()  # should not raise

    def test_invalid_ema_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            DriftDetectionParameters(ema_alpha=0.0).validate()

    def test_invalid_ema_alpha_negative(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            DriftDetectionParameters(ema_alpha=-0.1).validate()

    def test_invalid_deviation_tolerance_negative(self):
        with pytest.raises(ValueError, match="deviation_tolerance"):
            DriftDetectionParameters(deviation_tolerance=-0.01).validate()

    def test_invalid_sustained_ticks_to_decay(self):
        with pytest.raises(ValueError, match="sustained_ticks_to_decay"):
            DriftDetectionParameters(sustained_ticks_to_decay=0).validate()

    def test_invalid_sustained_ticks_to_recover(self):
        with pytest.raises(ValueError, match="sustained_ticks_to_recover"):
            DriftDetectionParameters(sustained_ticks_to_recover=0).validate()

    def test_invalid_decay_rate_zero(self):
        with pytest.raises(ValueError, match="decay_rate"):
            DriftDetectionParameters(decay_rate=0.0).validate()

    def test_invalid_decay_rate_one(self):
        with pytest.raises(ValueError, match="decay_rate"):
            DriftDetectionParameters(decay_rate=1.0).validate()

    def test_invalid_min_trust_above_max(self):
        with pytest.raises(ValueError, match="min_trust"):
            DriftDetectionParameters(min_trust=1.0, max_trust=1.0).validate()

    def test_invalid_recovery_rate(self):
        with pytest.raises(ValueError, match="recovery_rate"):
            DriftDetectionParameters(recovery_rate=0.0).validate()

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            DriftDetectionParameters(window_size=0).validate()


# ---------------------------------------------------------------------------
# DriftDetector — registration
# ---------------------------------------------------------------------------

class TestDriftDetectorRegistration:

    def test_register_agent(self):
        dd = DriftDetector()
        state = dd.register_agent("A", initial_trust=0.9)
        assert state.agent_id == "A"
        assert state.current_trust == 0.9
        assert state.original_trust == 0.9

    def test_register_clamps_trust(self):
        dd = DriftDetector()
        state = dd.register_agent("A", initial_trust=5.0)
        assert state.current_trust == 1.0

    def test_register_clamps_trust_below_min(self):
        dd = DriftDetector(DriftDetectionParameters(min_trust=0.2))
        state = dd.register_agent("A", initial_trust=0.05)
        assert state.current_trust == 0.2

    def test_get_current_trust(self):
        dd = DriftDetector()
        dd.register_agent("A", 0.85)
        assert dd.get_current_trust("A") == 0.85

    def test_get_current_trust_unregistered_raises(self):
        dd = DriftDetector()
        with pytest.raises(KeyError):
            dd.get_current_trust("nonexistent")

    def test_get_state_returns_same_object(self):
        dd = DriftDetector()
        dd.register_agent("A")
        s1 = dd.get_state("A")
        s2 = dd.get_state("A")
        assert s1 is s2

    def test_get_all_agent_ids(self):
        dd = DriftDetector()
        dd.register_agent("A")
        dd.register_agent("B")
        ids = dd.get_all_agent_ids()
        assert set(ids) == {"A", "B"}


# ---------------------------------------------------------------------------
# DriftDetector — no-drift scenario
# ---------------------------------------------------------------------------

class TestNoDrift:

    def test_perfect_alignment_no_decay(self):
        """When projected == realised, trust should never change."""
        dd = DriftDetector()
        dd.register_agent("A", 0.9)
        for _ in range(20):
            dd.record_observation("A", projected_influence=0.6, realised_impact=0.6)
        assert dd.get_current_trust("A") == 0.9

    def test_below_tolerance_no_decay(self):
        """Small deviations below the tolerance threshold should not trigger decay."""
        params = DriftDetectionParameters(deviation_tolerance=0.2)
        dd = DriftDetector(params)
        dd.register_agent("A", 0.9)
        # Deviation = 0.1 < tolerance of 0.2
        for _ in range(30):
            dd.record_observation("A", projected_influence=0.6, realised_impact=0.5)
        assert dd.get_current_trust("A") == 0.9


# ---------------------------------------------------------------------------
# DriftDetector — decay behaviour
# ---------------------------------------------------------------------------

class TestDecayBehaviour:

    def test_sustained_deviation_starts_decay(self):
        """Trust must decrease after sustained above-threshold deviation."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.1,
            sustained_ticks_to_decay=3,
            decay_rate=0.1,
            ema_alpha=1.0,  # instant EMA for test clarity
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # Feed 3 above-threshold observations to trigger decay
        for _ in range(3):
            dd.record_observation("A", 0.8, 0.2)  # deviation = 0.6

        # Now the next observation should apply decay
        state = dd.record_observation("A", 0.8, 0.2)
        assert state.current_trust < 1.0
        assert state.is_decaying

    def test_decay_is_monotonic_while_drifting(self):
        """Trust must decrease monotonically during sustained drift."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.05,
            sustained_ticks_to_decay=2,
            decay_rate=0.05,
            ema_alpha=1.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        trusts = []
        for _ in range(15):
            state = dd.record_observation("A", 0.9, 0.3)
            trusts.append(state.current_trust)

        # After patience period, trust should only go down
        decaying_trusts = [t for t in trusts if t < 1.0]
        assert len(decaying_trusts) > 0
        for i in range(1, len(decaying_trusts)):
            assert decaying_trusts[i] <= decaying_trusts[i - 1]

    def test_decay_respects_floor(self):
        """Trust must never drop below min_trust."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.01,
            sustained_ticks_to_decay=1,
            decay_rate=0.5,  # aggressive decay
            min_trust=0.2,
            ema_alpha=1.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        for _ in range(100):
            dd.record_observation("A", 1.0, 0.0)

        assert dd.get_current_trust("A") >= 0.2

    def test_exponential_decay_shape(self):
        """Verify the decay follows approximate exponential curve."""
        rate = 0.1
        params = DriftDetectionParameters(
            deviation_tolerance=0.01,
            sustained_ticks_to_decay=1,
            decay_rate=rate,
            min_trust=0.0,
            ema_alpha=1.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # One tick to start decay (sustained_ticks_to_decay=1 means first
        # above-threshold triggers is_decaying)
        dd.record_observation("A", 0.5, 0.0)  # tick 1: crosses threshold
        # Now 5 more decay ticks
        expected_trust = 1.0
        for tick in range(5):
            state = dd.record_observation("A", 0.5, 0.0)
            expected_trust *= (1.0 - rate)
            # Allow small tolerance for the first tick's EMA lag
            assert abs(state.current_trust - expected_trust) < 0.05, (
                f"tick {tick}: expected ~{expected_trust:.4f}, got {state.current_trust:.4f}"
            )


# ---------------------------------------------------------------------------
# DriftDetector — recovery behaviour
# ---------------------------------------------------------------------------

class TestRecoveryBehaviour:

    def _decay_then_align(
        self,
        dd: DriftDetector,
        agent_id: str,
        decay_ticks: int,
        align_ticks: int,
    ) -> List[float]:
        """Helper: drift for *decay_ticks*, then align for *align_ticks*.
        Returns the trust history for the alignment phase."""
        for _ in range(decay_ticks):
            dd.record_observation(agent_id, 0.9, 0.1)  # large deviation

        trusts = []
        for _ in range(align_ticks):
            state = dd.record_observation(agent_id, 0.5, 0.5)  # perfect alignment
            trusts.append(state.current_trust)
        return trusts

    def test_recovery_after_realignment(self):
        """Trust should start recovering once deviation drops below threshold."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.1,
            sustained_ticks_to_decay=2,
            sustained_ticks_to_recover=2,
            decay_rate=0.1,
            recovery_rate=0.1,
            ema_alpha=0.8,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # Decay for a while
        for _ in range(10):
            dd.record_observation("A", 0.9, 0.2)

        decayed_trust = dd.get_current_trust("A")
        assert decayed_trust < 1.0

        # Now send aligned observations
        for _ in range(30):
            dd.record_observation("A", 0.5, 0.5)

        recovered_trust = dd.get_current_trust("A")
        assert recovered_trust > decayed_trust

    def test_recovery_is_monotonic(self):
        """During pure recovery (no new drift), trust should only increase."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.05,
            sustained_ticks_to_decay=1,
            sustained_ticks_to_recover=1,
            decay_rate=0.15,
            recovery_rate=0.08,
            ema_alpha=1.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # Drift phase
        for _ in range(8):
            dd.record_observation("A", 0.8, 0.1)

        # Recovery phase
        recovery_trusts = []
        for _ in range(25):
            state = dd.record_observation("A", 0.5, 0.5)
            recovery_trusts.append(state.current_trust)

        # Once recovery starts, trust should only increase or stay flat
        # (first few may still be in patience window)
        started_recovery = False
        for i in range(1, len(recovery_trusts)):
            if recovery_trusts[i] > recovery_trusts[i - 1]:
                started_recovery = True
            if started_recovery:
                assert recovery_trusts[i] >= recovery_trusts[i - 1]

    def test_recovery_does_not_exceed_original(self):
        """Trust should never recover beyond the original trust value."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.05,
            sustained_ticks_to_decay=1,
            sustained_ticks_to_recover=1,
            decay_rate=0.1,
            recovery_rate=0.5,  # very aggressive recovery
            ema_alpha=1.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 0.8)

        # Decay
        for _ in range(5):
            dd.record_observation("A", 0.9, 0.1)

        # Recover for a long time
        for _ in range(100):
            dd.record_observation("A", 0.5, 0.5)

        assert dd.get_current_trust("A") <= 0.8

    def test_sigmoid_gated_recovery_is_gradual(self):
        """Early recovery steps should be smaller than later ones (sigmoid gate)."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.05,
            sustained_ticks_to_decay=1,
            sustained_ticks_to_recover=1,
            decay_rate=0.15,
            recovery_rate=0.1,
            ema_alpha=1.0,
            min_trust=0.0,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # Heavy decay
        for _ in range(20):
            dd.record_observation("A", 0.9, 0.0)

        trust_before_recovery = dd.get_current_trust("A")

        # Collect recovery increments
        increments = []
        prev_trust = trust_before_recovery
        for _ in range(12):
            state = dd.record_observation("A", 0.5, 0.5)
            inc = state.current_trust - prev_trust
            if inc > 0:
                increments.append(inc)
            prev_trust = state.current_trust

        # The first few positive increments should be smaller than later ones
        # (due to sigmoid gating), unless we've already hit ceiling
        if len(increments) >= 4:
            early_avg = sum(increments[:2]) / 2
            later_avg = sum(increments[2:4]) / 2
            assert later_avg >= early_avg - 1e-9  # later >= early (with tolerance)


# ---------------------------------------------------------------------------
# DriftDetector — EMA smoothing
# ---------------------------------------------------------------------------

class TestEMASmoothing:

    def test_ema_smooths_single_spike(self):
        """A single spike in deviation should not immediately cause decay."""
        params = DriftDetectionParameters(
            deviation_tolerance=0.15,
            sustained_ticks_to_decay=3,
            ema_alpha=0.3,
        )
        dd = DriftDetector(params)
        dd.register_agent("A", 1.0)

        # Lots of aligned observations
        for _ in range(10):
            dd.record_observation("A", 0.5, 0.5)

        # One big spike
        dd.record_observation("A", 1.0, 0.0)  # deviation = 1.0

        # Immediately follow with aligned
        for _ in range(3):
            dd.record_observation("A", 0.5, 0.5)

        # Trust should still be at 1.0 – the single spike was smoothed out
        assert dd.get_current_trust("A") == 1.0

    def test_ema_alpha_sensitivity(self):
        """Higher alpha should make EMA respond faster to deviations."""
        fast_params = DriftDetectionParameters(ema_alpha=0.9)
        slow_params = DriftDetectionParameters(ema_alpha=0.1)

        dd_fast = DriftDetector(fast_params)
        dd_slow = DriftDetector(slow_params)
        dd_fast.register_agent("A", 1.0)
        dd_slow.register_agent("A", 1.0)

        # Feed same deviations
        for _ in range(5):
            dd_fast.record_observation("A", 0.8, 0.2)
            dd_slow.record_observation("A", 0.8, 0.2)

        # Fast EMA should have higher deviation estimate
        assert dd_fast.get_state("A").ema_deviation > dd_slow.get_state("A").ema_deviation


# ---------------------------------------------------------------------------
# DriftDetector — auto-registration and edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_auto_register_on_first_observation(self):
        """Recording for an unregistered agent should auto-register it."""
        dd = DriftDetector()
        state = dd.record_observation("NEW", 0.5, 0.5)
        assert state.agent_id == "NEW"
        assert state.current_trust == 1.0  # default

    def test_get_observations_returns_window(self):
        dd = DriftDetector(DriftDetectionParameters(window_size=5))
        dd.register_agent("A")
        for i in range(10):
            dd.record_observation("A", 0.5, 0.5)
        obs = dd.get_observations("A")
        assert len(obs) == 5  # window capped at 5

    def test_get_deviation_summary(self):
        dd = DriftDetector()
        dd.register_agent("A", 0.9)
        dd.record_observation("A", 0.7, 0.5)
        summary = dd.get_deviation_summary("A")
        assert summary["agent_id"] == "A"
        assert summary["observation_count"] == 1
        assert summary["current_trust"] == 0.9
        assert "ema_deviation" in summary
        assert "parameters" in summary

    def test_deviation_summary_unregistered_raises(self):
        dd = DriftDetector()
        with pytest.raises(KeyError):
            dd.get_deviation_summary("ghost")

    def test_reset_agent(self):
        dd = DriftDetector()
        dd.register_agent("A", 1.0)
        # Decay it
        for _ in range(20):
            dd.record_observation("A", 0.9, 0.0)
        assert dd.get_current_trust("A") < 1.0

        # Reset
        state = dd.reset_agent("A")
        assert state.current_trust == 1.0
        assert state.ema_deviation == 0.0
        assert not state.is_decaying
        assert len(dd.get_observations("A")) == 0

    def test_reset_agent_with_new_trust(self):
        dd = DriftDetector()
        dd.register_agent("A", 1.0)
        state = dd.reset_agent("A", new_trust=0.7)
        assert state.current_trust == 0.7
        assert state.original_trust == 0.7

    def test_reset_unregistered_raises(self):
        dd = DriftDetector()
        with pytest.raises(KeyError):
            dd.reset_agent("ghost")

    def test_multiple_agents_independent(self):
        """Drift in one agent should not affect another."""
        dd = DriftDetector(DriftDetectionParameters(
            deviation_tolerance=0.05,
            sustained_ticks_to_decay=1,
            decay_rate=0.1,
            ema_alpha=1.0,
        ))
        dd.register_agent("A", 1.0)
        dd.register_agent("B", 1.0)

        # Only A drifts
        for _ in range(10):
            dd.record_observation("A", 0.9, 0.1)
            dd.record_observation("B", 0.5, 0.5)

        assert dd.get_current_trust("A") < 1.0
        assert dd.get_current_trust("B") == 1.0


# ---------------------------------------------------------------------------
# DriftDetector — full lifecycle (decay → recovery → stable)
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    def test_decay_recovery_cycle(self):
        """
        Simulate a complete lifecycle:
        1. Agent starts with full trust
        2. Sustained drift causes smooth decay
        3. Agent re-aligns and trust recovers
        4. Trust stabilises near original value
        """
        params = DriftDetectionParameters(
            deviation_tolerance=0.1,
            sustained_ticks_to_decay=3,
            sustained_ticks_to_recover=2,
            decay_rate=0.08,
            recovery_rate=0.1,
            ema_alpha=0.5,
            min_trust=0.1,
        )
        dd = DriftDetector(params)
        dd.register_agent("lifecycle", 1.0)

        # Phase 1: good behaviour
        for _ in range(5):
            dd.record_observation("lifecycle", 0.5, 0.48)

        assert dd.get_current_trust("lifecycle") == 1.0

        # Phase 2: sustained drift
        for _ in range(15):
            dd.record_observation("lifecycle", 0.9, 0.3)

        mid_trust = dd.get_current_trust("lifecycle")
        assert mid_trust < 1.0
        assert mid_trust >= params.min_trust

        # Phase 3: re-alignment
        for _ in range(40):
            dd.record_observation("lifecycle", 0.5, 0.5)

        final_trust = dd.get_current_trust("lifecycle")
        assert final_trust > mid_trust
        assert final_trust <= 1.0

    def test_intermittent_drift_does_not_fully_recover(self):
        """
        An agent that alternates between drift and alignment should maintain
        a reduced trust (the recovery is slower than re-triggering decay).
        """
        params = DriftDetectionParameters(
            deviation_tolerance=0.1,
            sustained_ticks_to_decay=2,
            sustained_ticks_to_recover=2,
            decay_rate=0.1,
            recovery_rate=0.05,
            ema_alpha=0.7,
        )
        dd = DriftDetector(params)
        dd.register_agent("flaky", 1.0)

        # 3 cycles of drift(5) → align(5)
        for _ in range(3):
            for _ in range(5):
                dd.record_observation("flaky", 0.8, 0.2)
            for _ in range(5):
                dd.record_observation("flaky", 0.5, 0.5)

        # Trust should be below original – not fully recovered
        assert dd.get_current_trust("flaky") < 1.0

    def test_history_snapshots_recorded(self):
        """State history should log a snapshot for every observation."""
        dd = DriftDetector()
        dd.register_agent("A", 1.0)
        for _ in range(5):
            dd.record_observation("A", 0.5, 0.5)
        state = dd.get_state("A")
        assert len(state.history) == 5


# ---------------------------------------------------------------------------
# Clamp utility
# ---------------------------------------------------------------------------

class TestClamp:

    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(3.0, lo=1.0, hi=2.0) == 2.0
        assert _clamp(0.5, lo=1.0, hi=2.0) == 1.0
