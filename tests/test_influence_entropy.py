import pytest

from models.influence_entropy import InfluenceEntropy


class TestInfluenceEntropy:

    def test_high_entropy_no_regularization(self):
        module = InfluenceEntropy(
            entropy_threshold=0.75,
            dominance_softness=0.5,
            opportunity_boost_strength=0.5,
        )
        weights, snapshot = module.evaluate(
            agent_ids=["A", "B", "C"],
            trust_coefficients=[0.34, 0.33, 0.33],
            base_weights=[0.34, 0.33, 0.33],
            reliability_scores=[0.8, 0.8, 0.8],
        )
        assert snapshot.regularization_applied is False
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_low_entropy_reduces_dominant_weight(self):
        module = InfluenceEntropy(
            entropy_threshold=0.9,
            dominance_softness=0.6,
            opportunity_boost_strength=0.4,
        )
        weights, snapshot = module.evaluate(
            agent_ids=["dominant", "other1", "other2"],
            trust_coefficients=[0.95, 0.03, 0.02],
            base_weights=[0.95, 0.03, 0.02],
            reliability_scores=[0.9, 0.8, 0.7],
        )
        assert snapshot.regularization_applied is True
        assert weights[0] < 0.95
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_underutilized_reliable_agent_gets_boost(self):
        module = InfluenceEntropy(
            entropy_threshold=0.95,
            dominance_softness=0.8,
            opportunity_boost_strength=0.9,
        )
        weights, snapshot = module.evaluate(
            agent_ids=["dominant", "reliable_underutilized", "low_reliability"],
            trust_coefficients=[0.98, 0.01, 0.01],
            base_weights=[0.98, 0.01, 0.01],
            reliability_scores=[0.7, 0.95, 0.1],
        )
        assert snapshot.regularization_applied is True
        assert "reliable_underutilized" in snapshot.underutilized_reliable_agent_ids
        assert weights[1] > 0.01
        assert weights[1] > weights[2]

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            InfluenceEntropy(entropy_threshold=0.0)
        with pytest.raises(ValueError):
            InfluenceEntropy(dominance_softness=1.1)
        with pytest.raises(ValueError):
            InfluenceEntropy(opportunity_boost_strength=-0.1)
