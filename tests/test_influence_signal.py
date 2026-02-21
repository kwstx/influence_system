import pytest
from datetime import datetime
from models.influence_signal import InfluenceSignal, DimensionValue

def test_influence_signal_creation():
    signal = InfluenceSignal(
        agent_id="agent_001",
        marginal_cooperative_influence=DimensionValue(0.85, version="1.1.0"),
        synergy_amplification_contribution=DimensionValue(0.42),
        predictive_calibration_accuracy=DimensionValue(0.99),
        cooperative_stability_coefficient=DimensionValue(0.75),
        long_term_temporal_impact_weight=DimensionValue(0.60)
    )
    
    assert signal.agent_id == "agent_001"
    assert signal.marginal_cooperative_influence.value == 0.85
    assert signal.marginal_cooperative_influence.version == "1.1.0"
    assert signal.predictive_calibration_accuracy.value == 0.99

def test_influence_signal_serialization():
    signal = InfluenceSignal(
        agent_id="agent_002",
        marginal_cooperative_influence=DimensionValue(0.5, version="2.0.0", metadata={"method": "causal_impact"})
    )
    
    data = signal.to_dict()
    assert data["agent_id"] == "agent_002"
    assert data["dimensions"]["marginal_cooperative_influence"]["value"] == 0.5
    assert data["dimensions"]["marginal_cooperative_influence"]["version"] == "2.0.0"
    assert data["dimensions"]["marginal_cooperative_influence"]["metadata"]["method"] == "causal_impact"
    
    # Deserialization
    new_signal = InfluenceSignal.from_dict(data)
    assert new_signal.agent_id == signal.agent_id
    assert new_signal.marginal_cooperative_influence.value == 0.5
    assert new_signal.marginal_cooperative_influence.version == "2.0.0"

def test_dimension_querying():
    signal = InfluenceSignal(agent_id="agent_003")
    dim = signal.get_dimension("synergy_amplification_contribution")
    assert dim is not None
    assert dim.value == 0.0
    
    invalid_dim = signal.get_dimension("non_existent")
    assert invalid_dim is None

def test_no_aggregate_score():
    """Ensure we haven't implemented an aggregate score by accident."""
    signal = InfluenceSignal(agent_id="agent_004")
    with pytest.raises(AttributeError):
        _ = signal.trust_score
