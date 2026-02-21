from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

@dataclass(frozen=True)
class DimensionValue:
    """
    Represents a single dimension of influence with its own versioning and metadata.
    This allows each metric to evolve independently.
    """
    value: float
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "version": self.version,
            "metadata": self.metadata
        }

@dataclass(frozen=True)
class InfluenceSignal:
    """
    Structured InfluenceSignal object capturing multi-dimensional cooperative impact.
    Each dimension is independently stored, versioned, and queryable.
    Do not compute a single aggregate trust score here.
    """
    agent_id: str
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Dimensions as specified in the requirements
    marginal_cooperative_influence: DimensionValue = field(
        default_factory=lambda: DimensionValue(0.0)
    )
    synergy_amplification_contribution: DimensionValue = field(
        default_factory=lambda: DimensionValue(0.0)
    )
    predictive_calibration_accuracy: DimensionValue = field(
        default_factory=lambda: DimensionValue(0.0)
    )
    cooperative_stability_coefficient: DimensionValue = field(
        default_factory=lambda: DimensionValue(1.0)
    )
    long_term_temporal_impact_weight: DimensionValue = field(
        default_factory=lambda: DimensionValue(1.0)
    )
    
    # Systemic context and traceability
    context: Dict[str, Any] = field(default_factory=dict)
    audit_id: Optional[str] = None

    def get_dimension(self, name: str) -> Optional[DimensionValue]:
        """Query a specific dimension by name."""
        return getattr(self, name, None)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dictionary representation for storage and querying."""
        return {
            "agent_id": self.agent_id,
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": {
                "marginal_cooperative_influence": self.marginal_cooperative_influence.to_dict(),
                "synergy_amplification_contribution": self.synergy_amplification_contribution.to_dict(),
                "predictive_calibration_accuracy": self.predictive_calibration_accuracy.to_dict(),
                "cooperative_stability_coefficient": self.cooperative_stability_coefficient.to_dict(),
                "long_term_temporal_impact_weight": self.long_term_temporal_impact_weight.to_dict()
            },
            "context": self.context,
            "audit_id": self.audit_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InfluenceSignal:
        """Hydrate an InfluenceSignal from a dictionary."""
        dims = data.get("dimensions", {})
        
        def parse_dim(key: str, default_val: float) -> DimensionValue:
            d = dims.get(key, {})
            if isinstance(d, dict):
                return DimensionValue(
                    value=float(d.get("value", default_val)),
                    version=str(d.get("version", "1.0.0")),
                    metadata=dict(d.get("metadata", {}))
                )
            return DimensionValue(value=float(d))

        return cls(
            agent_id=data["agent_id"],
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            marginal_cooperative_influence=parse_dim("marginal_cooperative_influence", 0.0),
            synergy_amplification_contribution=parse_dim("synergy_amplification_contribution", 0.0),
            predictive_calibration_accuracy=parse_dim("predictive_calibration_accuracy", 0.0),
            cooperative_stability_coefficient=parse_dim("cooperative_stability_coefficient", 1.0),
            long_term_temporal_impact_weight=parse_dim("long_term_temporal_impact_weight", 1.0),
            context=data.get("context", {}),
            audit_id=data.get("audit_id")
        )
