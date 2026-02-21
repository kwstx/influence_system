from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Any
import uuid

from models.influence_signal import InfluenceSignal
from models.cooperative_reliability_profile import CooperativeReliabilityProfile, CooperativeReliabilitySnapshot

@dataclass(frozen=True)
class InfluenceProjectionDistribution:
    """
    Represents a projected influence distribution with uncertainty bounds.
    """
    mean_projection: float
    lower_bound: float
    upper_bound: float
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_projection": self.mean_projection,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class InfluenceProjector:
    """
    Forecasts an agent's expected future impact contribution.
    """
    
    def __init__(
        self,
        weight_reliability: float = 0.4,
        weight_synergy: float = 0.3,
        weight_memory: float = 0.2,
        weight_slope: float = 0.1
    ):
        self.weights = {
            "reliability": weight_reliability,
            "synergy": weight_synergy,
            "memory": weight_memory,
            "slope": weight_slope
        }

    def project(
        self,
        agent_id: str,
        reliability_profile: CooperativeReliabilityProfile,
        recent_signals: List[InfluenceSignal],
    ) -> InfluenceProjectionDistribution:
        """
        Computes the projected influence distribution based on multiple factors.
        """
        if not recent_signals:
            # Return a default/neutral projection if no signal data is available
            return self._neutral_projection()

        latest_snapshot = reliability_profile.latest
        
        # 1. Historical Reliability
        reliability = latest_snapshot.collective_outcome_reliability
        
        # 2. Synergy Signature Participation Strength
        synergy_strength = latest_snapshot.synergy_density_participation
        
        # 3. Temporal Impact Memory
        # We take the average of the long-term temporal impact weight from recent signals
        memory_values = [s.long_term_temporal_impact_weight.value for s in recent_signals]
        avg_memory = mean(memory_values) if memory_values else 0.0
        
        # 4. Calibration Trend Slope
        # Calculate slope of reliability from history
        history_scores = [s.collective_outcome_reliability for s in reliability_profile.history]
        slope = self._calculate_slope(history_scores)
        
        # Compute mean projection
        # Normalize/clamp slope to a reasonable range (e.g., -0.5 to 0.5) to avoid overshooting
        clamped_slope = max(-0.5, min(0.5, slope))
        
        # Projected influence is a weighted combination
        # The slope is an additive factor (improving trend increases projection)
        base_projection = (
            (self.weights["reliability"] * reliability) +
            (self.weights["synergy"] * synergy_strength) +
            (self.weights["memory"] * avg_memory)
        )
        
        # Apply slope influence
        mean_proj = base_projection + (self.weights["slope"] * clamped_slope)
        
        # Clamp mean_proj to [0, 1] for normalization, but keep it flexible if impact can be larger
        # Requirement doesn't specify if it must be 0-1, but influence usually is.
        mean_proj = max(0.0, min(1.0, mean_proj))
        
        # Uncertainty bounds
        # Uncertainty is inversely proportional to reliability and the consistency of recent signals
        uncertainty_base = 1.0 - reliability
        
        # We also consider the variance in memory if available
        if len(memory_values) > 1:
            variance = mean([(x - avg_memory)**2 for x in memory_values])
            uncertainty_base = (uncertainty_base + variance**0.5) / 2.0
            
        # Bound width expands as uncertainty grows
        bound_width = 0.2 * uncertainty_base + 0.05 # Minimum 5% bound
        
        lower = max(0.0, mean_proj - bound_width)
        upper = min(1.0, mean_proj + bound_width)
        
        # Confidence score reflects how sure we are (inverse of uncertainty)
        confidence = 1.0 - uncertainty_base
        
        return InfluenceProjectionDistribution(
            mean_projection=mean_proj,
            lower_bound=lower,
            upper_bound=upper,
            confidence_score=confidence,
            metadata={
                "agent_id": agent_id,
                "factors": {
                    "reliability": reliability,
                    "synergy": synergy_strength,
                    "memory": avg_memory,
                    "slope": slope
                },
                "weights": self.weights
            }
        )

    def _calculate_slope(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        # Simple slope: average of differences
        diffs = [values[i] - values[i-1] for i in range(1, len(values))]
        return mean(diffs)

    def _neutral_projection(self) -> InfluenceProjectionDistribution:
        return InfluenceProjectionDistribution(
            mean_projection=0.5,
            lower_bound=0.25,
            upper_bound=0.75,
            confidence_score=0.0,
            metadata={"status": "insufficient_data"}
        )
