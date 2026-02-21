# Influence System

The Influence System is a high-integrity, multi-agent behavioral control layer designed to govern cooperative AI populations. It replaces traditional throughput-based metrics with a causal impact framework, utilizing predictive trust coefficients and structural influence reweighting to ensure systemic stability and genuine downstream value.

## Core Architecture

The system is built upon a hierarchy of adaptive models that continuously calibrate agent behavior against realized real-world outcomes.

### 1. Influence Projection and Propagation
The `InfluenceProjector` forecasts an agent's expected future impact by integrating historical reliability, synergy signature participation, and temporal impact memory. 
- **Propagation Scaling**: Individual projections are scaled by trust coefficients, amplifying the influence of high-integrity agents while attenuating those with lower predictive accuracy.
- **Uncertainty Bounds**: Every projection includes confidence scores derived from historical variance and reliability indices.

### 2. Collaborative Consensus Aggregation
The `CollaborativeProjectionAggregator` merges individual projections into a unified shared forecast for collaborative tasks.
- **Trust-Weighted Consensus**: Aggregation weights are dynamically adjusted based on trust, ensuring that consensus is driven by the most reliable actors.
- **Entropy Constraints**: To prevent over-centralization, the system enforces dominance limits, redistributing influence to maintain a diverse and robust forecasting pool.

### 3. Trust-Aware Task Formation
The synergy between agents is optimized via the `TrustTaskFormationEngine`. It biases team assembly toward high-density synergy clusters while maintaining entropy-driven exploration to prevent the formation of rigid, high-trust silos.

### 4. Behavioral Drift Detection
The `DriftDetector` identifies divergence between projected influence and realized downstream impact. Sustained deviations trigger automated trust decay, mitigating the risk of agents optimizing for short-term metric inflation rather than systemic value.

### 5. Governance and Causal Transparency
The `GovernanceAPI` provides deep introspection into the system's state. It exposes multi-dimensional tensors representing trust vectors, reliability curves, and entropy-adjusted weights, accompanied by causal traces for every governance decision.

## Technical Components

- **Causal Trust Weighting**: Nonlinear multiplicative mapping of predictive accuracy, marginal cooperative influence, and synergy density.
- **Real-World Calibration**: Automated alignment of forecasts with empirical outcomes across multiple temporal horizons.
- **Reliability Profiling**: Continuous generation of comprehensive agent profiles based on cooperative stability and impact persistence.

## Getting Started

### Installation
Ensure that Python 3.8+ is installed. Clone the repository and install necessary dependencies.

### Running Tests
The system includes an extensive suite of unit and integration tests covering all core projections and governance logic:

```bash
pytest
```
