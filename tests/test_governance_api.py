from datetime import UTC, datetime, timedelta

from models.governance_api import GovernanceAPI
from models.influence_signal import DimensionValue, InfluenceSignal


def _make_signal(agent_id: str, ts: datetime, memory: float, context: dict | None = None) -> InfluenceSignal:
    return InfluenceSignal(
        agent_id=agent_id,
        timestamp=ts,
        long_term_temporal_impact_weight=DimensionValue(memory),
        context=context or {},
    )


def _seed_api() -> GovernanceAPI:
    api = GovernanceAPI()
    start = datetime(2026, 1, 1, tzinfo=UTC)

    api.upsert_agent(
        agent_id="agent_A",
        trust_dimensions={
            "predictive_accuracy_index": 0.82,
            "marginal_cooperative_influence": 0.74,
            "synergy_density_contribution": 0.79,
            "cooperative_stability_score": 0.77,
            "long_term_impact_persistence": 0.70,
        },
        recent_signals=[
            _make_signal("agent_A", start, 0.65),
            _make_signal(
                "agent_A",
                start + timedelta(days=1),
                0.72,
                {"delayed_cooperative_outcome": True},
            ),
            _make_signal("agent_A", start + timedelta(days=2), 0.77),
        ],
        cohort_trust={"agent_B": 0.62, "agent_C": 0.41},
    )
    api.upsert_agent(
        agent_id="agent_B",
        trust_dimensions={
            "predictive_accuracy_index": 0.63,
            "marginal_cooperative_influence": 0.60,
            "synergy_density_contribution": 0.58,
            "cooperative_stability_score": 0.66,
            "long_term_impact_persistence": 0.59,
        },
    )
    api.upsert_agent(
        agent_id="agent_C",
        trust_dimensions={
            "predictive_accuracy_index": 0.44,
            "marginal_cooperative_influence": 0.46,
            "synergy_density_contribution": 0.40,
            "cooperative_stability_score": 0.47,
            "long_term_impact_persistence": 0.50,
        },
    )

    api.record_calibration_event(
        "agent_A",
        predicted_by_horizon={"short": {"gain": 1.0}, "long": {"gain": 2.0}},
        realized_by_horizon={"short": {"gain": 0.9}, "long": {"gain": 2.3}},
    )
    api.record_calibration_event(
        "agent_A",
        predicted_by_horizon={"short": {"gain": 1.1}},
        realized_by_horizon={"short": {"gain": 1.0}},
    )

    for projected, realized in [(0.82, 0.61), (0.81, 0.57), (0.79, 0.58), (0.78, 0.60)]:
        api.record_drift_observation("agent_A", projected, realized)

    return api


def test_all_governance_endpoints_return_structured_representations():
    api = _seed_api()
    paths = [
        "/governance/agents/agent_A/trust-vector",
        "/governance/agents/agent_A/influence-distribution",
        "/governance/agents/agent_A/entropy-adjusted-weight",
        "/governance/agents/agent_A/reliability-curve",
        "/governance/agents/agent_A/calibration-history",
        "/governance/agents/agent_A/drift-status",
    ]

    for path in paths:
        response = api.handle_request("GET", path)
        assert response["status_code"] == 200
        body = response["body"]
        assert "representation" in body
        assert "causal_trace" in body
        assert body["representation"]["type"] == "multi_dimensional_tensor"


def test_trust_vector_includes_dimension_level_traces():
    api = _seed_api()
    body = api.handle_request("GET", "/governance/agents/agent_A/trust-vector")["body"]

    traces = body["causal_trace"]["dimension_traces"]
    names = [trace["dimension"] for trace in traces]
    assert "predictive_accuracy_index" in names
    assert "trust_coefficient" in names
    assert len(body["representation"]["values"]) == 6


def test_influence_distribution_exposes_projection_tensor_and_factor_trace():
    api = _seed_api()
    body = api.handle_request("GET", "/governance/agents/agent_A/influence-distribution")["body"]

    values = body["representation"]["values"]
    assert len(values) == 2
    assert len(values[0]) == 3
    assert "factors" in body["causal_trace"]
    assert "propagation_scale" in body["causal_trace"]


def test_entropy_adjusted_weight_returns_cohort_matrix_and_selected_agent_slice():
    api = _seed_api()
    body = api.handle_request("GET", "/governance/agents/agent_A/entropy-adjusted-weight")["body"]

    assert body["representation"]["axes"][0]["name"] == "agent_id"
    assert len(body["representation"]["values"]) == 3
    selected = body["selected_agent"]
    assert 0.0 <= selected["entropy_adjusted_weight"] <= 1.0
    assert "entropy_snapshot" in body["causal_trace"]


def test_reliability_curve_supports_horizon_filter():
    api = _seed_api()
    full_curve = api.handle_request("GET", "/governance/agents/agent_A/reliability-curve")["body"]
    short_curve = api.handle_request(
        "GET", "/governance/agents/agent_A/reliability-curve?horizon=short"
    )["body"]

    assert len(full_curve["representation"]["values"]) >= len(short_curve["representation"]["values"])
    assert short_curve["horizon_filter"] == "short"


def test_calibration_history_returns_three_dimensional_tensor():
    api = _seed_api()
    body = api.handle_request("GET", "/governance/agents/agent_A/calibration-history")["body"]

    assert len(body["representation"]["axes"]) == 3
    assert len(body["records"]) == 2
    assert "magnitude_deviation" in body["causal_trace"]["formula"]


def test_drift_status_returns_lifecycle_and_deviation_state():
    api = _seed_api()
    body = api.handle_request("GET", "/governance/agents/agent_A/drift-status")["body"]

    assert body["status"]["observation_count"] == 4
    assert "state_transitions" in body["causal_trace"]
    assert body["representation"]["axes"][0]["labels"] == ["trust", "deviation", "lifecycle"]


def test_unknown_route_returns_404():
    api = _seed_api()
    response = api.handle_request("GET", "/governance/agents/agent_A/not-real")
    assert response["status_code"] == 404
