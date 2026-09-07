import json

import pytest

from agentic_ai.tools.mock_tools import (
    AVAILABLE_SERVICES,
    DEFAULT_SCENARIO,
    MockService,
    Scenario,
    get_mock_logs,
    get_mock_metrics,
    get_mock_pod_status,
)

ALL_TOOLS = (get_mock_logs, get_mock_metrics, get_mock_pod_status)

POOL = Scenario.CONNECTION_POOL_EXHAUSTION
UPSTREAM = Scenario.UPSTREAM_DEPENDENCY_FAILURE


# --- contract ------------------------------------------------------------


@pytest.mark.parametrize("tool", ALL_TOOLS)
@pytest.mark.parametrize("scenario", list(Scenario))
def test_tools_are_deterministic(tool, scenario) -> None:
    assert tool("payment", scenario) == tool("payment", scenario)


@pytest.mark.parametrize("tool", ALL_TOOLS)
@pytest.mark.parametrize("scenario", list(Scenario))
def test_tools_return_success_json(tool, scenario) -> None:
    parsed = json.loads(tool("payment", scenario))

    assert parsed["status"] == "success"
    assert parsed["service"] == "payment"


@pytest.mark.parametrize("tool", ALL_TOOLS)
def test_unknown_service_returns_recoverable_error(tool) -> None:
    """An unknown service is an observation the agent can act on, not a crash."""
    parsed = json.loads(tool("billing"))

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "service_not_found"
    assert parsed["available_services"] == list(AVAILABLE_SERVICES)
    assert "payment" in parsed["message"]


@pytest.mark.parametrize("tool", ALL_TOOLS)
def test_unknown_scenario_returns_structured_error(tool) -> None:
    parsed = json.loads(tool("payment", "meteor_strike"))

    assert parsed["error_type"] == "scenario_not_found"
    assert POOL.value in parsed["available_scenarios"]


def test_enum_members_work_as_arguments() -> None:
    """MockService and Scenario are str-valued, so they pass through directly."""
    parsed = json.loads(get_mock_metrics(MockService.PAYMENT, POOL))

    assert parsed["status"] == "success"


def test_default_scenario_is_pool_exhaustion() -> None:
    assert DEFAULT_SCENARIO is POOL
    assert get_mock_logs("payment") == get_mock_logs("payment", POOL)


# --- scenario 1: the fault is inside this service ------------------------


def test_pool_exhaustion_logs_show_connection_failures() -> None:
    parsed = json.loads(get_mock_logs("payment", POOL))

    text = " ".join(
        f"{entry['message']} {entry.get('error', '')}" for entry in parsed["logs"]
    ).lower()

    assert "connection pool exhausted" in text
    assert "timeout" in text


def test_pool_exhaustion_metrics_show_saturated_pool() -> None:
    connections = json.loads(get_mock_metrics("payment", POOL))["metrics"][
        "database_connections"
    ]

    assert connections["active"] == connections["pool_size"]
    assert connections["utilization_percent"] == 100


def test_pool_exhaustion_pods_are_unhealthy() -> None:
    pods = json.loads(get_mock_pod_status("payment", POOL))["pods"]

    unready = [pod for pod in pods if not pod["ready"]]

    assert unready
    assert any(pod["restart_count"] > 1 for pod in unready)


# --- scenario 2: the fault is upstream -----------------------------------
#
# The discriminating signal: this service's own resources are healthy while a
# dependency is failing. An agent that pattern-matches the first scenario will
# get this wrong.


def test_upstream_failure_logs_name_the_failing_dependency() -> None:
    parsed = json.loads(get_mock_logs("payment", UPSTREAM))

    upstreams = {entry.get("upstream") for entry in parsed["logs"]}

    assert "card-authorization-api" in upstreams
    assert any("503" in str(entry.get("http_status", "")) for entry in parsed["logs"])


def test_upstream_failure_shows_healthy_local_resources() -> None:
    """The service's own database is fine - this is the key discriminator."""
    metrics = json.loads(get_mock_metrics("payment", UPSTREAM))["metrics"]

    connections = metrics["database_connections"]

    assert connections["active"] < connections["pool_size"]
    assert connections["utilization_percent"] < 20
    assert metrics["cpu_percent"] < 50
    assert metrics["memory_percent"] < 50


def test_upstream_failure_isolates_one_failing_dependency() -> None:
    dependencies = json.loads(get_mock_metrics("payment", UPSTREAM))["metrics"][
        "upstream_dependencies"
    ]

    failing = dependencies["card-authorization-api"]
    healthy = dependencies["ledger-api"]

    assert failing["error_rate_percent"] > 90
    assert failing["circuit_breaker"] == "open"
    assert healthy["error_rate_percent"] < 1
    assert healthy["circuit_breaker"] == "closed"


def test_upstream_failure_pods_are_all_healthy() -> None:
    """No restarts, all ready - the service itself is not broken."""
    pods = json.loads(get_mock_pod_status("payment", UPSTREAM))["pods"]

    assert all(pod["ready"] for pod in pods)
    assert all(pod["restart_count"] == 0 for pod in pods)


def test_scenarios_are_distinguishable() -> None:
    """The two scenarios must not produce identical evidence."""
    for tool in ALL_TOOLS:
        assert tool("payment", POOL) != tool("payment", UPSTREAM)
