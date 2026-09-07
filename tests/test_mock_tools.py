import json

from agentic_ai.tools.mock_tools import (
    AVAILABLE_SERVICES,
    MockService,
    get_mock_logs,
    get_mock_metrics,
    get_mock_pod_status,
)


def test_mock_logs_are_deterministic() -> None:
    first = get_mock_logs("payment")
    second = get_mock_logs("payment")

    assert first == second


def test_mock_metrics_are_deterministic() -> None:
    first = get_mock_metrics("payment")
    second = get_mock_metrics("payment")

    assert first == second


def test_mock_pod_status_is_deterministic() -> None:
    first = get_mock_pod_status("payment")
    second = get_mock_pod_status("payment")

    assert first == second


def test_mock_logs_return_valid_json() -> None:
    result = get_mock_logs("payment")

    parsed = json.loads(result)

    assert parsed["status"] == "success"
    assert parsed["service"] == "payment"


def test_mock_metrics_return_valid_json() -> None:
    result = get_mock_metrics("payment")

    parsed = json.loads(result)

    assert parsed["status"] == "success"
    assert parsed["service"] == "payment"


def test_mock_pod_status_return_valid_json() -> None:
    result = get_mock_pod_status("payment")

    parsed = json.loads(result)

    assert parsed["status"] == "success"
    assert parsed["service"] == "payment"


def test_mock_logs_describe_connection_pool_exhaustion() -> None:
    """The fixture must keep telling the connection-pool-exhaustion story."""
    parsed = json.loads(get_mock_logs("payment"))

    messages = " ".join(
        f"{entry['message']} {entry.get('error', '')}" for entry in parsed["logs"]
    ).lower()

    assert "connection pool exhausted" in messages
    assert "timeout" in messages
    assert all(entry["level"] == "ERROR" for entry in parsed["logs"])


def test_mock_metrics_show_exhausted_connection_pool() -> None:
    """Connection utilisation must stay at capacity for the scenario to hold."""
    parsed = json.loads(get_mock_metrics("payment"))

    connections = parsed["metrics"]["database_connections"]

    assert connections["active"] == connections["pool_size"]
    assert connections["utilization_percent"] == 100
    assert parsed["metrics"]["error_rate_percent"] > 0


def test_mock_pod_status_shows_unhealthy_restarting_pods() -> None:
    """At least one pod must be unready and restarting for the scenario."""
    parsed = json.loads(get_mock_pod_status("payment"))

    unready = [pod for pod in parsed["pods"] if not pod["ready"]]

    assert unready, "scenario requires at least one unready pod"
    assert any(pod["restart_count"] > 1 for pod in unready)


def test_mock_logs_unknown_service() -> None:
    result = get_mock_logs("unknown-service")

    parsed = json.loads(result)

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "service_not_found"
    assert parsed["service"] == "unknown-service"


def test_mock_metrics_unknown_service() -> None:
    result = get_mock_metrics("unknown-service")

    parsed = json.loads(result)

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "service_not_found"
    assert parsed["service"] == "unknown-service"


def test_mock_pod_status_unknown_service() -> None:
    result = get_mock_pod_status("unknown-service")

    parsed = json.loads(result)

    assert parsed["status"] == "error"
    assert parsed["error_type"] == "service_not_found"
    assert parsed["service"] == "unknown-service"


def test_error_response_lists_available_services() -> None:
    """The error must teach the agent which service names actually work."""
    parsed = json.loads(get_mock_logs("payment-api"))

    assert parsed["available_services"] == AVAILABLE_SERVICES
    assert "payment" in parsed["available_services"]
    assert "payment" in parsed["message"]


def test_enum_value_is_accepted_as_service_name() -> None:
    """MockService members are str-valued, so they work as arguments directly."""
    parsed = json.loads(get_mock_metrics(MockService.PAYMENT))

    assert parsed["status"] == "success"