"""Deterministic mock observability tools.

Each tool returns a JSON-formatted string. JSON-in-a-string satisfies four
requirements at once: the output can be preserved verbatim as
``Evidence.raw_output``, read directly by the model as tool-result content,
compared exactly in tests, and parsed later if structured access is needed.

Scenarios
---------
The fixtures describe complete, internally coherent incidents. The ``scenario``
argument selects which one, and is an explicit parameter rather than module
state so the tools stay pure functions of their arguments. It is deliberately
NOT advertised in the tool schema handed to the model - the agent sees only
``service``, exactly as it would with a real observability tool.
"""

import json
from enum import Enum
from typing import Any


class MockService(str, Enum):
    """Services for which mock incident data is available."""

    PAYMENT = "payment"


class Scenario(str, Enum):
    """The incident each fixture set describes."""

    CONNECTION_POOL_EXHAUSTION = "connection_pool_exhaustion"
    """The service's own database connection pool is saturated."""

    UPSTREAM_DEPENDENCY_FAILURE = "upstream_dependency_failure"
    """The service is healthy; a downstream API it depends on is failing."""


DEFAULT_SCENARIO = Scenario.CONNECTION_POOL_EXHAUSTION

AVAILABLE_SERVICES = tuple(service.value for service in MockService)


def _error_response(service: str) -> str:
    """Return a deterministic JSON error response for an unknown service."""
    return json.dumps(
        {
            "status": "error",
            "error_type": "service_not_found",
            "service": service,
            "available_services": list(AVAILABLE_SERVICES),
            "message": (
                f"No mock data is available for '{service}'. "
                f"Available services: {', '.join(AVAILABLE_SERVICES)}."
            ),
        }
    )


def _is_known_service(service: str) -> bool:
    """Return True if mock data exists for the given service name."""
    return service in AVAILABLE_SERVICES


# --- fixtures ------------------------------------------------------------
#
# Each scenario's three fixtures must tell one coherent story. Incoherent
# fixtures (metrics saying one thing, logs another) would make it impossible
# to tell whether the agent reasoned correctly or simply guessed.

_LOGS: dict[Scenario, dict[str, Any]] = {
    Scenario.CONNECTION_POOL_EXHAUSTION: {
        "status": "success",
        "service": "payment",
        "logs": [
            {
                "level": "ERROR",
                "message": "Failed to acquire database connection",
                "error": "connection pool exhausted",
            },
            {
                "level": "ERROR",
                "message": "Database connection timeout",
                "timeout_seconds": 30,
            },
            {
                "level": "ERROR",
                "message": "Request failed while waiting for database connection",
            },
        ],
    },
    Scenario.UPSTREAM_DEPENDENCY_FAILURE: {
        "status": "success",
        "service": "payment",
        "logs": [
            {
                "level": "ERROR",
                "message": "Call to upstream service failed",
                "upstream": "card-authorization-api",
                "error": "read timeout after 10000ms",
            },
            {
                "level": "ERROR",
                "message": "Upstream returned 503 Service Unavailable",
                "upstream": "card-authorization-api",
                "http_status": 503,
            },
            {
                "level": "WARN",
                "message": "Circuit breaker opened for upstream dependency",
                "upstream": "card-authorization-api",
            },
            {
                "level": "INFO",
                "message": "Local database query completed normally",
                "duration_ms": 12,
            },
        ],
    },
}

_METRICS: dict[Scenario, dict[str, Any]] = {
    Scenario.CONNECTION_POOL_EXHAUSTION: {
        "status": "success",
        "service": "payment",
        "metrics": {
            "database_connections": {
                "active": 100,
                "pool_size": 100,
                "utilization_percent": 100,
            },
            "error_rate_percent": 12.5,
            "latency_ms": {"p50": 850, "p95": 2400, "p99": 5100},
            "requests_per_second": 125,
        },
    },
    Scenario.UPSTREAM_DEPENDENCY_FAILURE: {
        "status": "success",
        "service": "payment",
        "metrics": {
            "database_connections": {
                "active": 12,
                "pool_size": 100,
                "utilization_percent": 12,
            },
            "error_rate_percent": 34.0,
            "latency_ms": {"p50": 60, "p95": 10200, "p99": 10400},
            "requests_per_second": 118,
            "cpu_percent": 22,
            "memory_percent": 41,
            "upstream_dependencies": {
                "card-authorization-api": {
                    "error_rate_percent": 97.0,
                    "latency_ms": {"p50": 10000, "p95": 10400},
                    "circuit_breaker": "open",
                },
                "ledger-api": {
                    "error_rate_percent": 0.1,
                    "latency_ms": {"p50": 35, "p95": 80},
                    "circuit_breaker": "closed",
                },
            },
        },
    },
}

_POD_STATUS: dict[Scenario, dict[str, Any]] = {
    Scenario.CONNECTION_POOL_EXHAUSTION: {
        "status": "success",
        "service": "payment",
        "pods": [
            {
                "name": "payment-api-7d8f9c6b4-x1a2b",
                "status": "Running",
                "ready": False,
                "restart_count": 5,
            },
            {
                "name": "payment-api-7d8f9c6b4-y3c4d",
                "status": "Running",
                "ready": False,
                "restart_count": 4,
            },
            {
                "name": "payment-api-7d8f9c6b4-z5e6f",
                "status": "Running",
                "ready": True,
                "restart_count": 1,
            },
        ],
    },
    Scenario.UPSTREAM_DEPENDENCY_FAILURE: {
        "status": "success",
        "service": "payment",
        "pods": [
            {
                "name": "payment-api-7d8f9c6b4-x1a2b",
                "status": "Running",
                "ready": True,
                "restart_count": 0,
            },
            {
                "name": "payment-api-7d8f9c6b4-y3c4d",
                "status": "Running",
                "ready": True,
                "restart_count": 0,
            },
            {
                "name": "payment-api-7d8f9c6b4-z5e6f",
                "status": "Running",
                "ready": True,
                "restart_count": 0,
            },
        ],
    },
}


def _fixture(
    table: dict[Scenario, dict[str, Any]],
    service: str,
    scenario: Scenario | str,
) -> str:
    """Look up a fixture, or return a structured error for unknown inputs."""
    if not _is_known_service(service):
        return _error_response(service)

    try:
        selected = Scenario(scenario)
    except ValueError:
        return json.dumps(
            {
                "status": "error",
                "error_type": "scenario_not_found",
                "scenario": str(scenario),
                "available_scenarios": [item.value for item in Scenario],
            }
        )

    return json.dumps(table[selected])


def get_mock_logs(
    service: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> str:
    """
    Return recent application logs for a service.

    Use this to find error messages, exceptions and failure details that
    describe what is going wrong.

    Args:
        service: Name of the service whose logs should be retrieved.
            Must be one of: "payment".
        scenario: Which mock incident to return. Not part of the real tool
            contract; present only so fixtures can be selected in tests.

    Returns:
        A JSON-formatted string of log entries. If the service is unknown, an
        error-shaped JSON response listing the available services.
    """
    return _fixture(_LOGS, service, scenario)


def get_mock_metrics(
    service: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> str:
    """
    Return service metrics.

    Includes database connection pool usage, error rate, latency percentiles,
    throughput, and the health of upstream dependencies the service calls.
    Use this to quantify the problem and to tell whether the fault lies in
    this service or in something it depends on.

    Args:
        service: Name of the service whose metrics should be retrieved.
            Must be one of: "payment".
        scenario: Which mock incident to return. Not part of the real tool
            contract; present only so fixtures can be selected in tests.

    Returns:
        A JSON-formatted string of metric observations. If the service is
        unknown, an error-shaped JSON response listing the available services.
    """
    return _fixture(_METRICS, service, scenario)


def get_mock_pod_status(
    service: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> str:
    """
    Return Kubernetes pod health for a service.

    Includes readiness and restart counts. Use this to check whether the
    service's own instances are healthy, which helps distinguish a fault in
    this service from a fault elsewhere.

    Args:
        service: Name of the service whose pod status should be retrieved.
            Must be one of: "payment".
        scenario: Which mock incident to return. Not part of the real tool
            contract; present only so fixtures can be selected in tests.

    Returns:
        A JSON-formatted string of pod health observations. If the service is
        unknown, an error-shaped JSON response listing the available services.
    """
    return _fixture(_POD_STATUS, service, scenario)
