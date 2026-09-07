import json
from enum import Enum


class MockService(str, Enum):
    """Services for which mock incident data is available."""

    PAYMENT = "payment"


AVAILABLE_SERVICES = [service.value for service in MockService]


def _error_response(service: str) -> str:
    """Return a deterministic JSON error response for an unknown service."""
    return json.dumps(
        {
            "status": "error",
            "error_type": "service_not_found",
            "service": service,
            "available_services": AVAILABLE_SERVICES,
            "message": (
                f"No mock data is available for '{service}'. "
                f"Available services: {', '.join(AVAILABLE_SERVICES)}."
            ),
        }
    )


def _is_known_service(service: str) -> bool:
    """Return True if mock data exists for the given service name."""
    return service in AVAILABLE_SERVICES


def get_mock_logs(service: str) -> str:
    """
    Return deterministic mock application logs for an incident investigation.

    This tool is intended for an AI investigation agent. The returned JSON
    represents log observations that the agent can use as evidence when
    diagnosing a service incident.

    Args:
        service: Name of the service whose logs should be retrieved.
            Must be one of: "payment".

    Returns:
        A JSON-formatted string containing mock log observations. If the
        service is unknown, an error-shaped JSON response is returned that
        lists the available service names.
    """
    if not _is_known_service(service):
        return _error_response(service)

    return json.dumps(
        {
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
        }
    )


def get_mock_metrics(service: str) -> str:
    """
    Return deterministic mock metrics for an incident investigation.

    This tool is intended for an AI investigation agent. The returned JSON
    represents service health and database connection metrics that the agent
    can use as evidence when diagnosing an incident.

    Args:
        service: Name of the service whose metrics should be retrieved.
            Must be one of: "payment".

    Returns:
        A JSON-formatted string containing mock metric observations. If the
        service is unknown, an error-shaped JSON response is returned that
        lists the available service names.
    """
    if not _is_known_service(service):
        return _error_response(service)

    return json.dumps(
        {
            "status": "success",
            "service": "payment",
            "metrics": {
                "database_connections": {
                    "active": 100,
                    "pool_size": 100,
                    "utilization_percent": 100,
                },
                "error_rate_percent": 12.5,
                "latency_ms": {
                    "p50": 850,
                    "p95": 2400,
                    "p99": 5100,
                },
                "requests_per_second": 125,
            },
        }
    )


def get_mock_pod_status(service: str) -> str:
    """
    Return deterministic mock Kubernetes pod health information.

    This tool is intended for an AI investigation agent. The returned JSON
    represents pod health and restart observations that the agent can use as
    evidence when diagnosing an incident.

    Args:
        service: Name of the service whose pod status should be retrieved.
            Must be one of: "payment".

    Returns:
        A JSON-formatted string containing mock pod health observations. If
        the service is unknown, an error-shaped JSON response is returned that
        lists the available service names.
    """
    if not _is_known_service(service):
        return _error_response(service)

    return json.dumps(
        {
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
        }
    )