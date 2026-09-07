"""Tool registry: one source of truth for what the agent can call.

The schema handed to the LLM and the function actually dispatched are derived
from the same registry entry, so they cannot drift apart.
"""

from collections.abc import Callable
from typing import Any

from agentic_ai.tools.mock_tools import (
    AVAILABLE_SERVICES,
    get_mock_logs,
    get_mock_metrics,
    get_mock_pod_status,
)

_SERVICE_PARAMETER: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service": {
            "type": "string",
            "enum": list(AVAILABLE_SERVICES),
            "description": "Name of the service to inspect.",
        }
    },
    "required": ["service"],
}


class ToolNotFoundError(KeyError):
    """Raised when the model requests a tool that does not exist."""


INVESTIGATION_TOOLS: dict[str, Callable[..., str]] = {
    "get_mock_logs": get_mock_logs,
    "get_mock_metrics": get_mock_metrics,
    "get_mock_pod_status": get_mock_pod_status,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_mock_logs": (
        "Retrieve recent application logs for a service. Use this to find "
        "error messages, exceptions and failure details."
    ),
    "get_mock_metrics": (
        "Retrieve service metrics: database connection pool usage, error "
        "rate, latency percentiles and throughput. Use this to see "
        "quantitative signals of degradation."
    ),
    "get_mock_pod_status": (
        "Retrieve Kubernetes pod health: readiness and restart counts. Use "
        "this to check whether the service's instances are healthy."
    ),
}


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return the tool definitions in the shape Ollama's ``tools=`` expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_DESCRIPTIONS[name],
                "parameters": _SERVICE_PARAMETER,
            },
        }
        for name in INVESTIGATION_TOOLS
    ]


def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Call a registered tool by name.

    Raises:
        ToolNotFoundError: if the model hallucinated a tool that does not
            exist. This is a *protocol* error, distinct from a tool that runs
            and reports a domain failure in its result.
    """
    if name not in INVESTIGATION_TOOLS:
        raise ToolNotFoundError(name)

    return INVESTIGATION_TOOLS[name](**arguments)
