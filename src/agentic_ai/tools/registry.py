"""Tool registry: one source of truth for what the agent can call.

The schema handed to the LLM and the function actually dispatched are derived
from the same registry entry, so they cannot drift apart.
"""

from collections.abc import Callable
from typing import Any

from agentic_ai.tools.mock_tools import (
    AVAILABLE_SERVICES,
    DEFAULT_SCENARIO,
    Scenario,
    get_mock_logs,
    get_mock_metrics,
    get_mock_pod_status,
)
from agentic_ai.tools.search_tool import DEFAULT_TOP_K, search_knowledge_base

_SERVICE_PARAMETERS: dict[str, Any] = {
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

_KNOWLEDGE_BASE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A description of the current symptoms, written the way an "
                "engineer would describe the problem."
            ),
        },
        "top_k": {
            "type": "integer",
            "description": "How many past incidents to return.",
            "default": DEFAULT_TOP_K,
        },
    },
    "required": ["query"],
}


class ToolNotFoundError(KeyError):
    """Raised when the model requests a tool that does not exist."""


# A scenario-aware tool receives the active mock scenario as a keyword
# argument, in addition to whatever the model supplied; a plain tool receives
# only what the model supplied. Tools differ in which world they read from -
# the mock observability tools read whichever incident scenario is active,
# while the knowledge base is one real (synthetic) corpus that exists
# independently of any scenario - so dispatch must not force scenario onto
# every tool uniformly.
INVESTIGATION_TOOLS: dict[str, Callable[..., str]] = {
    "get_mock_logs": get_mock_logs,
    "get_mock_metrics": get_mock_metrics,
    "get_mock_pod_status": get_mock_pod_status,
    "search_knowledge_base": search_knowledge_base,
}

_SCENARIO_AWARE_TOOLS: frozenset[str] = frozenset(
    {"get_mock_logs", "get_mock_metrics", "get_mock_pod_status"}
)

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
    "search_knowledge_base": (
        "Search past-incident runbooks for symptoms similar to the current "
        "incident. Use this when the symptoms resemble a problem that may "
        "have happened before; it returns likely root causes to confirm "
        "against the actual evidence, not a confirmed diagnosis."
    ),
}

_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "get_mock_logs": _SERVICE_PARAMETERS,
    "get_mock_metrics": _SERVICE_PARAMETERS,
    "get_mock_pod_status": _SERVICE_PARAMETERS,
    "search_knowledge_base": _KNOWLEDGE_BASE_PARAMETERS,
}


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return the tool definitions in the shape Ollama's ``tools=`` expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_DESCRIPTIONS[name],
                "parameters": _TOOL_PARAMETERS[name],
            },
        }
        for name in INVESTIGATION_TOOLS
    ]


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> str:
    """Call a registered tool by name.

    ``scenario`` selects which mock fixture set the observability tools draw
    from. It is supplied by the caller, never by the model - the schema for
    those tools advertises only ``service``, so the agent cannot choose its
    own reality. Tools that are not scenario-aware (the knowledge base) never
    receive it, since it is meaningless to them.

    Raises:
        ToolNotFoundError: if the model hallucinated a tool that does not
            exist. This is a *protocol* error, distinct from a tool that runs
            and reports a domain failure in its result.
    """
    if name not in INVESTIGATION_TOOLS:
        raise ToolNotFoundError(name)

    # Ignore any 'scenario' the model may have invented; it is not ours to
    # take from the model's arguments.
    safe_arguments = {
        key: value for key, value in arguments.items() if key != "scenario"
    }

    if name in _SCENARIO_AWARE_TOOLS:
        safe_arguments["scenario"] = scenario

    return INVESTIGATION_TOOLS[name](**safe_arguments)
