"""MCP server tests, run over the SDK's in-memory transport (no subprocess)."""

import asyncio
import json

from mcp import Client

from agentic_ai.mcp_server.server import SCENARIO_ENV, build_server
from agentic_ai.tools.mock_tools import AVAILABLE_SERVICES, Scenario
from agentic_ai.tools.registry import INVESTIGATION_TOOLS, build_tool_schemas


def _run(coro):
    return asyncio.run(coro)


def _with_client(scenario, action):
    async def go():
        async with Client(build_server(scenario)) as client:
            return await action(client)

    return _run(go())


def _service_values(prop: dict) -> set[str]:
    if "const" in prop:
        return {prop["const"]}
    return set(prop["enum"])


def test_server_exposes_exactly_the_registry_tools() -> None:
    result = _with_client(None, lambda c: c.list_tools())
    assert {t.name for t in result.tools} == set(INVESTIGATION_TOOLS)


def test_advertised_schemas_match_the_registry() -> None:
    """Guards against the hand-written MCP signatures drifting from the registry."""
    result = _with_client(None, lambda c: c.list_tools())
    exposed = {t.name: t for t in result.tools}

    for schema in build_tool_schemas():
        spec = schema["function"]
        tool = exposed[spec["name"]]
        advertised = tool.input_schema

        assert tool.description == spec["description"]
        assert set(advertised["properties"]) == set(spec["parameters"]["properties"])
        assert set(advertised["required"]) == set(spec["parameters"]["required"])
        if "service" in advertised["properties"]:
            assert _service_values(advertised["properties"]["service"]) == set(
                AVAILABLE_SERVICES
            )


def test_every_tool_is_marked_read_only() -> None:
    result = _with_client(None, lambda c: c.list_tools())
    assert all(t.annotations.read_only_hint for t in result.tools)
    assert not any(t.annotations.destructive_hint for t in result.tools)


def test_tool_call_returns_the_same_output_as_in_process_dispatch() -> None:
    from agentic_ai.tools.registry import dispatch_tool

    result = _with_client(
        Scenario.UPSTREAM_DEPENDENCY_FAILURE,
        lambda c: c.call_tool("get_mock_logs", {"service": "payment"}),
    )

    assert not result.is_error
    assert result.content[0].text == dispatch_tool(
        "get_mock_logs", {"service": "payment"}, Scenario.UPSTREAM_DEPENDENCY_FAILURE
    )


def test_client_cannot_choose_the_scenario() -> None:
    result = _with_client(
        Scenario.CONNECTION_POOL_EXHAUSTION,
        lambda c: c.call_tool(
            "get_mock_metrics", {"service": "payment", "scenario": "healthy"}
        ),
    )

    metrics = json.loads(result.content[0].text)["metrics"]
    assert metrics["database_connections"]["utilization_percent"] >= 100


def test_scenario_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(SCENARIO_ENV, "healthy")
    result = _with_client(
        None, lambda c: c.call_tool("get_mock_metrics", {"service": "payment"})
    )
    metrics = json.loads(result.content[0].text)["metrics"]
    assert metrics["database_connections"]["utilization_percent"] < 100


def test_unknown_service_is_rejected_as_a_tool_error() -> None:
    result = _with_client(
        None, lambda c: c.call_tool("get_mock_logs", {"service": "nope"})
    )
    assert result.is_error


def test_knowledge_search_works_over_mcp() -> None:
    result = _with_client(
        None,
        lambda c: c.call_tool(
            "search_knowledge_base",
            {"query": "database connection pool exhausted", "top_k": 1},
        ),
    )
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "success"
    assert payload["results"][0]["source"] == "curated"
