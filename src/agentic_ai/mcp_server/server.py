"""MCP server exposing the investigation tools to any MCP client.

Every call goes through ``registry.dispatch_tool``, so the tools behave
identically whether the investigation agent calls them in-process or an
external client (an IDE, another agent) calls them over MCP.

Two boundaries are held here:

- **Scenario is the operator's, not the client's.** The mock observability
  tools read from a scenario chosen by ``AGENTIC_MCP_SCENARIO`` when the
  server starts. The tool schemas advertise only ``service``, and dispatch
  ignores any ``scenario`` argument a client sends - the same rule that stops
  the LLM choosing its own reality.
- **Read-only.** Only the four read-only investigation tools are exposed.
  Remediation actions are deliberately absent: nothing that changes the world
  should be reachable over a protocol before the Phase 10 guardrails exist.

The MCP-facing signatures are written out by hand (the SDK derives the input
schema from them), so a test compares the advertised schemas against the
registry to catch drift.
"""

from __future__ import annotations

import os
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.tools.registry import _TOOL_DESCRIPTIONS, dispatch_tool
from agentic_ai.tools.search_tool import DEFAULT_TOP_K

SCENARIO_ENV = "AGENTIC_MCP_SCENARIO"

ServiceName = Literal["payment"]  # mirrors AVAILABLE_SERVICES; drift is tested

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def active_scenario() -> Scenario:
    return Scenario(os.environ.get(SCENARIO_ENV, DEFAULT_SCENARIO.value))


def build_server(scenario: Scenario | None = None) -> MCPServer:
    """Build the server. ``scenario`` fixes the mock world for its lifetime."""
    fixed = scenario or active_scenario()
    server = MCPServer("agentic-ai-investigation-tools")

    def get_mock_logs(service: ServiceName) -> str:
        return dispatch_tool("get_mock_logs", {"service": service}, fixed)

    def get_mock_metrics(service: ServiceName) -> str:
        return dispatch_tool("get_mock_metrics", {"service": service}, fixed)

    def get_mock_pod_status(service: ServiceName) -> str:
        return dispatch_tool("get_mock_pod_status", {"service": service}, fixed)

    def search_knowledge_base(query: str, top_k: int = DEFAULT_TOP_K) -> str:
        return dispatch_tool("search_knowledge_base", {"query": query, "top_k": top_k}, fixed)

    for fn in (get_mock_logs, get_mock_metrics, get_mock_pod_status, search_knowledge_base):
        server.add_tool(
            fn,
            name=fn.__name__,
            description=_TOOL_DESCRIPTIONS[fn.__name__],
            annotations=_READ_ONLY,
            structured_output=False,
        )

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
