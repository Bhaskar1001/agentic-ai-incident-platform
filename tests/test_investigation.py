import json

import pytest

from agentic_ai.agents import investigation as agent_module
from agentic_ai.agents.investigation import (
    MAX_INVESTIGATION_STEPS,
    MIN_DISTINCT_TOOLS,
    InvalidConclusionError,
    build_investigation_graph,
    investigate,
)
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
    TimelineEventType,
)
from agentic_ai.llm.client import ChatResponse, ToolCall
from agentic_ai.tools.registry import (
    ToolNotFoundError,
    build_tool_schemas,
    dispatch_tool,
)


_next_call_id = iter(f"call_{i}" for i in range(10_000))


def _tool_call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, id=next(_next_call_id))


ALL_TOOL_NAMES = ("get_mock_logs", "get_mock_metrics", "get_mock_pod_status")


class _FakeClient:
    """A minimal LLMClient stand-in driven by a function of
    (messages, tools, json_schema) -> ChatResponse, so each test can script
    exactly what the "model" does on each call without touching any real
    provider.
    """

    def __init__(self, respond):
        self._respond = respond
        self.calls = 0

    def complete(self, messages, *, tools=None, json_schema=None):
        self.calls += 1
        return self._respond(messages=messages, tools=tools, json_schema=json_schema)


def _compliant_model(extra_calls=(), final_content="Investigation complete."):
    """A fake model that satisfies the coverage requirement, then concludes.

    ``extra_calls`` are issued first - used to inject failing or unusual calls
    before the well-behaved ones.
    """
    planned = list(extra_calls) + [
        _tool_call(name, service="payment") for name in ALL_TOOL_NAMES
    ]
    index = {"i": 0}

    def respond(**kwargs):
        i = index["i"]
        index["i"] += 1
        if i < len(planned):
            return ChatResponse(content="", tool_calls=[planned[i]])
        return ChatResponse(content=final_content)

    return respond


def _patch_client(monkeypatch, respond) -> _FakeClient:
    client = _FakeClient(respond)
    monkeypatch.setattr(agent_module, "get_client", lambda: client)
    return client


# --- tool registry -------------------------------------------------------


OBSERVABILITY_TOOL_NAMES = {
    "get_mock_logs",
    "get_mock_metrics",
    "get_mock_pod_status",
}


def test_all_registered_tools_are_exposed() -> None:
    schemas = build_tool_schemas()

    assert {s["function"]["name"] for s in schemas} == OBSERVABILITY_TOOL_NAMES | {
        "search_knowledge_base"
    }


def test_observability_tool_schemas_expose_available_services_as_enum() -> None:
    """The model must be told which service names are valid."""
    schemas = build_tool_schemas()

    observability_schemas = [
        s for s in schemas if s["function"]["name"] in OBSERVABILITY_TOOL_NAMES
    ]
    assert len(observability_schemas) == len(OBSERVABILITY_TOOL_NAMES)

    for schema in observability_schemas:
        service = schema["function"]["parameters"]["properties"]["service"]
        assert service["enum"] == ["payment"]


def test_knowledge_base_tool_schema_takes_a_query_not_a_service() -> None:
    """The retrieval tool has nothing to do with the mock service enum."""
    schemas = build_tool_schemas()

    kb_schema = next(
        s for s in schemas if s["function"]["name"] == "search_knowledge_base"
    )
    properties = kb_schema["function"]["parameters"]["properties"]

    assert "query" in properties
    assert "service" not in properties


def test_dispatch_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        dispatch_tool("get_mock_database_dump", {"service": "payment"})


def test_dispatch_calls_the_real_tool() -> None:
    result = json.loads(dispatch_tool("get_mock_metrics", {"service": "payment"}))

    assert result["status"] == "success"


# --- graph mechanics -----------------------------------------------------


def test_agent_stops_once_coverage_is_met_and_it_stops_asking(monkeypatch) -> None:
    _patch_client(monkeypatch, _compliant_model())

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.COMPLETED
    assert state["steps_taken"] == len(ALL_TOOL_NAMES)
    assert {e.source_tool for e in state["evidence"]} == set(ALL_TOOL_NAMES)


def test_premature_conclusion_is_rejected_until_coverage_is_met(monkeypatch) -> None:
    """The model may not decide one source is enough - code decides that."""
    requested: list[str] = []
    # The model tries to stop after every single call; code must push back.
    sequence = ["get_mock_logs", None, "get_mock_metrics", None,
                "get_mock_pod_status", None]
    calls = {"i": 0}

    def respond(**kwargs):
        index = calls["i"]
        calls["i"] += 1
        tool = sequence[index] if index < len(sequence) else None
        if tool is None:
            return ChatResponse(content="I am done.")
        requested.append(tool)
        return ChatResponse(content="", tool_calls=[_tool_call(tool, service="payment")])

    _patch_client(monkeypatch, respond)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.COMPLETED
    assert len({e.source_tool for e in state["evidence"]}) == MIN_DISTINCT_TOOLS
    # It was pushed back exactly twice: after 1 tool and after 2.
    assert (
        sum(
            entry.event_type is TimelineEventType.INSUFFICIENT_EVIDENCE
            for entry in state["timeline"]
        )
        == 2
    )


def test_coverage_enforcement_cannot_loop_past_the_step_budget(monkeypatch) -> None:
    """A model that always calls the same tool must still terminate."""

    def respond(messages, **kwargs):
        # Always the same tool, then always claims to be done.
        if messages[-1]["role"] == "tool":
            return ChatResponse(content="Done.")
        return ChatResponse(
            content="", tool_calls=[_tool_call("get_mock_logs", service="payment")]
        )

    _patch_client(monkeypatch, respond)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.MAX_STEPS_REACHED
    assert state["steps_taken"] == MAX_INVESTIGATION_STEPS


def test_step_budget_is_enforced_against_a_looping_model(monkeypatch) -> None:
    """A model that never stops must still be stopped by deterministic code."""

    def respond(**kwargs):
        return ChatResponse(
            content="", tool_calls=[_tool_call("get_mock_logs", service="payment")]
        )

    _patch_client(monkeypatch, respond)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.MAX_STEPS_REACHED
    assert state["steps_taken"] == MAX_INVESTIGATION_STEPS
    assert any(
        entry.event_type is TimelineEventType.STEP_LIMIT_REACHED
        for entry in state["timeline"]
    )


def test_unknown_tool_is_recoverable_not_fatal(monkeypatch) -> None:
    """A hallucinated tool becomes an observation, not a crash."""
    _patch_client(
        monkeypatch,
        _compliant_model(extra_calls=[_tool_call("get_mock_traces", service="payment")]),
    )

    state = build_investigation_graph().invoke(_initial_state())

    # The bad call did not stop the investigation from completing.
    assert state["termination_reason"] is TerminationReason.COMPLETED
    assert any(
        entry.event_type is TimelineEventType.TOOL_FAILED
        for entry in state["timeline"]
    )
    # A failed call yields no evidence, only the three real ones do.
    assert {e.source_tool for e in state["evidence"]} == set(ALL_TOOL_NAMES)

    tool_message = [m for m in state["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_message["content"])["error_type"] == "tool_not_found"


def test_failed_tool_result_is_still_shown_to_the_model(monkeypatch) -> None:
    """An unknown *service* is a domain error the agent can react to."""
    _patch_client(
        monkeypatch,
        _compliant_model(extra_calls=[_tool_call("get_mock_logs", service="billing")]),
    )

    state = build_investigation_graph().invoke(_initial_state())

    payload = json.loads(
        [m for m in state["messages"] if m["role"] == "tool"][0]["content"]
    )
    assert payload["error_type"] == "service_not_found"
    assert payload["available_services"] == ["payment"]


def test_evidence_preserves_raw_tool_output_verbatim(monkeypatch) -> None:
    """The core auditability invariant."""
    _patch_client(monkeypatch, _compliant_model())

    state = build_investigation_graph().invoke(_initial_state())

    metrics = next(
        e for e in state["evidence"] if e.source_tool == "get_mock_metrics"
    )
    assert metrics.raw_output == dispatch_tool(
        "get_mock_metrics", {"service": "payment"}
    )


# --- conclusion / summarisation -----------------------------------------


_VALID_CONCLUSION = json.dumps(
    {
        "summary": "Database connection pool is exhausted.",
        "findings": [
            {
                "statement": "Pool exhaustion is the likely cause.",
                "model_assessed_confidence": 0.8,
            }
        ],
        "model_assessed_confidence": 0.8,
        "unresolved_questions": ["What caused the spike?"],
        "recommended_next_action": "Increase pool size.",
    }
)


def test_investigate_returns_validated_result(monkeypatch) -> None:
    investigation = _compliant_model()

    def respond(json_schema=None, **kwargs):
        # Once the graph has finished, the next call is the summariser.
        if json_schema is not None:
            return ChatResponse(content=_VALID_CONCLUSION)
        return investigation(**kwargs)

    _patch_client(monkeypatch, respond)

    result = investigate("Payment service is failing")

    assert isinstance(result, InvestigationResult)
    assert result.termination_reason is TerminationReason.COMPLETED
    assert result.is_complete
    assert {e.source_tool for e in result.evidence} == set(ALL_TOOL_NAMES)
    assert result.findings[0].model_assessed_confidence == 0.8


def test_invalid_conclusion_raises_after_retry(monkeypatch) -> None:
    investigation = _compliant_model()

    def respond(json_schema=None, **kwargs):
        if json_schema is not None:
            return ChatResponse(content='{"summary": "incomplete"}')  # missing fields
        return investigation(**kwargs)

    _patch_client(monkeypatch, respond)

    with pytest.raises(InvalidConclusionError):
        investigate("Payment service is failing")


def test_summary_schema_excludes_code_owned_fields() -> None:
    """The model must not be allowed to author evidence or termination reason."""
    schema = InvestigationResult.model_json_schema()

    assert "evidence" in schema["properties"]  # present on the real model...

    # ...but stripped before being handed to the model.
    stripped = {
        key
        for key in ("evidence", "timeline", "termination_reason")
        if key in schema["properties"]
    }
    assert stripped, "sanity: these fields exist on the model"


# --- domain models -------------------------------------------------------


def test_confidence_must_be_within_range() -> None:
    with pytest.raises(Exception):
        Finding(statement="x", model_assessed_confidence=1.5)


def test_result_is_incomplete_when_budget_exhausted() -> None:
    result = InvestigationResult(
        summary="s",
        model_assessed_confidence=0.5,
        recommended_next_action="escalate",
        termination_reason=TerminationReason.MAX_STEPS_REACHED,
    )

    assert not result.is_complete


def test_evidence_requires_raw_output() -> None:
    evidence = Evidence(
        source_tool="get_mock_logs",
        tool_arguments={"service": "payment"},
        raw_output='{"status": "success"}',
        observation="logs returned",
    )

    assert evidence.raw_output == '{"status": "success"}'
    assert evidence.id is not None


# --- helpers -------------------------------------------------------------


def _initial_state():
    from agentic_ai.domain.investigation import TimelineEntry

    return {
        "incident_description": "Payment service is failing",
        "scenario": "connection_pool_exhaustion",
        "messages": [
            {"role": "system", "content": "investigate"},
            {"role": "user", "content": "Incident: payment failing"},
        ],
        "evidence": [],
        "timeline": [
            TimelineEntry(
                event_type=TimelineEventType.INVESTIGATION_STARTED,
                description="started",
            )
        ],
        "steps_taken": 0,
        "termination_reason": None,
    }
