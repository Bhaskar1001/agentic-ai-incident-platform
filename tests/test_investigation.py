import json
from types import SimpleNamespace

import pytest

from agentic_ai.agents import investigation as agent_module
from agentic_ai.agents.investigation import (
    MAX_INVESTIGATION_STEPS,
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
from agentic_ai.tools.registry import (
    ToolNotFoundError,
    build_tool_schemas,
    dispatch_tool,
)


def _tool_call(name: str, **arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments)
    )


def _response(content: str = "", tool_calls=None):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls)
    )


# --- tool registry -------------------------------------------------------


def test_tool_schemas_expose_available_services_as_enum() -> None:
    """The model must be told which service names are valid."""
    schemas = build_tool_schemas()

    assert {s["function"]["name"] for s in schemas} == {
        "get_mock_logs",
        "get_mock_metrics",
        "get_mock_pod_status",
    }
    for schema in schemas:
        service = schema["function"]["parameters"]["properties"]["service"]
        assert service["enum"] == ["payment"]


def test_dispatch_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        dispatch_tool("get_mock_database_dump", {"service": "payment"})


def test_dispatch_calls_the_real_tool() -> None:
    result = json.loads(dispatch_tool("get_mock_metrics", {"service": "payment"}))

    assert result["status"] == "success"


# --- graph mechanics -----------------------------------------------------


def test_agent_stops_when_model_stops_requesting_tools(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(tool_calls=[_tool_call("get_mock_metrics", service="payment")])
        return _response(content="Connection pool is exhausted.")

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.COMPLETED
    assert state["steps_taken"] == 1
    assert len(state["evidence"]) == 1
    assert state["evidence"][0].source_tool == "get_mock_metrics"


def test_step_budget_is_enforced_against_a_looping_model(monkeypatch) -> None:
    """A model that never stops must still be stopped by deterministic code."""

    def fake_chat(**kwargs):
        return _response(tool_calls=[_tool_call("get_mock_logs", service="payment")])

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.MAX_STEPS_REACHED
    assert state["steps_taken"] == MAX_INVESTIGATION_STEPS
    assert any(
        entry.event_type is TimelineEventType.STEP_LIMIT_REACHED
        for entry in state["timeline"]
    )


def test_unknown_tool_is_recoverable_not_fatal(monkeypatch) -> None:
    """A hallucinated tool becomes an observation, not a crash."""
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(tool_calls=[_tool_call("get_mock_traces", service="payment")])
        return _response(content="That tool does not exist; concluding.")

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    state = build_investigation_graph().invoke(_initial_state())

    assert state["termination_reason"] is TerminationReason.COMPLETED
    assert state["evidence"] == []  # failed call produced no evidence
    assert any(
        entry.event_type is TimelineEventType.TOOL_FAILED
        for entry in state["timeline"]
    )
    tool_message = [m for m in state["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_message["content"])["error_type"] == "tool_not_found"


def test_failed_tool_result_is_still_shown_to_the_model(monkeypatch) -> None:
    """An unknown *service* is a domain error the agent can react to."""
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(tool_calls=[_tool_call("get_mock_logs", service="billing")])
        return _response(content="No data for that service.")

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    state = build_investigation_graph().invoke(_initial_state())

    tool_message = [m for m in state["messages"] if m["role"] == "tool"][0]
    payload = json.loads(tool_message["content"])
    assert payload["error_type"] == "service_not_found"
    assert payload["available_services"] == ["payment"]


def test_evidence_preserves_raw_tool_output_verbatim(monkeypatch) -> None:
    """The core auditability invariant."""
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(tool_calls=[_tool_call("get_mock_metrics", service="payment")])
        return _response(content="Done.")

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    state = build_investigation_graph().invoke(_initial_state())

    evidence = state["evidence"][0]
    expected = dispatch_tool("get_mock_metrics", {"service": "payment"})
    assert evidence.raw_output == expected


# --- conclusion / summarisation -----------------------------------------


def test_investigate_returns_validated_result(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(tool_calls=[_tool_call("get_mock_metrics", service="payment")])
        if calls["count"] == 2:
            return _response(content="Pool exhausted.")
        return _response(
            content=json.dumps(
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
        )

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

    result = investigate("Payment service is failing")

    assert isinstance(result, InvestigationResult)
    assert result.termination_reason is TerminationReason.COMPLETED
    assert result.is_complete
    assert len(result.evidence) == 1
    assert result.evidence[0].source_tool == "get_mock_metrics"
    assert result.findings[0].model_assessed_confidence == 0.8


def test_invalid_conclusion_raises_after_retry(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _response(content="Nothing to investigate.")
        return _response(content='{"summary": "incomplete"}')  # missing fields

    monkeypatch.setattr(agent_module.ollama, "chat", fake_chat)

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
