"""Tests for the agentic-ai-demo CLI (src/agentic_ai/cli.py).

All tests mock ``handle_incident`` so they run fully offline without LLM
credentials, external services, or incident-memory side-effects.

Since this project uses typer 0.27.x which has a single-command app (the
``demo`` function is the default command), tests invoke ``app`` with ``[]``
rather than ``["demo"]``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentic_ai.cli import app
from agentic_ai.observability.events import (
    DiagnosticReason,
    IncidentTrace,
    StageStatus,
    StageTrace,
    ToolInvocationTrace,
)
from agentic_ai.tools.mock_tools import Scenario

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _make_stage(name: str = "investigate") -> StageTrace:
    now = datetime.now(_UTC)
    return StageTrace(
        stage_name=name,
        status=StageStatus.COMPLETED,
        start_time=now,
        end_time=now,
        duration_ms=10.0,
    )


def _make_tool() -> ToolInvocationTrace:
    return ToolInvocationTrace(
        tool_name="get_mock_metrics",
        duration_ms=5.0,
        status="success",
    )


def _make_trace(outcome: str = "escalate") -> IncidentTrace:
    diag = DiagnosticReason(
        decision=outcome,  # type: ignore[arg-type]
        primary_reason="connection pool exhausted",
        trigger_type="evidence_metrics",
    )
    return IncidentTrace(
        incident_id="INC-test01",
        correlation_id="CORR-test01",
        final_outcome=outcome,  # type: ignore[arg-type]
        diagnostic=diag,
        stages=[_make_stage()],
        tools=[_make_tool()],
        total_duration_ms=350.0,
    )


def _make_result(outcome: str = "escalate") -> dict[str, Any]:
    """Return a deterministic workflow result dict."""
    severity_mock = MagicMock()
    severity_mock.severity.value = "HIGH"

    verification_mock = MagicMock()
    verification_mock.status.value = "unresolved"
    verification_mock.attempts = 2

    return {
        "incident_id": "INC-test01",
        "correlation_id": "CORR-test01",
        "next_step": outcome,
        "severity_assessment": severity_mock,
        "verification": verification_mock,
        "trace": _make_trace(outcome),
    }


# ---------------------------------------------------------------------------
# Basic invocation — single-command app: invoke with [] not ["demo"]
# ---------------------------------------------------------------------------


def test_demo_default_scenario_succeeds():
    """Running the app with no arguments completes with exit code 0."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("escalate")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "ESCALATED" in result.output


def test_demo_prints_incident_id():
    """The demo output must contain the incident ID from the workflow result."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("resolve")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "INC-test01" in result.output


def test_demo_resolve_outcome():
    """A resolve outcome shows the resolved label."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("resolve")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "RESOLVED" in result.output


def test_demo_monitor_outcome():
    """A monitor outcome shows the monitor label."""
    result_dict = _make_result("monitor")
    result_dict["verification"] = None  # monitor path has no verification
    with patch("agentic_ai.cli.handle_incident", return_value=result_dict):
        result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "MONITOR" in result.output


# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------


def test_demo_connection_pool_scenario():
    """Passing --scenario connection_pool_exhaustion without description uses its default description."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result()) as mock_hi:
        result = runner.invoke(app, ["--scenario", "connection_pool_exhaustion"])
    assert result.exit_code == 0, result.output
    mock_hi.assert_called_once()
    _, kwargs = mock_hi.call_args
    assert kwargs["scenario"] == Scenario.CONNECTION_POOL_EXHAUSTION
    assert "database connection timeouts" in kwargs["incident_description"]


def test_demo_healthy_scenario():
    """Passing --scenario healthy without description uses benign default description and shows MONITOR."""
    healthy_result = _make_result("monitor")
    healthy_result["verification"] = None
    healthy_result["severity_assessment"].severity.value = "low"
    with patch("agentic_ai.cli.handle_incident", return_value=healthy_result) as mock_hi:
        result = runner.invoke(app, ["--scenario", "healthy"])
    assert result.exit_code == 0, result.output
    mock_hi.assert_called_once()
    _, kwargs = mock_hi.call_args
    assert kwargs["scenario"] == Scenario.HEALTHY
    assert kwargs["incident_description"] == (
        "Routine health check for payment service; minor retry warning logged."
    )
    assert "MONITOR" in result.output
    assert "low" in result.output


def test_demo_upstream_dependency_scenario():
    """Passing --scenario upstream_dependency_failure without description uses matching default description."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result()) as mock_hi:
        result = runner.invoke(app, ["--scenario", "upstream_dependency_failure"])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_hi.call_args
    assert kwargs["scenario"] == Scenario.UPSTREAM_DEPENDENCY_FAILURE
    assert kwargs["incident_description"] == (
        "Elevated 503 errors and payment processing timeouts reported on payment-service."
    )


def test_demo_healthy_scenario_with_critical_description_escalates():
    """When a healthy scenario is explicitly given a critical description, the workflow escalates."""
    critical_desc = "Critical payment gateway failure: 100% of transactions failing"
    critical_result = _make_result("escalate")
    critical_result["severity_assessment"].severity.value = "CRITICAL"
    with patch("agentic_ai.cli.handle_incident", return_value=critical_result) as mock_hi:
        result = runner.invoke(app, ["--scenario", "healthy", "--description", critical_desc])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_hi.call_args
    assert kwargs["scenario"] == Scenario.HEALTHY
    assert kwargs["incident_description"] == critical_desc
    assert "ESCALATED" in result.output
    assert "CRITICAL" in result.output


def test_demo_invalid_scenario_exits_with_error():
    """An unrecognised scenario name must exit with a non-zero code and print an error."""
    result = runner.invoke(app, ["--scenario", "not_a_real_scenario"])
    assert result.exit_code != 0
    assert "invalid scenario" in result.output.lower()


# ---------------------------------------------------------------------------
# Custom description
# ---------------------------------------------------------------------------


def test_demo_custom_description_is_passed_to_workflow():
    """A custom --description is forwarded verbatim to handle_incident."""
    custom = "Disk full on node-42"
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result()) as mock_hi:
        result = runner.invoke(app, ["--description", custom])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_hi.call_args
    assert kwargs["incident_description"] == custom


# ---------------------------------------------------------------------------
# Post-remediation scenario
# ---------------------------------------------------------------------------


def test_demo_post_remediation_scenario():
    """A valid --post-remediation-scenario is forwarded to handle_incident."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result()) as mock_hi:
        result = runner.invoke(app, ["--post-remediation-scenario", "healthy"])
    assert result.exit_code == 0, result.output
    _, kwargs = mock_hi.call_args
    assert kwargs["post_remediation_scenario"] == Scenario.HEALTHY


def test_demo_invalid_post_remediation_scenario_exits():
    """An invalid --post-remediation-scenario must produce a non-zero exit."""
    result = runner.invoke(app, ["--post-remediation-scenario", "explode_everything"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Error handling — workflow failure
# ---------------------------------------------------------------------------


def test_demo_workflow_exception_is_caught():
    """If handle_incident raises, the CLI must exit with code 2 without
    printing the raw exception message (which might leak secrets)."""
    with patch("agentic_ai.cli.handle_incident", side_effect=RuntimeError("boom")):
        result = runner.invoke(app, [])
    assert result.exit_code == 2
    # The raw exception message "boom" must not appear in stdout
    assert "boom" not in result.output


# ---------------------------------------------------------------------------
# Trace and diagnostic fields
# ---------------------------------------------------------------------------


def test_demo_prints_diagnostic_fields():
    """Diagnostic trigger type and primary reason are shown in output."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("escalate")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "evidence_metrics" in result.output
    assert "connection pool exhausted" in result.output


def test_demo_prints_trace_summary():
    """Stage count, tool count, and total duration are printed."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("escalate")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "1" in result.output  # stage count
    assert "350.0 ms" in result.output


# ---------------------------------------------------------------------------
# Secret / credential redaction
# ---------------------------------------------------------------------------


def test_demo_output_does_not_contain_api_key_patterns(monkeypatch):
    """If an API key is present in the environment, it must never appear in output."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-deadbeef1234")
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("resolve")):
        result = runner.invoke(app, [])
    assert "sk-or-secret-deadbeef1234" not in result.output


def test_demo_output_does_not_echo_description_fully_when_truncated():
    """Descriptions longer than 80 chars are truncated in the preamble."""
    long_desc = "X" * 200
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("escalate")):
        result = runner.invoke(app, ["--description", long_desc])
    assert result.exit_code == 0, result.output
    # The preamble should show at most 80 chars + ellipsis, not the full 200
    assert "X" * 200 not in result.output
    assert "..." in result.output


def test_demo_output_encodable_in_cp1252():
    """Verify entire CLI output can be encoded in cp1252 (Windows default)."""
    with patch("agentic_ai.cli.handle_incident", return_value=_make_result("resolve")):
        result = runner.invoke(app, [])
    assert result.exit_code == 0, result.output
    encoded = result.output.encode("cp1252")
    assert isinstance(encoded, bytes)
