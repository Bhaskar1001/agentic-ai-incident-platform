"""Tests for Phase 14 Step 2: Operational Tracer & Tool-Invocation Tracing.

Verifies:
- StructuredJsonFormatter emits valid, single-line JSON log lines.
- UTC ISO-8601 timestamps, log levels, and correlation identifiers are present.
- Secrets embedded in messages, extra payloads, and exception tracebacks are redacted.
- Stage spans measure positive duration via perf_counter, capture start/end, and record status.
- Tool spans capture tool name, target service, duration, sanitized arguments, and status.
- Exceptions in stages and tools are categorized and cleanly re-raised.
- Trace snapshots are immutable and isolated from subsequent span additions.
- Safe fallback behavior prevents logging formatter errors from crashing execution.
- Tracer uses an isolated logger and never modifies, reconfigures, or replaces the root logger.
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_ai.observability.events import (
    DiagnosticReason,
    ErrorCategory,
    IncidentTrace,
    StageStatus,
)
from agentic_ai.observability.redaction import REDACTED_PLACEHOLDER
from agentic_ai.observability.tracer import (
    OBSERVABILITY_LOGGER_NAME,
    OperationalTracer,
    StageSpanHandle,
    StructuredJsonFormatter,
    ToolSpanHandle,
)


class LogCaptureHandler(logging.Handler):
    """Test helper collecting formatted log records in memory."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.formatted_lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self.formatter:
            self.formatted_lines.append(self.formatter.format(record))


# --- 1. StructuredJsonFormatter Tests -----------------------------------------


def test_structured_json_formatter_single_line_json() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Service health check passed for %s",
        args=("payment",),
        exc_info=None,
    )
    line = formatter.format(record)

    # Must be valid single-line JSON without unescaped newlines
    assert "\n" not in line
    data = json.loads(line)
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Service health check passed for payment"
    assert "timestamp" in data
    # Verify ISO-8601 UTC timestamp format
    assert data["timestamp"].endswith("+00:00") or data["timestamp"].endswith("Z")


def test_structured_json_formatter_redacts_embedded_secrets_in_message() -> None:
    formatter = StructuredJsonFormatter()
    raw_msg = (
        "Failed authenticating with Bearer sk-ant-api03-secret1234567890 "
        "and db password=supersecretpass"
    )
    record = logging.LogRecord(
        name="auth_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=10,
        msg=raw_msg,
        args=(),
        exc_info=None,
    )
    line = formatter.format(record)
    data = json.loads(line)

    assert "secret1234567890" not in data["message"]
    assert "supersecretpass" not in data["message"]
    assert "Bearer [REDACTED]" in data["message"]
    assert "password=[REDACTED]" in data["message"]


def test_structured_json_formatter_redacts_extra_attributes() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="User session initiated",
        args=(),
        exc_info=None,
    )
    # Attach extra sensitive metadata
    record.__dict__["apiKey"] = "sk-1234567890abcdef1234567890"
    record.__dict__["user_config"] = {
        "clientSecret": "top_secret_cs",
        "service": "payment",
    }
    record.__dict__["correlation_id"] = "CORR-4242"

    line = formatter.format(record)
    data = json.loads(line)

    assert data["correlation_id"] == "CORR-4242"
    assert data["apiKey"] == REDACTED_PLACEHOLDER
    assert data["user_config"]["clientSecret"] == REDACTED_PLACEHOLDER
    assert data["user_config"]["service"] == "payment"
    assert "top_secret_cs" not in line


def test_structured_json_formatter_exception_traceback_redacted() -> None:
    formatter = StructuredJsonFormatter()
    try:
        raise ValueError("Connection failed using Bearer eyJhbGciOi_secret_jwt")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="err_logger",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Unhandled stage error",
            args=(),
            exc_info=sys.exc_info(),
        )

    line = formatter.format(record)
    data = json.loads(line)

    assert "exception" in data
    assert data["exception"]["type"] == "ValueError"
    assert "Bearer [REDACTED]" in data["exception"]["message"]
    assert "eyJhbGciOi_secret_jwt" not in data["exception"]["traceback"]


def test_structured_json_formatter_handles_unserializable_extra_safely() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="bad_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    class ExplosiveVal:
        def __str__(self) -> str:
            raise RuntimeError("Str explosion with secret=pwd123")

    record.__dict__["explosive"] = ExplosiveVal()
    line = formatter.format(record)

    # Must sanitize explosive attribute safely without crashing or leaking secrets
    assert "\n" not in line
    data = json.loads(line)
    assert data["message"] == "Test message"
    assert data["explosive"] == "[UNSERIALIZABLE_PAYLOAD_REDACTED]"
    assert "pwd123" not in line


def test_structured_json_formatter_outer_fallback_on_catastrophic_error() -> None:
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="corrupt_logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Normal message",
        args=(),
        exc_info=None,
    )
    # Simulate a catastrophic failure during record inspection (e.g. getMessage raising)
    from unittest.mock import Mock

    record.getMessage = Mock(
        side_effect=RuntimeError("Fatal logger failure with token=sk-supersecret")
    )

    line = formatter.format(record)
    assert "\n" not in line
    data = json.loads(line)
    assert data["message"] == "[LOG_FORMATTING_ERROR_SAFE_FALLBACK]"
    assert "sk-supersecret" not in line


# --- 2. OperationalTracer Stage Span Tests ------------------------------------


def test_stage_span_success_and_timing() -> None:
    tracer = OperationalTracer(incident_id="INC-001", correlation_id="CORR-001")

    with tracer.stage_span("assess_severity", metadata={"mode": "fast"}) as handle:
        handle.metadata["steps"] = 1
        time.sleep(0.01)

    trace = tracer.get_trace()
    assert len(trace.stages) == 1
    stage = trace.stages[0]

    assert stage.stage_name == "assess_severity"
    assert stage.status == StageStatus.COMPLETED
    assert stage.duration_ms > 0.0
    assert stage.start_time <= stage.end_time
    assert stage.metadata["mode"] == "fast"
    assert stage.metadata["steps"] == 1
    assert stage.error_category is None


def test_stage_span_failure_and_exception_propagation() -> None:
    tracer = OperationalTracer()

    with pytest.raises(ZeroDivisionError, match="division by zero"):
        with tracer.stage_span("investigate") as handle:
            _ = 1 / 0

    trace = tracer.get_trace()
    assert len(trace.stages) == 1
    stage = trace.stages[0]

    assert stage.stage_name == "investigate"
    assert stage.status == StageStatus.FAILED
    assert stage.error_category == ErrorCategory.APP_ERROR
    assert "division by zero" in (stage.error_message or "")


def test_stage_span_custom_blocked_or_skipped_status() -> None:
    tracer = OperationalTracer()

    with tracer.stage_span("apply_guardrails") as handle:
        handle.status = StageStatus.BLOCKED
        handle.error_category = ErrorCategory.SAFETY_GATE_BLOCKED
        handle.error_message = "Tier 4 failover blocked by policy."

    trace = tracer.get_trace()
    assert len(trace.stages) == 1
    stage = trace.stages[0]
    assert stage.status == StageStatus.BLOCKED
    assert stage.error_category == ErrorCategory.SAFETY_GATE_BLOCKED


# --- 3. OperationalTracer Tool Span Tests -------------------------------------


def test_tool_span_success_and_argument_sanitization() -> None:
    tracer = OperationalTracer()

    args = {
        "service": "payment",
        "api_key": "sk-secret1234567890",
        "nested": {"token": "secret_tok"},
    }
    with tracer.tool_span("get_mock_metrics", target_service="payment", arguments=args) as handle:
        time.sleep(0.005)

    trace = tracer.get_trace()
    assert len(trace.tools) == 1
    tool = trace.tools[0]

    assert tool.tool_name == "get_mock_metrics"
    assert tool.target_service == "payment"
    assert tool.status == "success"
    assert tool.duration_ms >= 0.0
    # Arguments must be sanitized
    assert tool.sanitized_arguments["service"] == "payment"
    assert tool.sanitized_arguments["api_key"] == REDACTED_PLACEHOLDER
    assert tool.sanitized_arguments["nested"]["token"] == REDACTED_PLACEHOLDER
    assert "secret1234567890" not in str(tool.sanitized_arguments)


def test_tool_span_failure_and_exception_propagation() -> None:
    tracer = OperationalTracer()

    class CustomToolError(Exception):
        pass

    with pytest.raises(CustomToolError, match="Service timeout"):
        with tracer.tool_span("get_mock_pod_status", target_service="payment"):
            raise CustomToolError("Service timeout with auth token=secret_pwd")

    trace = tracer.get_trace()
    assert len(trace.tools) == 1
    tool = trace.tools[0]

    assert tool.status == "failure"
    assert tool.error_category == ErrorCategory.TOOL_ERROR
    assert "secret_pwd" not in (tool.error_message or "")
    assert "token=[REDACTED]" in (tool.error_message or "")


def test_trace_tool_call_decorator() -> None:
    tracer = OperationalTracer()

    def sample_tool(service: str, count: int) -> str:
        return f"OK-{service}-{count}"

    wrapped = tracer.trace_tool_call(sample_tool, tool_name="sample_tool")
    result = wrapped(service="payment", count=5)

    assert result == "OK-payment-5"
    trace = tracer.get_trace()
    assert len(trace.tools) == 1
    assert trace.tools[0].tool_name == "sample_tool"
    assert trace.tools[0].target_service == "payment"
    assert trace.tools[0].status == "success"


# --- 4. Trace Snapshot & Diagnostics Tests -----------------------------------


def test_trace_snapshot_isolation() -> None:
    tracer = OperationalTracer(incident_id="INC-TEST", correlation_id="CORR-TEST")

    with tracer.stage_span("stage_1"):
        pass

    snapshot1 = tracer.get_trace()
    assert len(snapshot1.stages) == 1

    # Adding a second stage after snapshot extraction
    with tracer.stage_span("stage_2"):
        pass

    # snapshot1 must NOT have mutated
    assert len(snapshot1.stages) == 1

    # snapshot2 has both stages
    snapshot2 = tracer.get_trace()
    assert len(snapshot2.stages) == 2


def test_set_final_outcome_and_diagnostic() -> None:
    tracer = OperationalTracer()
    diag = DiagnosticReason(
        decision="resolve",
        primary_reason="Verified recovery after connection pool expansion.",
        trigger_type="verification_recovered",
        details={"service": "payment"},
    )
    tracer.set_final_outcome("resolve", diagnostic=diag)

    trace = tracer.get_trace()
    assert trace.final_outcome == "resolve"
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "resolve"
    assert trace.diagnostic.trigger_type == "verification_recovered"


# --- 5. Logger Isolation & Root Logger Protection Tests -----------------------


def test_tracer_does_not_modify_or_reconfigure_root_logger() -> None:
    root_logger = logging.getLogger()
    initial_handlers = list(root_logger.handlers)
    initial_level = root_logger.level

    # Initialize tracer with logging handler attached to isolated logger
    isolated_logger = logging.getLogger(OBSERVABILITY_LOGGER_NAME)
    capture = LogCaptureHandler()
    capture.setFormatter(StructuredJsonFormatter())
    isolated_logger.addHandler(capture)
    isolated_logger.setLevel(logging.INFO)

    tracer = OperationalTracer(incident_id="INC-ROOT", logger=isolated_logger)

    with tracer.stage_span("assess_severity"):
        pass

    # Verify that log event was emitted to the isolated logger
    assert len(capture.formatted_lines) >= 1
    logged_json = json.loads(capture.formatted_lines[0])
    assert logged_json["incident_id"] == "INC-ROOT"

    # Verify that the ROOT logger was NOT modified or reconfigured
    assert root_logger.handlers == initial_handlers
    assert root_logger.level == initial_level

    # Clean up test handler
    isolated_logger.removeHandler(capture)


# --- 6. Step 2 Audit Regression Tests -----------------------------------------


def test_snapshot_deep_isolation_mutating_nested_dict_does_not_corrupt_tracer() -> None:
    """Audit: Mutating nested dictionaries in a returned snapshot must not mutate tracer state."""
    tracer = OperationalTracer()
    with tracer.stage_span("investigate", metadata={"initial_key": "initial_value"}):
        pass

    with tracer.tool_span(
        "get_mock_metrics", target_service="payment", arguments={"service": "payment"}
    ):
        pass

    # Extract first snapshot
    snapshot1 = tracer.get_trace()
    assert snapshot1.stages[0].metadata["initial_key"] == "initial_value"
    assert snapshot1.tools[0].sanitized_arguments["service"] == "payment"

    # Maliciously mutate dictionaries and lists in snapshot1
    snapshot1.stages[0].metadata["corrupted_key"] = "malicious_mutation"
    snapshot1.stages[0].metadata["initial_key"] = "overwritten"
    snapshot1.tools[0].sanitized_arguments["injected_arg"] = "malicious_arg"
    snapshot1.stages.append(snapshot1.stages[0])

    # Extract second snapshot from tracer
    snapshot2 = tracer.get_trace()

    # Verify that tracer internal state was NOT corrupted
    assert "corrupted_key" not in snapshot2.stages[0].metadata
    assert snapshot2.stages[0].metadata["initial_key"] == "initial_value"
    assert "injected_arg" not in snapshot2.tools[0].sanitized_arguments
    assert len(snapshot2.stages) == 1
    assert len(tracer._stages) == 1


def test_record_blocked_tool_without_execution() -> None:
    """Audit: Blocked tools must be recorded with status 'blocked' and duration 0.0ms without running."""
    tracer = OperationalTracer()

    tool_trace = tracer.record_blocked_tool(
        tool_name="restart_service",
        target_service="auth_service",
        arguments={"service": "auth_service", "auth_token": "sk-secret1234567890"},
        reason="Requires human approval: uninvestigated service",
    )

    assert tool_trace.tool_name == "restart_service"
    assert tool_trace.target_service == "auth_service"
    assert tool_trace.status == "blocked"
    assert tool_trace.duration_ms == 0.0
    assert tool_trace.error_category == ErrorCategory.SAFETY_GATE_BLOCKED
    assert "Requires human approval" in (tool_trace.error_message or "")
    # Arguments must be sanitized
    assert tool_trace.sanitized_arguments["auth_token"] == REDACTED_PLACEHOLDER

    # Present in trace snapshot
    trace = tracer.get_trace()
    assert len(trace.tools) == 1
    assert trace.tools[0].status == "blocked"


def test_duration_and_finalization_behavior() -> None:
    """Audit: Duration uses perf_counter; finalization locks end_time and duration idempotently."""
    tracer = OperationalTracer()

    time.sleep(0.01)
    trace_before = tracer.get_trace()
    assert trace_before.end_time is None
    assert trace_before.final_outcome is None
    assert trace_before.total_duration_ms is not None
    assert trace_before.total_duration_ms > 0.0

    diag = DiagnosticReason(
        decision="resolve",
        primary_reason="Verified recovery",
        trigger_type="verification_recovered",
    )
    tracer.set_final_outcome("resolve", diagnostic=diag)

    trace_after = tracer.get_trace()
    assert trace_after.end_time is not None
    assert trace_after.final_outcome == "resolve"
    locked_duration = trace_after.total_duration_ms
    locked_end_time = trace_after.end_time

    # Sleep again to verify that duration is locked and does not keep growing
    time.sleep(0.02)
    tracer.set_final_outcome("resolve", diagnostic=diag)  # Idempotent re-finalization

    trace_subsequent = tracer.get_trace()
    assert trace_subsequent.end_time == locked_end_time
    assert trace_subsequent.total_duration_ms == locked_duration


def test_tracing_failure_never_suppresses_original_tool_exception() -> None:
    """Audit: Tracing record-keeping failures must NEVER suppress original tool exceptions."""
    tracer = OperationalTracer()

    class FailingCollection:
        def append(self, item: Any) -> None:
            raise RuntimeError("Internal tracer buffer full")

    tracer._tools = FailingCollection()  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Underlying tool failed"):
        with tracer.tool_span("failing_tool"):
            raise ValueError("Underlying tool failed")


def test_tracing_failure_never_suppresses_original_stage_exception() -> None:
    """Audit: Tracing record-keeping failures must NEVER suppress original stage exceptions."""
    tracer = OperationalTracer()

    class FailingCollection:
        def append(self, item: Any) -> None:
            raise RuntimeError("Internal tracer buffer full")

    tracer._stages = FailingCollection()  # type: ignore[assignment]

    with pytest.raises(KeyError, match="Original stage error"):
        with tracer.stage_span("failing_stage"):
            raise KeyError("Original stage error")
