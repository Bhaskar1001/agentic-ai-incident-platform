"""Operational tracer and structured JSON logging for incident observability.

Provides:
- StructuredJsonFormatter: Emits single-line, secret-redacted JSON log lines.
- OperationalTracer: Diagnostic tracer recording stage spans and tool invocations.
- Context managers for capturing durations, statuses, errors, and sanitized parameters.
- Isolated logger that does not reconfigure or replace the application root logger.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Any, Callable, Generator, Literal
from uuid import uuid4

from pydantic import ValidationError

from agentic_ai.observability.events import (
    DiagnosticReason,
    ErrorCategory,
    IncidentTrace,
    StageStatus,
    StageTrace,
    ToolInvocationTrace,
)
from agentic_ai.observability.redaction import (
    UNSAFE_FALLBACK_PLACEHOLDER,
    mask_secrets,
    sanitize_dict,
)

# Standard LogRecord attribute names to ignore when extracting custom extra fields
_STANDARD_LOG_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)

OBSERVABILITY_LOGGER_NAME = "agentic_ai.observability"


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as sanitized single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            # Timestamp in UTC ISO 8601
            dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
            timestamp_str = dt.isoformat()

            # Message with embedded secret masking
            raw_msg = record.getMessage()
            safe_msg = mask_secrets(raw_msg)

            data: dict[str, Any] = {
                "timestamp": timestamp_str,
                "level": record.levelname,
                "logger": record.name,
                "message": safe_msg,
            }

            # Extract and sanitize custom extra attributes
            extra_data: dict[str, Any] = {}
            for key, val in record.__dict__.items():
                if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                    extra_data[key] = val

            if extra_data:
                sanitized_extra = sanitize_dict(extra_data)
                for k, v in sanitized_extra.items():
                    data[k] = v

            # Include sanitized exception information if present
            if record.exc_info:
                exc_type, exc_val, _ = record.exc_info
                exc_type_name = exc_type.__name__ if exc_type else "Exception"
                exc_msg = mask_secrets(str(exc_val)) if exc_val else ""
                data["exception"] = {
                    "type": exc_type_name,
                    "message": exc_msg,
                }
                # Format exception traceback safely with secret masking
                formatted_tb = self.formatException(record.exc_info)
                data["exception"]["traceback"] = mask_secrets(formatted_tb)

            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        except Exception:
            # Strict fail-safe fallback: never emit unredacted data or crash logging
            fallback: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": getattr(record, "levelname", "ERROR"),
                "logger": getattr(record, "name", OBSERVABILITY_LOGGER_NAME),
                "message": "[LOG_FORMATTING_ERROR_SAFE_FALLBACK]",
                "error": UNSAFE_FALLBACK_PLACEHOLDER,
            }
            return json.dumps(fallback, separators=(",", ":"))


class StageSpanHandle:
    """Mutable handle to configure an in-flight stage span before completion."""

    def __init__(self, stage_name: str, metadata: dict[str, Any] | None = None) -> None:
        self.stage_name = stage_name
        self.status: StageStatus = StageStatus.COMPLETED
        self.error_category: ErrorCategory | None = None
        self.error_message: str | None = None
        self.metadata: dict[str, Any] = dict(metadata or {})


class ToolSpanHandle:
    """Mutable handle to configure an in-flight tool invocation trace."""

    def __init__(self, tool_name: str, target_service: str | None = None) -> None:
        self.tool_name = tool_name
        self.target_service = target_service
        self.status: Literal["success", "failure", "blocked"] = "success"
        self.error_category: ErrorCategory | None = None
        self.error_message: str | None = None


class OperationalTracer:
    """Thread-safe operational tracer capturing workflow stages and tool calls."""

    def __init__(
        self,
        incident_id: str | None = None,
        correlation_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.incident_id: str = incident_id or f"INC-{uuid4().hex[:8]}"
        self.correlation_id: str = correlation_id or f"CORR-{uuid4().hex[:8]}"
        self._start_time: datetime = datetime.now(timezone.utc)
        self._start_perf: float = time.perf_counter()
        self._end_time: datetime | None = None
        self._end_perf: float | None = None
        self._final_outcome: Literal["resolve", "escalate", "monitor"] | None = None
        self._diagnostic: DiagnosticReason | None = None

        self._stages: list[StageTrace] = []
        self._tools: list[ToolInvocationTrace] = []
        self._lock = threading.Lock()

        # Dedicated, isolated logger; never touches the root logger
        self.logger = logger or logging.getLogger(OBSERVABILITY_LOGGER_NAME)

    def _log_event(self, level: int, message: str, **extra: Any) -> None:
        """Emit an operational log event tagged with incident and correlation IDs."""
        try:
            extra_payload = {
                "incident_id": self.incident_id,
                "correlation_id": self.correlation_id,
                **extra,
            }
            self.logger.log(level, message, extra=extra_payload)
        except Exception:
            # Observability logging failures must never raise or alter workflow execution
            pass

    @contextmanager
    def stage_span(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[StageSpanHandle, None, None]:
        """Context manager for tracing a workflow stage execution.

        Records start/end UTC timestamps, duration_ms via perf_counter, and status.
        Re-raises any caught exception to preserve production error handling.
        """
        handle = StageSpanHandle(stage_name=stage_name, metadata=metadata)
        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        self._log_event(
            logging.INFO,
            f"Stage '{stage_name}' started",
            stage=stage_name,
            stage_status=StageStatus.STARTED.value,
        )

        try:
            yield handle
        except Exception as exc:
            try:
                elapsed_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)
                end_time = max(start_time, datetime.now(timezone.utc))

                handle.status = StageStatus.FAILED
                if handle.error_category is None:
                    if isinstance(exc, ValidationError):
                        handle.error_category = ErrorCategory.VALIDATION_ERROR
                    else:
                        handle.error_category = ErrorCategory.APP_ERROR

                handle.error_message = mask_secrets(str(exc))

                stage_trace = StageTrace(
                    stage_name=stage_name,
                    status=handle.status,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=round(elapsed_ms, 2),
                    error_category=handle.error_category,
                    error_message=handle.error_message,
                    metadata=sanitize_dict(handle.metadata),
                )
                with self._lock:
                    self._stages.append(stage_trace)

                self._log_event(
                    logging.ERROR,
                    f"Stage '{stage_name}' failed: {handle.error_message}",
                    stage=stage_name,
                    stage_status=handle.status.value,
                    duration_ms=round(elapsed_ms, 2),
                    error_category=handle.error_category.value,
                )
            except Exception:
                # Tracing failures must NEVER suppress the original application exception
                pass
            raise

        # Completed normally
        elapsed_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)
        end_time = max(start_time, datetime.now(timezone.utc))

        stage_trace = StageTrace(
            stage_name=stage_name,
            status=handle.status,
            start_time=start_time,
            end_time=end_time,
            duration_ms=round(elapsed_ms, 2),
            error_category=handle.error_category,
            error_message=handle.error_message,
            metadata=sanitize_dict(handle.metadata),
        )
        with self._lock:
            self._stages.append(stage_trace)

        self._log_event(
            logging.INFO,
            f"Stage '{stage_name}' {handle.status.value}",
            stage=stage_name,
            stage_status=handle.status.value,
            duration_ms=round(elapsed_ms, 2),
        )

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        target_service: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> Generator[ToolSpanHandle, None, None]:
        """Context manager for tracing a tool invocation.

        Captures execution duration, status ('success', 'failure', 'blocked'),
        sanitized arguments, and error categorization.
        Re-raises any caught exception to preserve production error handling.
        """
        handle = ToolSpanHandle(tool_name=tool_name, target_service=target_service)
        sanitized_args = sanitize_dict(arguments or {})
        start_perf = time.perf_counter()

        self._log_event(
            logging.DEBUG,
            f"Tool '{tool_name}' invoked",
            tool_name=tool_name,
            target_service=target_service,
            arguments=sanitized_args,
        )

        try:
            yield handle
        except Exception as exc:
            try:
                elapsed_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)

                handle.status = "failure"
                if handle.error_category is None:
                    # Classify tool-specific error vs application error
                    exc_type_name = type(exc).__name__.lower()
                    if "tool" in exc_type_name:
                        handle.error_category = ErrorCategory.TOOL_ERROR
                    elif isinstance(exc, ValidationError):
                        handle.error_category = ErrorCategory.VALIDATION_ERROR
                    elif isinstance(exc, TimeoutError):
                        handle.error_category = ErrorCategory.TIMEOUT
                    else:
                        handle.error_category = ErrorCategory.APP_ERROR

                handle.error_message = mask_secrets(str(exc))

                tool_trace = ToolInvocationTrace(
                    tool_name=tool_name,
                    target_service=target_service,
                    duration_ms=round(elapsed_ms, 2),
                    status=handle.status,
                    error_category=handle.error_category,
                    error_message=handle.error_message,
                    sanitized_arguments=sanitized_args,
                )
                with self._lock:
                    self._tools.append(tool_trace)

                self._log_event(
                    logging.WARNING,
                    f"Tool '{tool_name}' failed: {handle.error_message}",
                    tool_name=tool_name,
                    target_service=target_service,
                    duration_ms=round(elapsed_ms, 2),
                    status=handle.status,
                    error_category=handle.error_category.value,
                )
            except Exception:
                # Tracing failures must NEVER suppress the original application exception
                pass
            raise

        # Completed normally
        elapsed_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)

        tool_trace = ToolInvocationTrace(
            tool_name=tool_name,
            target_service=target_service,
            duration_ms=round(elapsed_ms, 2),
            status=handle.status,
            error_category=handle.error_category,
            error_message=handle.error_message,
            sanitized_arguments=sanitized_args,
        )
        with self._lock:
            self._tools.append(tool_trace)

        self._log_event(
            logging.DEBUG,
            f"Tool '{tool_name}' completed with status '{handle.status}'",
            tool_name=tool_name,
            target_service=target_service,
            duration_ms=round(elapsed_ms, 2),
            status=handle.status,
        )

    def record_blocked_tool(
        self,
        tool_name: str,
        target_service: str | None = None,
        arguments: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> ToolInvocationTrace:
        """Record a tool invocation that was blocked before execution (e.g. by safety guardrails).

        Does not execute the underlying tool. Duration is recorded as 0.0ms.
        """
        sanitized_args = sanitize_dict(arguments or {})
        msg = mask_secrets(reason) if reason else "Tool execution blocked by safety policy"

        tool_trace = ToolInvocationTrace(
            tool_name=tool_name,
            target_service=target_service,
            duration_ms=0.0,
            status="blocked",
            error_category=ErrorCategory.SAFETY_GATE_BLOCKED,
            error_message=msg,
            sanitized_arguments=sanitized_args,
        )
        with self._lock:
            self._tools.append(tool_trace)

        self._log_event(
            logging.WARNING,
            f"Tool '{tool_name}' blocked: {msg}",
            tool_name=tool_name,
            target_service=target_service,
            duration_ms=0.0,
            status="blocked",
            error_category=ErrorCategory.SAFETY_GATE_BLOCKED.value,
        )
        return tool_trace

    def trace_tool_call(
        self,
        func: Callable[..., Any],
        tool_name: str | None = None,
        target_service_extractor: Callable[..., str | None] | None = None,
    ) -> Callable[..., Any]:
        """Wrap a function with tool_span observability, preserving its return value."""
        name = tool_name or getattr(func, "__name__", "unknown_tool")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_service = (
                target_service_extractor(*args, **kwargs)
                if target_service_extractor is not None
                else kwargs.get("service")
            )
            with self.tool_span(tool_name=name, target_service=target_service, arguments=kwargs):
                return func(*args, **kwargs)

        return wrapper

    def set_final_outcome(
        self,
        outcome: Literal["resolve", "escalate", "monitor"],
        diagnostic: DiagnosticReason | None = None,
    ) -> None:
        """Record the final workflow resolution and optional diagnostic reason.

        Idempotent on end_time and end_perf; locks the final duration timestamp.
        """
        with self._lock:
            if self._final_outcome is None:
                self._final_outcome = outcome
                self._diagnostic = diagnostic
            if self._end_time is None:
                self._end_time = max(self._start_time, datetime.now(timezone.utc))
                self._end_perf = time.perf_counter()

        self._log_event(
            logging.INFO,
            f"Incident outcome: {outcome}",
            final_outcome=outcome,
            diagnostic=diagnostic.model_dump() if diagnostic else None,
        )

    def get_trace(self) -> IncidentTrace:
        """Return an immutable snapshot of the current operational trace.

        Does not expose mutable internal collections; subsequent tracing or mutations
        to returned dictionaries cannot mutate the tracer's internal state.
        """
        with self._lock:
            if self._end_perf is not None:
                total_duration = round((self._end_perf - self._start_perf) * 1000.0, 2)
            else:
                total_duration = round((time.perf_counter() - self._start_perf) * 1000.0, 2)

            return IncidentTrace(
                incident_id=self.incident_id,
                correlation_id=self.correlation_id,
                start_time=self._start_time,
                end_time=self._end_time,
                total_duration_ms=total_duration,
                final_outcome=self._final_outcome,
                diagnostic=copy.deepcopy(self._diagnostic),
                stages=[copy.deepcopy(s) for s in self._stages],
                tools=[copy.deepcopy(t) for t in self._tools],
            )


__all__ = [
    "OBSERVABILITY_LOGGER_NAME",
    "OperationalTracer",
    "StageSpanHandle",
    "StructuredJsonFormatter",
    "ToolSpanHandle",
]

