"""Domain models for operational tracing and incident observability.

Defines schemas for operational traces, stage-level spans, tool invocations,
error categorization, and diagnostic reasons.

All models are validated and immutable (frozen=True). They provide purely
observational diagnostic records and do not alter workflow decisions, safety
gates, or recovery criteria.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """Categorization of failures encountered during incident workflows."""

    APP_ERROR = "app_error"
    """Unhandled internal exception or framework error."""

    TOOL_ERROR = "tool_error"
    """Mock or remote tool execution error, protocol failure, or unreadable response."""

    TIMEOUT = "timeout"
    """Step limit reached or execution budget exhausted."""

    SAFETY_GATE_BLOCKED = "safety_gate_blocked"
    """Action blocked or halted by safety guardrails or required approval."""

    VALIDATION_ERROR = "validation_error"
    """Schema, parameter, or Pydantic validation failure."""


class StageStatus(str, Enum):
    """Execution status for a workflow stage span."""

    STARTED = "started"
    """Stage execution has begun."""

    COMPLETED = "completed"
    """Stage finished successfully."""

    FAILED = "failed"
    """Stage encountered an unrecoverable failure or returned an error state."""

    SKIPPED = "skipped"
    """Stage was bypassed due to workflow routing or unapproved prerequisites."""

    BLOCKED = "blocked"
    """Stage was halted by a safety guardrail or policy rule."""


class ToolInvocationTrace(BaseModel, frozen=True):
    """Structured record of a single tool invocation during an incident."""

    tool_name: str
    target_service: str | None = None
    duration_ms: float
    status: Literal["success", "failure", "blocked"]
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    sanitized_arguments: dict[str, Any] = Field(default_factory=dict)


class StageTrace(BaseModel, frozen=True):
    """Structured span for a workflow stage."""

    stage_name: str
    status: StageStatus
    start_time: datetime
    end_time: datetime
    duration_ms: float
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticReason(BaseModel, frozen=True):
    """Structured explanation of why the incident workflow reached its final outcome."""

    decision: Literal["resolve", "escalate", "monitor"]
    primary_reason: str
    trigger_type: Literal[
        "severity_baseline",
        "evidence_metrics",
        "investigation_incomplete",
        "guardrail_rejection",
        "execution_failure",
        "verification_unresolved",
        "verification_recovered",
    ]
    details: dict[str, Any] = Field(default_factory=dict)


class IncidentTrace(BaseModel, frozen=True):
    """Top-level operational trace for an incident execution."""

    incident_id: str
    correlation_id: str
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    end_time: datetime | None = None
    total_duration_ms: float | None = None
    final_outcome: Literal["resolve", "escalate", "monitor"] | None = None
    diagnostic: DiagnosticReason | None = None
    stages: list[StageTrace] = Field(default_factory=list)
    tools: list[ToolInvocationTrace] = Field(default_factory=list)


__all__ = [
    "DiagnosticReason",
    "ErrorCategory",
    "IncidentTrace",
    "StageStatus",
    "StageTrace",
    "ToolInvocationTrace",
]

