"""Observability and operational tracing package for the Incident Resolution Platform.

Provides structured trace models, error categorizations, diagnostic summaries,
deterministic secret sanitization helpers, and an operational tracer.
"""

from agentic_ai.observability.events import (
    DiagnosticReason,
    ErrorCategory,
    IncidentTrace,
    StageStatus,
    StageTrace,
    ToolInvocationTrace,
)
from agentic_ai.observability.redaction import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_STRING_LENGTH,
    MAX_DEPTH_PLACEHOLDER,
    REDACTED_PLACEHOLDER,
    UNSAFE_FALLBACK_PLACEHOLDER,
    mask_secrets,
    sanitize_dict,
    sanitize_payload,
)
from agentic_ai.observability.tracer import (
    OBSERVABILITY_LOGGER_NAME,
    OperationalTracer,
    StageSpanHandle,
    StructuredJsonFormatter,
    ToolSpanHandle,
)

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_STRING_LENGTH",
    "DiagnosticReason",
    "ErrorCategory",
    "IncidentTrace",
    "MAX_DEPTH_PLACEHOLDER",
    "OBSERVABILITY_LOGGER_NAME",
    "OperationalTracer",
    "REDACTED_PLACEHOLDER",
    "StageSpanHandle",
    "StageStatus",
    "StageTrace",
    "StructuredJsonFormatter",
    "ToolInvocationTrace",
    "ToolSpanHandle",
    "UNSAFE_FALLBACK_PLACEHOLDER",
    "mask_secrets",
    "sanitize_dict",
    "sanitize_payload",
]
