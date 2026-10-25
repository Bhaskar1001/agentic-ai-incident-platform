"""Pydantic schemas for the agentic-ai REST API layer.

Defines typed request and response contracts for:
- Liveness (/health)
- Readiness (/ready)
- Incident ingestion and execution (/incidents)
- Structured diagnostic, verification, and error envelopes
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_ai.observability.events import IncidentTrace
from agentic_ai.tools.mock_tools import Scenario
from agentic_ai.domain.summaries import DiagnosticSummary, VerificationSummary


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = "ok"
    service: str = "agentic-ai"
    version: str = "0.1.0"

    model_config = ConfigDict(frozen=True)


class ReadyResponse(BaseModel):
    """Readiness probe response.

    Reflects whether the platform is configured for its selected operating mode.
    Does not expose secrets, credentials, or raw environment variables.
    """

    status: str  # "ready" or "not_ready"
    llm_provider: str
    model: str
    configured: bool
    connectivity_verified: bool = False
    details: str

    model_config = ConfigDict(frozen=True)


class IncidentRequest(BaseModel):
    """Payload for submitting an incident to be resolved by the workflow."""

    incident_description: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Observed incident symptoms or alert text.",
    )
    scenario: Scenario | None = Field(
        default=None,
        description="Optional simulation scenario override for test/mock runs.",
    )
    post_remediation_scenario: Scenario | None = Field(
        default=None,
        description="Optional post-remediation scenario override.",
    )
    incident_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional caller-supplied incident identifier.",
    )
    correlation_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional distributed tracing correlation identifier.",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("incident_description")
    @classmethod
    def _validate_description_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("incident_description cannot be blank or whitespace.")
        return cleaned


class IncidentResponse(BaseModel):
    """Structured response returned upon completing incident triage and resolution."""

    incident_id: str
    correlation_id: str
    outcome: str  # "resolve", "monitor", "escalate"
    severity: str | None = None
    diagnostic: DiagnosticSummary | None = None
    verification: VerificationSummary | None = None
    trace: IncidentTrace | None = None
    persisted: bool = False
    """Whether this run was written to the incident repository. False means the
    result is valid but the audit record was not saved (see the server logs)."""

    model_config = ConfigDict(frozen=True)


class ErrorResponse(BaseModel):
    """Standardized error envelope preventing stack trace or secret exposure."""

    detail: str
    incident_id: str | None = None
    correlation_id: str | None = None

    model_config = ConfigDict(frozen=True)
