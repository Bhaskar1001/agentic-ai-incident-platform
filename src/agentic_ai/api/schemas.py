"""Pydantic schemas for the agentic-ai REST API layer.

Defines typed request and response contracts for:
- Liveness (/health)
- Readiness (/ready)
- Incident ingestion and execution (/incidents)
- Structured diagnostic, verification, and error envelopes
"""

from __future__ import annotations

from datetime import datetime

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
    """Whether this run was fully written: the incident record and, if any step is
    waiting on a human, its pending approvals. False means the result is valid but
    something was not saved (see the server logs); it never means a retry happened."""
    pending_approval_ids: list[str] = Field(default_factory=list)
    """Approvals this run is waiting on. Decide them through the approvals endpoints,
    then resubmit the same incident_id so the workflow can act on the decision."""

    model_config = ConfigDict(frozen=True)


class ErrorResponse(BaseModel):
    """Standardized error envelope preventing stack trace or secret exposure."""

    detail: str
    incident_id: str | None = None
    correlation_id: str | None = None

    model_config = ConfigDict(frozen=True)


class ApprovalDecisionRequest(BaseModel):
    """A human decision on one pending approval.

    ``approver`` is a free-text label supplied by the caller. **It is not an
    authenticated identity**: this service has no authentication, so anyone who can
    reach the endpoint can claim any name. It is recorded for the audit trail as
    provisional, and the response says so (``approver_verified: false``).
    """

    approver: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=1000)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("approver")
    @classmethod
    def _approver_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("approver must not be blank.")
        return cleaned

    @field_validator("reason")
    @classmethod
    def _reason_cleaned(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ApprovalResponse(BaseModel):
    """One approval, with what it authorizes and how it was decided."""

    approval_id: str
    incident_id: str
    step_index: int
    action_type: str
    target_service: str
    rationale: str
    status: str  # "pending", "approved", "rejected"
    requested_at: datetime
    decided_at: datetime | None = None
    approver: str | None = None
    decision_reason: str | None = None
    approver_verified: bool = False
    """Always False: the approver name is caller-supplied and unauthenticated."""
    stale: bool | None = None
    """True if the incident's stored plan no longer matches this approval (the plan
    was replaced), so it can never authorize anything. None if it could not be checked."""
    decidable: bool | None = None
    """True only if a decision would be accepted right now. None if it could not be checked."""

    model_config = ConfigDict(frozen=True)


class ApprovalListResponse(BaseModel):
    incident_id: str
    approvals: list[ApprovalResponse]

    model_config = ConfigDict(frozen=True)


class ResumeRequest(BaseModel):
    """Resume an incident's stored plan. Both fields are optional.

    ``requested_by`` is a free-text label recorded in the audit trail. **It is not an
    authenticated identity** (this service has no authentication).
    """

    post_remediation_scenario: Scenario | None = None
    requested_by: str | None = Field(default=None, max_length=128)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("requested_by")
    @classmethod
    def _requested_by_cleaned(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ResumeStepResponse(BaseModel):
    step_index: int
    action_type: str
    target_service: str
    disposition: str
    execution_status: str
    """executed, skipped, blocked, failed or unknown."""
    output: str
    error: str | None = None
    ran_now: bool
    """True only if this resume executed the step; False if it only reported on it."""

    model_config = ConfigDict(frozen=True)


class ResumeResponse(BaseModel):
    incident_id: str
    plan_fingerprint: str
    snapshot_id: str
    outcome: str  # "resolve" or "escalate"
    status: str
    steps: list[ResumeStepResponse]
    verification: VerificationSummary
    persisted: bool
    """False means the steps ran (they are in the execution ledger) but the incident record
    could not be saved; resuming again completes the save without running anything twice."""
    error: str | None = None
    requested_by_verified: bool = False
    """Always False: ``requested_by`` is caller-supplied and unauthenticated."""

    model_config = ConfigDict(frozen=True)


class PendingApprovalSummary(BaseModel):
    """A pending approval as listed across incidents.

    Deliberately minimal: no rationale, fingerprints or execution detail. Fetch the
    incident's own approvals for more.
    """

    approval_id: str
    incident_id: str
    step_index: int
    action_type: str
    target_service: str
    status: str
    requested_at: datetime

    model_config = ConfigDict(frozen=True)


class PendingApprovalsResponse(BaseModel):
    approvals: list[PendingApprovalSummary]
    total: int
    limit: int
    offset: int
    has_more: bool

    model_config = ConfigDict(frozen=True)


class StepExecutionResponse(BaseModel):
    """One ledger entry. Output and error text are left out on purpose."""

    execution_id: str
    step_index: int
    action_type: str
    target_service: str
    status: str  # started, succeeded, failed
    source: str  # workflow, resume
    started_at: datetime
    finished_at: datetime | None = None
    outcome_unknown: bool
    """True for a step that started and never recorded a result: it may or may not have
    taken effect, and it is never replayed automatically."""

    model_config = ConfigDict(frozen=True)


class StepExecutionListResponse(BaseModel):
    incident_id: str
    executions: list[StepExecutionResponse]

    model_config = ConfigDict(frozen=True)
