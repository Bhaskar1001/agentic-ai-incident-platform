"""Domain models for operational incident persistence (Phase 16).

Defines strongly typed, immutable schemas for persisting operational incident
runs, lifecycle state transitions, stage execution spans, and remediation steps.

Operational persistence records are strictly decoupled from learned incident
memory (JSONL) and represent the authoritative audit trail for incident
execution and retrieval.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_ai.api.schemas import DiagnosticSummary, VerificationSummary
from agentic_ai.domain.incident import Severity
from agentic_ai.observability.redaction import sanitize_dict


class IncidentLifecycleStatus(str, Enum):
    """Operational lifecycle progression of an incident execution."""

    DETECTED = "detected"
    """Incident alert received; triage or investigation pending."""

    INVESTIGATING = "investigating"
    """Investigation agent actively exploring evidence via observability tools."""

    MITIGATION = "mitigation"
    """Root cause established; remediation plan being guarded or executed."""

    RESOLVED = "resolved"
    """Remediation executed and service recovery verified."""

    ESCALATED = "escalated"
    """Safety gate blocked, verification inconclusive/unresolved, or high severity handoff to human."""

    MONITORED = "monitored"
    """Investigation confirmed healthy telemetry baseline; low severity under observation."""

    CLOSED = "closed"
    """Terminal state: incident archived after resolution or administrative closure."""

    @property
    def is_terminal(self) -> bool:
        """Return True if this status represents an unalterable terminal lifecycle state."""
        return self in (
            IncidentLifecycleStatus.RESOLVED,
            IncidentLifecycleStatus.ESCALATED,
            IncidentLifecycleStatus.MONITORED,
            IncidentLifecycleStatus.CLOSED,
        )


class IncidentStageRecord(BaseModel):
    """Auditable operational record of a single workflow stage span."""

    stage_name: str
    status: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    error_category: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @field_validator("metadata", mode="before")
    @classmethod
    def _sanitize_metadata(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return sanitize_dict(value)
        return {}


class IncidentRemediationRecord(BaseModel):
    """Auditable operational record of a remediation plan step and its execution."""

    step_order: int
    action_type: str
    target_service: str
    disposition: str
    execution_status: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("parameters", mode="before")
    @classmethod
    def _sanitize_parameters(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return sanitize_dict(value)
        return {}


class IncidentRecord(BaseModel):
    """Top-level operational incident record for relational persistence."""

    incident_id: str
    correlation_id: str
    scenario: str
    incident_description: str
    severity: Severity
    status: IncidentLifecycleStatus
    final_outcome: Literal["resolve", "monitor", "escalate"] | None = None
    diagnostic: DiagnosticSummary | None = None
    verification: VerificationSummary | None = None
    stages: list[IncidentStageRecord] = Field(default_factory=list)
    remediations: list[IncidentRemediationRecord] = Field(default_factory=list)
    total_duration_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_workflow_state(
        cls,
        state: Mapping[str, Any],
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> IncidentRecord:
        """Map LangGraph IncidentWorkflowState to an immutable IncidentRecord.

        Extracts identity, telemetry scenario, severity, diagnostic summaries,
        verification outcomes, stage spans, and remediation audits while
        guaranteeing credential redaction.
        """
        now = datetime.now(timezone.utc)
        effective_created_at = created_at or now
        effective_updated_at = updated_at or now

        incident_id = str(state.get("incident_id") or f"INC-{uuid4().hex[:8]}")
        correlation_id = str(state.get("correlation_id") or f"CORR-{uuid4().hex[:8]}")
        scenario = str(state.get("scenario") or "connection_pool_exhaustion")
        description = str(state.get("incident_description") or "")

        # Extract severity
        severity = Severity.LOW
        sev_assessment = state.get("severity_assessment")
        if sev_assessment is not None:
            if hasattr(sev_assessment, "severity"):
                severity = Severity(str(sev_assessment.severity.value))
            elif isinstance(sev_assessment, dict) and "severity" in sev_assessment:
                severity = Severity(str(sev_assessment["severity"]))

        # Extract terminal outcome
        outcome = state.get("next_step")
        final_outcome: Literal["resolve", "monitor", "escalate"] | None = (
            outcome if outcome in ("resolve", "monitor", "escalate") else None
        )

        # Derive lifecycle status
        if final_outcome == "resolve":
            status = IncidentLifecycleStatus.RESOLVED
        elif final_outcome == "escalate":
            status = IncidentLifecycleStatus.ESCALATED
        elif final_outcome == "monitor":
            status = IncidentLifecycleStatus.MONITORED
        elif (
            state.get("remediation_execution") is not None
            or state.get("remediation_plan") is not None
        ):
            status = IncidentLifecycleStatus.MITIGATION
        elif state.get("investigation") is not None:
            status = IncidentLifecycleStatus.INVESTIGATING
        else:
            status = IncidentLifecycleStatus.DETECTED

        # Determine resolved_at timestamp
        resolved_at: datetime | None = None
        if status.is_terminal:
            resolved_at = now

        # Extract diagnostic summary
        diagnostic: DiagnosticSummary | None = None
        trace = state.get("trace")
        if trace is not None and getattr(trace, "diagnostic", None) is not None:
            diag_obj = trace.diagnostic
            diagnostic = DiagnosticSummary(
                decision=str(getattr(diag_obj, "decision", "")),
                trigger_type=str(getattr(diag_obj, "trigger_type", "")),
                primary_reason=str(getattr(diag_obj, "primary_reason", "")),
            )

        # Extract verification summary
        verification: VerificationSummary | None = None
        ver_obj = state.get("verification")
        if ver_obj is not None:
            ver_status = str(getattr(ver_obj, "status", ""))
            if hasattr(ver_obj.status, "value"):
                ver_status = ver_obj.status.value
            ver_attempts = int(getattr(ver_obj, "attempts", 0))
            ver_reasons = list(getattr(ver_obj, "reasons", []))
            verification = VerificationSummary(
                status=ver_status,
                attempts=ver_attempts,
                reasons=ver_reasons,
            )

        # Extract stage spans
        stages: list[IncidentStageRecord] = []
        if trace is not None and getattr(trace, "stages", None):
            for st in trace.stages:
                st_status = st.status.value if hasattr(st.status, "value") else str(st.status)
                st_err_cat = (
                    st.error_category.value
                    if st.error_category and hasattr(st.error_category, "value")
                    else (str(st.error_category) if st.error_category else None)
                )
                stages.append(
                    IncidentStageRecord(
                        stage_name=st.stage_name,
                        status=st_status,
                        start_time=st.start_time,
                        end_time=st.end_time,
                        duration_ms=st.duration_ms,
                        error_category=st_err_cat,
                        error_message=st.error_message,
                        metadata=st.metadata if hasattr(st, "metadata") else {},
                    )
                )

        # Extract remediations
        remediations: list[IncidentRemediationRecord] = []
        guarded_plan = state.get("guarded_plan")
        rem_exec = state.get("remediation_execution")

        exec_results_by_order: dict[int, Any] = {}
        if rem_exec is not None and getattr(rem_exec, "step_results", None):
            for res in rem_exec.step_results:
                exec_results_by_order[res.step.step_order] = res

        if guarded_plan is not None and getattr(guarded_plan, "steps", None):
            for g_step in guarded_plan.steps:
                step_order = g_step.step.step_order
                action_type = (
                    g_step.step.action_type.value
                    if hasattr(g_step.step.action_type, "value")
                    else str(g_step.step.action_type)
                )
                target_service = g_step.step.target_service
                disposition = (
                    g_step.disposition.value
                    if hasattr(g_step.disposition, "value")
                    else str(g_step.disposition)
                )
                raw_params = (
                    g_step.step.parameters
                    if hasattr(g_step.step, "parameters")
                    and isinstance(g_step.step.parameters, dict)
                    else {}
                )

                exec_res = exec_results_by_order.get(step_order)
                if exec_res is not None:
                    exec_status = (
                        exec_res.status.value
                        if hasattr(exec_res.status, "value")
                        else str(exec_res.status)
                    )
                    output = exec_res.output
                else:
                    exec_status = "skipped"
                    output = None

                remediations.append(
                    IncidentRemediationRecord(
                        step_order=step_order,
                        action_type=action_type,
                        target_service=target_service,
                        disposition=disposition,
                        execution_status=exec_status,
                        parameters=raw_params,
                        output=output,
                    )
                )

        total_duration_ms: float | None = None
        if trace is not None and getattr(trace, "total_duration_ms", None) is not None:
            total_duration_ms = float(trace.total_duration_ms)

        return cls(
            incident_id=incident_id,
            correlation_id=correlation_id,
            scenario=scenario,
            incident_description=description,
            severity=severity,
            status=status,
            final_outcome=final_outcome,
            diagnostic=diagnostic,
            verification=verification,
            stages=stages,
            remediations=remediations,
            total_duration_ms=total_duration_ms,
            created_at=effective_created_at,
            updated_at=effective_updated_at,
            resolved_at=resolved_at,
        )


__all__ = [
    "IncidentLifecycleStatus",
    "IncidentRecord",
    "IncidentRemediationRecord",
    "IncidentStageRecord",
]
