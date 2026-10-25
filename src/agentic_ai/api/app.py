"""FastAPI application factory and route definitions for agentic-ai.

Exposes:
- GET /health: Liveness probe without LLM or external calls.
- GET /ready: Readiness check reflecting configuration status without exposing credentials.
- POST /incidents: Ingests an incident, triggers handle_incident(), saves the run, and returns
  the structured outcome.
- GET /incidents/{incident_id}: Returns a previously saved incident record.
- GET /incidents/{incident_id}/approvals: Lists an incident's approvals (optionally by status).
- POST /incidents/{incident_id}/approvals/{approval_id}/approve and /reject: Record a human
  decision. A decision never executes anything.
- POST /incidents/{incident_id}/resume: Carry the stored, approved plan forward without
  re-planning (no LLM). Executes only eligible steps, through the execution ledger.
- GET /approvals/pending: Pending approvals across incidents, paginated and filterable.
- GET /incidents/{incident_id}/executions: The incident's step execution ledger.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agentic_ai.api.schemas import (
    ApprovalDecisionRequest,
    ApprovalListResponse,
    ApprovalResponse,
    DiagnosticSummary,
    ErrorResponse,
    HealthResponse,
    IncidentRequest,
    IncidentResponse,
    PendingApprovalsResponse,
    PendingApprovalSummary,
    ReadyResponse,
    ResumeRequest,
    ResumeResponse,
    ResumeStepResponse,
    StepExecutionListResponse,
    StepExecutionResponse,
    VerificationSummary,
)
from agentic_ai.config import (
    ConfigurationError,
    LLMProviderType,
    Settings,
    get_settings,
)
from agentic_ai.domain.approvals import (
    ApprovalDecisionError,
    ApprovalRecord,
    check_decidable,
    plan_fingerprint,
)
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.llm.client import LLMConfigurationError, LLMRateLimitError
from agentic_ai.persistence.factory import build_repository
from agentic_ai.persistence.repository import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    IncidentNotFoundError,
    IncidentPersistenceError,
    IncidentStore,
)
from agentic_ai.persistence.runs import RunPersistence, persist_workflow_run
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.workflows.incident import handle_incident
from agentic_ai.workflows.resume import ResumeRefusedError, resume_incident

logger = logging.getLogger("agentic_ai.api")


def create_app(
    settings: Settings | None = None,
    repository: IncidentStore | None = None,
) -> FastAPI:
    """Create and configure the FastAPI service instance.

    ``repository`` is injected for tests; otherwise it is built lazily from the
    settings on first use (PostgreSQL when ``DATABASE_URL`` is set).
    """
    app = FastAPI(
        title="AI Engineering & Incident Resolution Platform API",
        description="Autonomous, safety-bounded incident investigation and resolution service.",
        version="0.1.0",
    )

    # Attach settings or dynamic resolver
    app.state.settings = settings

    app.state.repository = repository

    def _current_settings() -> Settings:
        if app.state.settings is not None:
            return app.state.settings
        return get_settings()

    def _repository() -> IncidentStore:
        if app.state.repository is None:
            app.state.repository = build_repository(_current_settings())
        return app.state.repository

    @app.exception_handler(RequestValidationError)
    def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Sanitize validation errors: omit raw input values to prevent credential leakage
        sanitized_errors = []
        for err in exc.errors():
            sanitized_errors.append(
                {
                    "loc": err.get("loc"),
                    "msg": err.get("msg"),
                    "type": err.get("type"),
                }
            )
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation error", "errors": sanitized_errors},
        )

    @app.exception_handler(LLMRateLimitError)
    def _rate_limit_handler(request: Request, exc: LLMRateLimitError) -> JSONResponse:
        logger.warning("LLM rate limit reached: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail="LLM provider rate limit exceeded. Please retry later."
            ).model_dump(),
        )

    @app.exception_handler(LLMConfigurationError)
    @app.exception_handler(ConfigurationError)
    def _configuration_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Configuration error: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(detail="Service provider misconfiguration.").model_dump(),
        )

    @app.exception_handler(Exception)
    def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak raw stack trace, internal paths, or unmasked exception strings
        logger.exception("Unhandled error processing request: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal error occurred while processing the incident."
            ).model_dump(),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Liveness Probe",
        tags=["System"],
    )
    def health() -> HealthResponse:
        """Lightweight liveness check. Does not invoke LLMs or external networks."""
        return HealthResponse(status="ok", service="agentic-ai", version="0.1.0")

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        responses={
            503: {"model": ReadyResponse, "description": "Service not ready"},
        },
        summary="Service Readiness Probe",
        tags=["System"],
    )
    def ready(response: Response) -> Any:
        """Report configuration readiness for the active provider mode.

        Distinguishes configuration readiness from external provider reachability.
        Never exposes raw secrets or API keys.
        """
        cfg = _current_settings()
        provider = cfg.llm_provider

        if provider == LLMProviderType.OLLAMA:
            return ReadyResponse(
                status="ready",
                llm_provider=provider.value,
                model=cfg.ollama_model,
                configured=True,
                connectivity_verified=False,
                details="Ollama configured for local mode (connectivity not verified).",
            )

        if provider == LLMProviderType.OPENROUTER:
            has_key = cfg.openrouter_api_key is not None
            if has_key:
                return ReadyResponse(
                    status="ready",
                    llm_provider=provider.value,
                    model=cfg.openrouter_model,
                    configured=True,
                    connectivity_verified=False,
                    details="OpenRouter configured with credentials (connectivity not verified).",
                )
            # Not ready for live OpenRouter calls
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(
                status="not_ready",
                llm_provider=provider.value,
                model=cfg.openrouter_model,
                configured=False,
                connectivity_verified=False,
                details="OpenRouter API key is not configured.",
            )

        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="not_ready",
            llm_provider=str(provider),
            model="unknown",
            configured=False,
            connectivity_verified=False,
            details=f"Unknown provider: {provider}",
        )

    @app.post(
        "/incidents",
        response_model=IncidentResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Validation Error"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            503: {"model": ErrorResponse, "description": "Service Unavailable"},
        },
        summary="Ingest and Resolve Incident",
        tags=["Incidents"],
    )
    def submit_incident(req: IncidentRequest) -> IncidentResponse:
        """Trigger the incident resolution workflow.

        Validates inputs, invokes handle_incident(), and returns structured diagnostics,
        verification status, and sanitized trace.
        """
        scenario = req.scenario or DEFAULT_SCENARIO
        post_scenario = req.post_remediation_scenario or Scenario.HEALTHY

        # The workflow reads stored approvals at its execution boundary, so it needs the
        # store. If the store cannot even be built, run without it: steps awaiting a
        # human then simply stay unexecuted, and the response says nothing was persisted.
        try:
            store: IncidentStore | None = _repository()
        except Exception as exc:
            logger.error("Incident storage could not be initialised: %s", type(exc).__name__)
            store = None

        try:
            state = handle_incident(
                incident_description=req.incident_description,
                scenario=scenario,
                post_remediation_scenario=post_scenario,
                incident_id=req.incident_id,
                correlation_id=req.correlation_id,
                approval_repository=store,
                execution_ledger=store,
            )
        except (LLMRateLimitError, LLMConfigurationError, ConfigurationError):
            raise
        except Exception as exc:
            logger.error("Failed executing incident workflow: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Incident resolution workflow encountered an internal error.",
            ) from exc

        # Extract diagnostic summary
        diag_summary: DiagnosticSummary | None = None
        if state.get("trace") and state["trace"].diagnostic:
            diag = state["trace"].diagnostic
            diag_summary = DiagnosticSummary(
                decision=diag.decision,
                trigger_type=diag.trigger_type,
                primary_reason=diag.primary_reason,
            )

        # Extract verification summary
        verif_summary: VerificationSummary | None = None
        if state.get("verification"):
            v = state["verification"]
            verif_summary = VerificationSummary(
                status=v.status.value,
                attempts=v.attempts,
                reasons=list(v.reasons),
            )

        severity_val: str | None = None
        if state.get("severity_assessment"):
            severity_val = state["severity_assessment"].severity.value

        # The run has already happened (and any simulated remediation with it), so a
        # failure to save must not hide the result. Report it instead.
        run = (
            persist_workflow_run(store, state)
            if store is not None
            else RunPersistence(persisted=False, error="incident storage unavailable")
        )

        return IncidentResponse(
            incident_id=state["incident_id"],
            correlation_id=state["correlation_id"],
            outcome=state["next_step"],
            severity=severity_val,
            diagnostic=diag_summary,
            verification=verif_summary,
            trace=state.get("trace"),
            persisted=run.persisted,
            pending_approval_ids=run.pending_approval_ids,
        )

    @app.get(
        "/incidents/{incident_id}",
        response_model=IncidentRecord,
        responses={404: {"model": ErrorResponse, "description": "Incident not found"}},
        summary="Fetch a Saved Incident",
        tags=["Incidents"],
    )
    def get_incident(incident_id: str) -> IncidentRecord:
        """Return the stored audit record for one incident."""
        try:
            record = _repository().get_incident(incident_id)
        except Exception as exc:
            logger.error("Failed reading incident record: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Incident storage is unavailable.",
            ) from exc
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident '{incident_id}' not found.",
            )
        return record

    # -- approvals ---------------------------------------------------------------

    def _approval_view(approval: ApprovalRecord, record: IncidentRecord | None) -> ApprovalResponse:
        stale: bool | None = None
        decidable: bool | None = None
        if record is not None:
            ordered = sorted(record.remediations, key=lambda r: r.step_order)
            current = plan_fingerprint((r.action_type, r.target_service) for r in ordered)
            stale = current != approval.plan_fingerprint
            try:
                check_decidable(approval, record.status, record.remediations)
                decidable = True
            except ApprovalDecisionError:
                decidable = False
        return ApprovalResponse(
            approval_id=approval.approval_id,
            incident_id=approval.incident_id,
            step_index=approval.step_index,
            action_type=approval.action_type,
            target_service=approval.target_service,
            rationale=approval.rationale,
            status=approval.status.value,
            requested_at=approval.requested_at,
            decided_at=approval.decided_at,
            approver=approval.approver,
            decision_reason=approval.decision_reason,
            stale=stale,
            decidable=decidable,
        )

    def _storage_unavailable(exc: Exception, action: str) -> HTTPException:
        logger.error("Failed %s: %s", action, type(exc).__name__)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Incident storage is unavailable.",
        )

    @app.get(
        "/incidents/{incident_id}/approvals",
        response_model=ApprovalListResponse,
        responses={404: {"model": ErrorResponse, "description": "Incident not found"}},
        summary="List an Incident's Approvals",
        tags=["Approvals"],
    )
    def list_approvals(
        incident_id: str,
        status_filter: Literal["pending", "approved", "rejected"] | None = Query(
            default=None,
            alias="status",
            description="Only approvals in this status.",
        ),
    ) -> ApprovalListResponse:
        """Approvals for one incident, oldest request first.

        ``stale`` marks an approval whose plan was replaced; ``decidable`` says whether
        a decision would be accepted right now.
        """
        try:
            store = _repository()
            record = store.get_incident(incident_id)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Incident '{incident_id}' not found.",
                )
            approvals = store.list_approvals(
                incident_id, ApprovalState(status_filter) if status_filter else None
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _storage_unavailable(exc, "listing approvals") from exc
        return ApprovalListResponse(
            incident_id=incident_id,
            approvals=[_approval_view(a, record) for a in approvals],
        )

    def _decide(
        incident_id: str,
        approval_id: str,
        req: ApprovalDecisionRequest,
        outcome: ApprovalState,
    ) -> ApprovalResponse:
        try:
            store = _repository()
            decided = store.decide_approval(
                incident_id, approval_id, outcome, req.approver, req.reason
            )
        except IncidentNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ApprovalConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except Exception as exc:
            raise _storage_unavailable(exc, "recording an approval decision") from exc

        # The decision is committed. A failure to re-read the incident only costs the
        # stale/decidable hints; it must not turn a successful decision into an error.
        record: IncidentRecord | None = None
        try:
            record = store.get_incident(incident_id)
        except Exception as exc:
            logger.error("Failed re-reading incident after a decision: %s", type(exc).__name__)
        return _approval_view(decided, record)

    _DECISION_NOTE = (
        " Recording a decision does not execute anything and does not resolve the incident: "
        "resubmit the same incident_id and the workflow will act on it, subject to every "
        "guardrail. The approver name is caller-supplied and NOT authenticated."
    )

    @app.post(
        "/incidents/{incident_id}/approvals/{approval_id}/approve",
        response_model=ApprovalResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Incident or approval not found"},
            409: {
                "model": ErrorResponse,
                "description": "Already decided, incident closed, or plan superseded",
            },
        },
        summary="Approve a Remediation Step",
        description="Approve exactly one pending remediation step." + _DECISION_NOTE,
        tags=["Approvals"],
    )
    def approve_step(
        incident_id: str, approval_id: str, req: ApprovalDecisionRequest
    ) -> ApprovalResponse:
        """Approve exactly one pending remediation step."""
        return _decide(incident_id, approval_id, req, ApprovalState.APPROVED)

    @app.post(
        "/incidents/{incident_id}/approvals/{approval_id}/reject",
        response_model=ApprovalResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Incident or approval not found"},
            409: {
                "model": ErrorResponse,
                "description": "Already decided, incident closed, or plan superseded",
            },
        },
        summary="Reject a Remediation Step",
        description=(
            "Reject exactly one pending remediation step; it will never execute." + _DECISION_NOTE
        ),
        tags=["Approvals"],
    )
    def reject_step(
        incident_id: str, approval_id: str, req: ApprovalDecisionRequest
    ) -> ApprovalResponse:
        """Reject exactly one pending remediation step; it will never execute."""
        return _decide(incident_id, approval_id, req, ApprovalState.REJECTED)

    # -- resume, pending queue, ledger -----------------------------------------------

    _RESUME_NOTE = (
        " Never calls the LLM planner: it takes the plan that was stored when it was presented"
        " for approval, re-runs the deterministic guardrails on the stored inputs, applies the"
        " stored approval decisions, and refuses (409, nothing executed) if any required"
        " approval is pending or rejected, the incident is closed, the stored plan is missing or"
        " fails its integrity check, or a step has an unknown outcome from an earlier attempt."
        " Each step is claimed in the execution ledger before it runs, so a step that already ran"
        " is not run again. This does not guarantee exactly-once execution against an external"
        " system. `requested_by` is a caller-supplied label and is NOT authenticated."
    )

    @app.post(
        "/incidents/{incident_id}/resume",
        response_model=ResumeResponse,
        responses={
            404: {"model": ErrorResponse, "description": "Incident not found"},
            409: {"model": ErrorResponse, "description": "Refused; nothing was executed"},
        },
        summary="Resume an Incident's Stored Plan",
        description="Resume the exact approved plan without re-planning." + _RESUME_NOTE,
        tags=["Approvals"],
    )
    def resume_stored_plan(
        incident_id: str,
        req: ResumeRequest = Body(default_factory=ResumeRequest),
    ) -> ResumeResponse:
        """Resume the stored plan after its approvals have been recorded."""
        post_scenario = req.post_remediation_scenario or Scenario.HEALTHY
        try:
            result = resume_incident(
                _repository(),
                incident_id,
                post_remediation_scenario=post_scenario.value,
                requested_by=req.requested_by,
            )
        except IncidentNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ResumeRefusedError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except IncidentPersistenceError as exc:
            raise _storage_unavailable(exc, "resuming an incident") from exc
        except Exception as exc:
            logger.error("Resume failed unexpectedly: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Resuming the incident encountered an internal error.",
            ) from exc

        return ResumeResponse(
            incident_id=result.incident_id,
            plan_fingerprint=result.plan_fingerprint,
            snapshot_id=result.snapshot_id,
            outcome=result.outcome,
            status=result.status.value,
            steps=[
                ResumeStepResponse(
                    step_index=s.step_index,
                    action_type=s.action_type,
                    target_service=s.target_service,
                    disposition=s.disposition,
                    execution_status=s.execution_status,
                    output=s.output,
                    error=s.error,
                    ran_now=s.ran_now,
                )
                for s in result.steps
            ],
            verification=VerificationSummary(
                status=result.verification.status.value,
                attempts=result.verification.attempts,
                reasons=list(result.verification.reasons),
            ),
            persisted=result.persisted,
            error=result.error,
        )

    @app.get(
        "/approvals/pending",
        response_model=PendingApprovalsResponse,
        summary="List Pending Approvals Across Incidents",
        description=(
            "Oldest request first. Approvals on a closed incident are left out unless"
            " `include_closed=true`, because nobody can decide them. This service has no"
            " authentication: anyone who can reach it can read this queue."
        ),
        tags=["Approvals"],
    )
    def list_pending_approvals(
        limit: int = Query(default=50, ge=1, le=200, description="Page size."),
        offset: int = Query(default=0, ge=0, description="Items to skip."),
        action_type: str | None = Query(default=None, max_length=64),
        target_service: str | None = Query(default=None, max_length=64),
        include_closed: bool = Query(default=False),
    ) -> PendingApprovalsResponse:
        try:
            store = _repository()
            items = store.list_pending_approvals(
                limit,
                offset,
                action_type=action_type,
                target_service=target_service,
                include_closed=include_closed,
            )
            total = store.count_pending_approvals(
                action_type=action_type,
                target_service=target_service,
                include_closed=include_closed,
            )
        except Exception as exc:
            raise _storage_unavailable(exc, "listing pending approvals") from exc
        return PendingApprovalsResponse(
            approvals=[
                PendingApprovalSummary(
                    approval_id=a.approval_id,
                    incident_id=a.incident_id,
                    step_index=a.step_index,
                    action_type=a.action_type,
                    target_service=a.target_service,
                    status=a.status.value,
                    requested_at=a.requested_at,
                )
                for a in items
            ],
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(items) < total,
        )

    @app.get(
        "/incidents/{incident_id}/executions",
        response_model=StepExecutionListResponse,
        responses={404: {"model": ErrorResponse, "description": "Incident not found"}},
        summary="List an Incident's Step Executions",
        description=(
            "The execution ledger: which steps were started, how they ended, and by which"
            " path. `outcome_unknown` marks a step that started and never recorded a result."
        ),
        tags=["Approvals"],
    )
    def list_step_executions(incident_id: str) -> StepExecutionListResponse:
        try:
            store = _repository()
            if store.get_incident(incident_id) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Incident '{incident_id}' not found.",
                )
            records = store.list_step_executions(incident_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise _storage_unavailable(exc, "listing step executions") from exc
        return StepExecutionListResponse(
            incident_id=incident_id,
            executions=[
                StepExecutionResponse(
                    execution_id=r.execution_id,
                    step_index=r.step_index,
                    action_type=r.action_type,
                    target_service=r.target_service,
                    status=r.status.value,
                    source=r.source.value,
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                    outcome_unknown=r.is_unknown,
                )
                for r in records
            ],
        )

    return app


app = create_app()
