"""FastAPI application factory and route definitions for agentic-ai.

Exposes:
- GET /health: Liveness probe without LLM or external calls.
- GET /ready: Readiness check reflecting configuration status without exposing credentials.
- POST /incidents: Ingests an incident, triggers handle_incident(), saves the run, and returns
  the structured outcome.
- GET /incidents/{incident_id}: Returns a previously saved incident record.
- GET /incidents/{incident_id}/approvals: Lists an incident's approvals (optionally by status).
- POST /incidents/{incident_id}/approvals/{approval_id}/approve and /reject: Record a human
  decision. A decision never executes anything; resubmit the incident to act on it.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
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
    ReadyResponse,
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
    IncidentStore,
)
from agentic_ai.persistence.runs import RunPersistence, persist_workflow_run
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.workflows.incident import handle_incident

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

    return app


app = create_app()
