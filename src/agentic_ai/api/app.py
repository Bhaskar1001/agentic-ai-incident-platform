"""FastAPI application factory and route definitions for agentic-ai.

Exposes:
- GET /health: Liveness probe without LLM or external calls.
- GET /ready: Readiness check reflecting configuration status without exposing credentials.
- POST /incidents: Ingests an incident, triggers handle_incident(), saves the run, and returns
  the structured outcome.
- GET /incidents/{incident_id}: Returns a previously saved incident record.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agentic_ai.api.schemas import (
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
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.llm.client import LLMConfigurationError, LLMRateLimitError
from agentic_ai.persistence.factory import build_repository
from agentic_ai.persistence.repository import IncidentConflictError, IncidentRepository
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.workflows.incident import handle_incident

logger = logging.getLogger("agentic_ai.api")


def create_app(
    settings: Settings | None = None,
    repository: IncidentRepository | None = None,
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

    def _repository() -> IncidentRepository:
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

        try:
            state = handle_incident(
                incident_description=req.incident_description,
                scenario=scenario,
                post_remediation_scenario=post_scenario,
                incident_id=req.incident_id,
                correlation_id=req.correlation_id,
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
        persisted = False
        try:
            _repository().save_incident(IncidentRecord.from_workflow_state(state))
            persisted = True
        except IncidentConflictError as exc:
            logger.error("Incident record conflicts with the stored one: %s", exc)
        except Exception as exc:
            logger.error("Failed saving incident record: %s", type(exc).__name__)

        return IncidentResponse(
            incident_id=state["incident_id"],
            correlation_id=state["correlation_id"],
            outcome=state["next_step"],
            severity=severity_val,
            diagnostic=diag_summary,
            verification=verif_summary,
            trace=state.get("trace"),
            persisted=persisted,
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

    return app


app = create_app()
