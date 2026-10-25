"""Integration tests for the FastAPI service layer (Phase 15 Step 2).

Validates:
1. /health returns liveness without credentials or external calls.
2. /ready reports readiness across Ollama and OpenRouter configurations.
3. /ready distinguishes configuration readiness without leaking secrets.
4. /incidents validates requests (422 on invalid payloads).
5. /incidents executes handle_incident() and returns structured responses.
6. Custom and auto-generated correlation and incident identifiers.
7. Redaction and sanitization in API responses (zero secret leakage).
8. Workflow exceptions mapped to appropriate HTTP status codes (500, 503).
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_ai.api import create_app
from agentic_ai.config import LLMProviderType, Settings
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import (
    RemediationActionType,
    RemediationPlan,
    RemediationStep,
)
from agentic_ai.domain.verification import VerificationResult, VerificationStatus
from agentic_ai.llm.client import LLMConfigurationError, LLMRateLimitError
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.tools.mock_tools import Scenario, get_mock_metrics, get_mock_pod_status
from agentic_ai.workflows import incident as workflow_module


def _assessment(severity: Severity) -> SeverityAssessment:
    return SeverityAssessment(severity=severity, reasoning="Test severity assessment")


def _evidence(scenario: Scenario) -> list[Evidence]:
    return [
        Evidence(
            source_tool=name,
            tool_arguments={"service": "payment"},
            raw_output=tool("payment", scenario),
            observation=f"{name} returned data",
        )
        for name, tool in (
            ("get_mock_metrics", get_mock_metrics),
            ("get_mock_pod_status", get_mock_pod_status),
        )
    ]


def _result(
    termination: TerminationReason = TerminationReason.COMPLETED,
    scenario: Scenario = Scenario.HEALTHY,
    findings: list[Finding] | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        summary="Investigation summary.",
        model_assessed_confidence=0.8,
        recommended_next_action="No action required.",
        termination_reason=termination,
        evidence=_evidence(scenario),
        findings=findings or [],
    )


def _finding() -> Finding:
    return Finding(
        statement="Connection pool utilisation is 100%.",
        model_assessed_confidence=0.9,
    )


def _analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Database connection pool exhaustion.",
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )


@pytest.fixture
def stub_dependencies(monkeypatch):
    """Replace the LLM-backed steps with deterministic stubs."""

    def _apply(
        severity: Severity,
        investigation: InvestigationResult | None = None,
        error: Exception | None = None,
        rca_result: RootCauseAnalysis | None = None,
    ):
        monkeypatch.setattr(
            workflow_module,
            "assess_severity",
            lambda description: _assessment(severity),
        )

        def fake_investigate(description, scenario=None):
            if error is not None:
                raise error
            return investigation if investigation is not None else _result()

        monkeypatch.setattr(workflow_module, "investigate", fake_investigate)

        monkeypatch.setattr(
            workflow_module,
            "analyze_root_cause",
            lambda result: rca_result if rca_result is not None else _analysis(),
        )

        monkeypatch.setattr(
            workflow_module,
            "plan_remediation",
            lambda description, analysis: RemediationPlan(
                steps=[
                    RemediationStep(
                        action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                        target_service="payment",
                        rationale="Increase database pool limit to relieve connection pressure",
                    )
                ]
            ),
        )

    return _apply


def test_health_endpoint() -> None:
    """GET /health returns liveness status without credentials or external calls."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "agentic-ai"
    assert data["version"] == "0.1.0"


def test_ready_endpoint_ollama_mode() -> None:
    """GET /ready returns 200 for Ollama without needing external credentials."""
    ollama_settings = Settings(llm_provider=LLMProviderType.OLLAMA)
    app = create_app(settings=ollama_settings)
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["llm_provider"] == "ollama"
    assert data["configured"] is True
    assert data["connectivity_verified"] is False


def test_ready_endpoint_openrouter_with_credentials() -> None:
    """GET /ready returns 200 when OpenRouter has credentials configured."""
    secret_key = "sk-ant-test-api-key-12345"
    openrouter_settings = Settings(
        llm_provider=LLMProviderType.OPENROUTER,
        openrouter_api_key=secret_key,
    )
    app = create_app(settings=openrouter_settings)
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["llm_provider"] == "openrouter"
    assert data["configured"] is True
    assert data["connectivity_verified"] is False
    # Ensure raw secret is never in the response
    assert secret_key not in response.text


def test_ready_endpoint_openrouter_without_credentials() -> None:
    """GET /ready returns 503 when OpenRouter is active but missing credentials."""
    openrouter_settings = Settings(
        llm_provider=LLMProviderType.OPENROUTER,
        openrouter_api_key=None,
    )
    app = create_app(settings=openrouter_settings)
    client = TestClient(app)

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["configured"] is False
    assert "not configured" in data["details"].lower()


def test_submit_incident_valid_monitor_flow(stub_dependencies) -> None:
    """POST /incidents processes low-severity incidents to the monitor stage."""
    stub_dependencies(Severity.LOW)
    app = create_app()
    client = TestClient(app)

    payload = {
        "incident_description": "Elevated latency on payments api",
        "scenario": "healthy",
    }
    response = client.post("/incidents", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["outcome"] == "monitor"
    assert data["severity"] == "low"
    assert data["incident_id"].startswith("INC-")
    assert data["correlation_id"].startswith("CORR-")
    assert data["diagnostic"] is not None
    assert data["diagnostic"]["decision"] == "monitor"
    assert data["trace"] is not None


def test_submit_incident_custom_identifiers_preserved(stub_dependencies) -> None:
    """Caller-provided incident_id and correlation_id must be strictly preserved."""
    stub_dependencies(Severity.LOW)
    app = create_app()
    client = TestClient(app)

    payload = {
        "incident_description": "Minor database alert",
        "incident_id": "INC-CUSTOM-1234",
        "correlation_id": "CORR-FLOW-5678",
    }
    response = client.post("/incidents", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["incident_id"] == "INC-CUSTOM-1234"
    assert data["correlation_id"] == "CORR-FLOW-5678"
    assert data["trace"]["incident_id"] == "INC-CUSTOM-1234"
    assert data["trace"]["correlation_id"] == "CORR-FLOW-5678"


def test_submit_incident_recovery_flow(stub_dependencies) -> None:
    """POST /incidents processes critical incidents through full recovery and verification."""
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
    )
    app = create_app()
    client = TestClient(app)

    payload = {
        "incident_description": "Database connection pool saturated",
        "scenario": "connection_pool_exhaustion",
        "post_remediation_scenario": "healthy",
    }
    response = client.post("/incidents", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["outcome"] == "resolve"
    assert data["severity"] == "critical"
    assert data["verification"] is not None
    assert data["verification"]["status"] == "recovered"
    assert data["diagnostic"]["trigger_type"] == "verification_recovered"


def test_submit_incident_validation_error_empty_description() -> None:
    """POST /incidents returns 422 for empty or whitespace incident descriptions."""
    app = create_app()
    client = TestClient(app)

    response = client.post("/incidents", json={"incident_description": "   "})
    assert response.status_code == 422
    data = response.json()
    assert "validation error" in data["detail"].lower()


def test_submit_incident_validation_error_invalid_scenario() -> None:
    """POST /incidents returns 422 when an unknown scenario is supplied."""
    app = create_app()
    client = TestClient(app)

    payload = {
        "incident_description": "Valid incident description",
        "scenario": "unsupported_scenario_xyz",
    }
    response = client.post("/incidents", json=payload)
    assert response.status_code == 422


def test_submit_incident_rate_limit_mapped_to_503(stub_dependencies) -> None:
    """LLMRateLimitError raised by workflow maps to safe 503 response."""
    stub_dependencies(
        Severity.CRITICAL,
        error=LLMRateLimitError("Account rate limit exceeded"),
    )
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/incidents",
        json={"incident_description": "Payment service down"},
    )
    assert response.status_code == 503
    data = response.json()
    assert "rate limit exceeded" in data["detail"].lower()


def test_submit_incident_configuration_error_mapped_to_503(stub_dependencies) -> None:
    """LLMConfigurationError raised by workflow maps to safe 503 response."""
    stub_dependencies(
        Severity.CRITICAL,
        error=LLMConfigurationError("Missing provider setup"),
    )
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/incidents",
        json={"incident_description": "Payment service down"},
    )
    assert response.status_code == 503
    data = response.json()
    assert "misconfiguration" in data["detail"].lower()


def test_submit_incident_unexpected_error_mapped_to_500_without_leak(
    stub_dependencies,
) -> None:
    """Unhandled exceptions map to generic 500 without leaking stack traces or sensitive strings."""
    secret_token = "db-password-super-secret-xyz"
    stub_dependencies(
        Severity.CRITICAL,
        error=RuntimeError(f"Catastrophic failure with key={secret_token}"),
    )
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/incidents",
        json={"incident_description": "Payment service down"},
    )
    assert response.status_code == 500
    data = response.json()
    assert "internal error" in data["detail"].lower()
    # Confirm secret token is not in response
    assert secret_token not in response.text


def test_submit_incident_sanitized_trace_no_secret_leak(stub_dependencies) -> None:
    """Investigation and tool argument secrets remain sanitized in API responses."""
    sensitive_evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={
                "service": "payment",
                "api_key": "sk-ant-secret-key-12345",
            },
            raw_output=get_mock_metrics("payment", Scenario.HEALTHY),
            observation="Metrics with auth token",
        )
    ]
    investigation = InvestigationResult(
        summary="Investigation completed",
        model_assessed_confidence=0.9,
        recommended_next_action="Remediate",
        termination_reason=TerminationReason.COMPLETED,
        evidence=sensitive_evidence,
        findings=[_finding()],
    )
    stub_dependencies(Severity.CRITICAL, investigation=investigation)
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/incidents",
        json={"incident_description": "Payment outage with credentials"},
    )
    assert response.status_code == 200
    assert "sk-ant-secret-key-12345" not in response.text
