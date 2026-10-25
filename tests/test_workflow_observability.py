"""Integration regression tests for Phase 14 Step 3: Workflow Integration & Operational Diagnostics.

Validates:
1. Default trace identifier generation and preservation.
2. Supplied identifier and tracer preservation.
3. Trace attachment without breaking existing state fields.
4. Stage traces corresponding precisely to nodes that actually executed.
5. Consistent identifiers across events and traces.
6. Diagnostic decision and rationale consistency across resolve, escalate, and monitor paths.
7. Redaction of secrets in metadata and tool parameters.
8. Tracing failures do not change workflow results or routing.
9. No duplicate tool events.
10. Incident memory does not persist operational traces.
"""

from unittest.mock import Mock, patch

import pytest

from agentic_ai.agents.investigation import InvestigationError
from agentic_ai.domain.guardrails import Disposition, GuardedPlan, GuardedStep
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationActionType,
    RemediationExecution,
    RemediationPlan,
    RemediationStep,
    RiskTier,
    StepExecutionResult,
)
from agentic_ai.domain.verification import VerificationResult, VerificationStatus
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.observability.events import IncidentTrace
from agentic_ai.observability.tracer import OperationalTracer
from agentic_ai.tools.incident_memory import load_memory
from agentic_ai.tools.mock_tools import Scenario, get_mock_metrics, get_mock_pod_status
from agentic_ai.workflows import incident as workflow_module
from agentic_ai.workflows.incident import handle_incident


def _assessment(severity: Severity) -> SeverityAssessment:
    return SeverityAssessment(severity=severity, reasoning="Test severity assessment")


def _evidence(scenario: Scenario) -> list[Evidence]:
    return [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_metrics("payment", scenario),
            observation="Metrics data retrieved",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_pod_status("payment", scenario),
            observation="Pod status data retrieved",
        ),
    ]


def _investigation_result(
    termination: TerminationReason = TerminationReason.COMPLETED,
    scenario: Scenario = Scenario.HEALTHY,
    findings: list[Finding] | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        summary="Automated test investigation summary.",
        model_assessed_confidence=0.9,
        recommended_next_action="Remediation required.",
        termination_reason=termination,
        evidence=_evidence(scenario),
        findings=findings or [],
    )


def _finding() -> Finding:
    return Finding(
        statement="Database connection pool exhausted.",
        model_assessed_confidence=0.95,
    )


def _analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Database connection pool exhaustion on payment service.",
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )


@pytest.fixture
def stub_workflow_deps(monkeypatch):
    """Deterministic stubs for LLM-backed steps."""

    def _apply(
        severity: Severity = Severity.LOW,
        investigation: InvestigationResult | None = None,
        investigation_error: Exception | None = None,
        rca: RootCauseAnalysis | None = None,
        plan: RemediationPlan | None = None,
    ):
        monkeypatch.setattr(
            workflow_module,
            "assess_severity",
            lambda description: _assessment(severity),
        )

        def fake_investigate(description, scenario=None):
            if investigation_error is not None:
                raise investigation_error
            return investigation if investigation is not None else _investigation_result()

        monkeypatch.setattr(workflow_module, "investigate", fake_investigate)

        monkeypatch.setattr(
            workflow_module,
            "analyze_root_cause",
            lambda result: rca if rca is not None else _analysis(),
        )

        default_plan = RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                    target_service="payment",
                    parameters={"service": "payment", "increment": 20},
                    rationale="Increase connection pool size.",
                )
            ]
        )
        monkeypatch.setattr(
            workflow_module,
            "plan_remediation",
            lambda description, analysis: plan if plan is not None else default_plan,
        )

    return _apply


def test_existing_callers_without_trace_identifiers(stub_workflow_deps) -> None:
    """Existing callers providing no identifiers receive auto-generated IDs and valid trace."""
    stub_workflow_deps(Severity.LOW, investigation=_investigation_result())

    state = handle_incident("Minor latency warning", scenario=Scenario.HEALTHY)

    # Identifiers auto-generated with standard prefixes
    assert state["incident_id"] is not None
    assert state["incident_id"].startswith("INC-")
    assert state["correlation_id"] is not None
    assert state["correlation_id"].startswith("CORR-")

    # Trace object attached and matches identifiers
    trace = state["trace"]
    assert isinstance(trace, IncidentTrace)
    assert trace.incident_id == state["incident_id"]
    assert trace.correlation_id == state["correlation_id"]

    # Existing result fields remain backward-compatible
    assert state["incident_description"] == "Minor latency warning"
    assert state["next_step"] == "monitor"
    assert state["severity_assessment"] is not None


def test_supplied_trace_identifiers_preserved(stub_workflow_deps) -> None:
    """Caller-provided incident_id and correlation_id must be strictly preserved."""
    stub_workflow_deps(Severity.LOW, investigation=_investigation_result())

    custom_inc_id = "INC-USER-SPECIFIED-99"
    custom_corr_id = "CORR-FLOW-XYZ-123"

    state = handle_incident(
        "Low severity issue",
        scenario=Scenario.HEALTHY,
        incident_id=custom_inc_id,
        correlation_id=custom_corr_id,
    )

    assert state["incident_id"] == custom_inc_id
    assert state["correlation_id"] == custom_corr_id

    trace = state["trace"]
    assert trace is not None
    assert trace.incident_id == custom_inc_id
    assert trace.correlation_id == custom_corr_id


def test_supplied_tracer_instance_preserved(stub_workflow_deps) -> None:
    """When a pre-instantiated OperationalTracer is passed, it is used directly."""
    stub_workflow_deps(Severity.LOW, investigation=_investigation_result())

    custom_tracer = OperationalTracer(
        incident_id="INC-PREINST-001",
        correlation_id="CORR-PREINST-002",
    )

    state = handle_incident(
        "Low severity issue",
        scenario=Scenario.HEALTHY,
        tracer=custom_tracer,
    )

    assert state["tracer"] is custom_tracer
    assert state["incident_id"] == "INC-PREINST-001"
    assert state["correlation_id"] == "CORR-PREINST-002"
    assert state["trace"].incident_id == "INC-PREINST-001"


def test_stage_traces_correspond_to_nodes_actually_executed_normal_path(
    stub_workflow_deps,
) -> None:
    """On the monitor path, only executed nodes are recorded in trace.stages."""
    stub_workflow_deps(Severity.LOW, investigation=_investigation_result())

    state = handle_incident("Minor latency warning", scenario=Scenario.HEALTHY)

    trace = state["trace"]
    assert trace is not None
    executed_stages = [s.stage_name for s in trace.stages]

    # Executed nodes
    assert executed_stages == ["assess_severity", "investigate", "monitor"]

    # Skipped nodes must NOT appear in trace.stages
    assert "analyze_root_cause" not in executed_stages
    assert "plan_remediation" not in executed_stages
    assert "apply_guardrails" not in executed_stages
    assert "execute_remediation" not in executed_stages
    assert "verify_remediation" not in executed_stages
    assert "remember_incident" not in executed_stages
    assert "resolve" not in executed_stages
    assert "escalate" not in executed_stages


def test_stage_traces_correspond_to_nodes_actually_executed_recovery_path(
    stub_workflow_deps,
) -> None:
    """On the full recovery path, all remediation and verification nodes appear."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    state = handle_incident(
        "Payment service outage",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
    )

    assert state["next_step"] == "resolve"
    trace = state["trace"]
    assert trace is not None

    executed_stages = [s.stage_name for s in trace.stages]
    expected_stages = [
        "assess_severity",
        "investigate",
        "analyze_root_cause",
        "plan_remediation",
        "apply_guardrails",
        "execute_remediation",
        "verify_remediation",
        "remember_incident",
        "resolve",
    ]
    assert executed_stages == expected_stages
    assert "monitor" not in executed_stages
    assert "escalate" not in executed_stages


def test_diagnostic_reason_resolve_path(stub_workflow_deps) -> None:
    """Verified recovery produces a 'verification_recovered' diagnostic."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    state = handle_incident(
        "Payment service down",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
    )

    trace = state["trace"]
    assert trace is not None
    assert trace.final_outcome == "resolve"
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "resolve"
    assert trace.diagnostic.trigger_type == "verification_recovered"
    assert "recovered" in trace.diagnostic.primary_reason.lower()


def test_diagnostic_reason_monitor_path(stub_workflow_deps) -> None:
    """Low severity monitor outcome produces a 'severity_baseline' diagnostic."""
    stub_workflow_deps(Severity.LOW, investigation=_investigation_result())

    state = handle_incident("Minor logging jitter", scenario=Scenario.HEALTHY)

    trace = state["trace"]
    assert trace is not None
    assert trace.final_outcome == "monitor"
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "monitor"
    assert trace.diagnostic.trigger_type == "severity_baseline"


def test_diagnostic_reason_verification_unresolved(stub_workflow_deps) -> None:
    """Post-remediation anomalies produce a 'verification_unresolved' diagnostic on escalation."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    state = handle_incident(
        "Payment service down",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.final_outcome == "escalate"
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "verification_unresolved"


def test_diagnostic_reason_guardrail_rejection(stub_workflow_deps) -> None:
    """Blocked steps in guarded plan produce a 'guardrail_rejection' diagnostic."""
    blocked_plan = RemediationPlan(
        steps=[
            RemediationStep(
                action_type=RemediationActionType.ROLLBACK_DEPLOYMENT,
                target_service="payment",
                parameters={"service": "payment"},
                rationale="Emergency rollback",
            )
        ]
    )
    low_conf_rca = RootCauseAnalysis(
        root_cause="Uncertain root cause",
        root_cause_confidence=0.3,  # < 0.6 raises HIGH -> CRITICAL -> BLOCKED
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
        rca=low_conf_rca,
        plan=blocked_plan,
    )

    state = handle_incident(
        "Payment service down",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "guardrail_rejection"

    # Confirm blocked tool is recorded in tool traces
    blocked_tools = [t for t in trace.tools if t.status == "blocked"]
    assert len(blocked_tools) >= 1
    assert blocked_tools[0].tool_name == "rollback_deployment"
    assert blocked_tools[0].target_service == "payment"


def test_diagnostic_reason_investigation_incomplete(stub_workflow_deps) -> None:
    """Incomplete or failed investigation produces an 'investigation_incomplete' diagnostic."""
    stub_workflow_deps(
        Severity.LOW,
        investigation_error=InvestigationError("Upstream metrics endpoint timed out"),
    )

    state = handle_incident("Mild report", scenario=Scenario.HEALTHY)

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "investigation_incomplete"
    assert "timed out" in trace.diagnostic.primary_reason


def test_no_duplicate_tool_events(stub_workflow_deps) -> None:
    """Verify tool invocations are traced without duplicates."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    state = handle_incident(
        "Payment service down",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
    )

    trace = state["trace"]
    assert trace is not None

    tool_names = [t.tool_name for t in trace.tools]
    # Expected: get_mock_metrics (1), get_mock_pod_status (1), increase_connection_pool (1), verify_service_health (1)
    assert tool_names.count("get_mock_metrics") == 1
    assert tool_names.count("get_mock_pod_status") == 1
    assert tool_names.count("increase_connection_pool") == 1
    assert tool_names.count("verify_service_health") == 1
    assert len(trace.tools) == 4


def test_secrets_redacted_in_recorded_tool_traces(stub_workflow_deps) -> None:
    """Sensitive parameters in investigation or remediation tools are redacted in traces."""
    sensitive_evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={
                "service": "payment",
                "api_key": "sk-ant-api03-abcdefg1234567890",
                "auth_token": "Bearer super-secret-password-12345",
            },
            raw_output=get_mock_metrics("payment", Scenario.HEALTHY),
            observation="Metrics data retrieved with credentials",
        )
    ]
    investigation = InvestigationResult(
        summary="Investigation with sensitive credentials.",
        model_assessed_confidence=0.9,
        recommended_next_action="Remediation required.",
        termination_reason=TerminationReason.COMPLETED,
        evidence=sensitive_evidence,
        findings=[_finding()],
    )
    step = RemediationStep(
        action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
        target_service="payment",
        rationale="Apply config with secret token sk-ant-api03-secretkey123",
    )
    sensitive_plan = RemediationPlan(steps=[step])
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=investigation,
        plan=sensitive_plan,
    )

    state = handle_incident(
        "Payment service down",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
    )

    trace = state["trace"]
    assert trace is not None

    # Check investigation tool arguments were redacted
    investigation_tool = next(t for t in trace.tools if t.tool_name == "get_mock_metrics")
    args = investigation_tool.sanitized_arguments
    assert args["api_key"] == "[REDACTED]"
    assert args["auth_token"] == "[REDACTED]"
    assert args["service"] == "payment"

    # Check remediation step rationale with embedded secret was masked
    remediation_tool = next(t for t in trace.tools if t.tool_name == "increase_connection_pool")
    rem_args = remediation_tool.sanitized_arguments
    assert "sk-ant-api03-secretkey123" not in rem_args["rationale"]
    assert "[REDACTED]" in rem_args["rationale"]

    # Confirm raw secrets never appear anywhere in the serialized JSON
    json_repr = trace.model_dump_json()
    assert "super-secret-password-12345" not in json_repr
    assert "sk-ant-api03-abcdefg1234567890" not in json_repr
    assert "sk-ant-api03-secretkey123" not in json_repr


def test_tracing_failure_does_not_alter_workflow_routing(monkeypatch, stub_workflow_deps) -> None:
    """Errors inside tracing methods must never crash the workflow or change outcomes."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    # Injected catastrophic failure inside tracer.stage_span
    def exploding_stage_span(*args, **kwargs):
        raise RuntimeError("Catastrophic tracing hardware failure")

    tracer = OperationalTracer()
    monkeypatch.setattr(tracer, "stage_span", exploding_stage_span)

    # Workflow must still complete and route to resolve
    state = handle_incident(
        "Payment service outage",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
        tracer=tracer,
    )

    assert state["next_step"] == "resolve"
    assert state["verification"] is not None
    assert state["verification"].status == VerificationStatus.RECOVERED


def test_incident_memory_contains_no_operational_traces(stub_workflow_deps) -> None:
    """Incident memory must only persist domain knowledge, never operational traces."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    state = handle_incident(
        "Payment service outage for memory check",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=Scenario.HEALTHY,
    )

    assert state["memory_recorded"] is True
    entries = load_memory()
    assert len(entries) >= 1
    target_entry = next(
        e for e in entries if "Payment service outage for memory check" in e.symptom_description
    )

    # Memory entry should have domain attributes only
    assert hasattr(target_entry, "root_cause")
    assert hasattr(target_entry, "resolution")
    assert not hasattr(target_entry, "trace")
    assert not hasattr(target_entry, "tracer")
    assert not hasattr(target_entry, "diagnostic")

    # The entry representation should not contain operational trace class names
    entry_dict = target_entry.model_dump()
    assert "OperationalTracer" not in str(entry_dict)
    assert "IncidentTrace" not in str(entry_dict)


def test_diagnostic_reason_evidence_metrics(stub_workflow_deps) -> None:
    """Low assessed severity with dangerous raw telemetry triggers 'evidence_metrics' escalation."""
    stub_workflow_deps(
        Severity.LOW,
        investigation=_investigation_result(scenario=Scenario.UPSTREAM_DEPENDENCY_FAILURE),
    )

    state = handle_incident(
        "Low severity report",
        scenario=Scenario.UPSTREAM_DEPENDENCY_FAILURE,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "evidence_metrics"
    assert "exceeded" in trace.diagnostic.primary_reason.lower()


def test_diagnostic_reason_severity_baseline_escalate(stub_workflow_deps) -> None:
    """High initial severity with no findings/remediation triggers 'severity_baseline' escalation."""
    stub_workflow_deps(
        Severity.HIGH,
        investigation=_investigation_result(findings=[]),  # No findings -> no RCA -> no remediation
    )

    state = handle_incident(
        "Severe payment issue reported",
        scenario=Scenario.HEALTHY,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "severity_baseline"
    assert "high" in trace.diagnostic.primary_reason.lower()


def test_diagnostic_reason_execution_failure(stub_workflow_deps, monkeypatch) -> None:
    """Remediation execution failures trigger 'execution_failure' diagnostic."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    failed_execution = RemediationExecution(
        step_results=[
            StepExecutionResult(
                step=RemediationStep(
                    action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                    target_service="payment",
                    rationale="Fix pool exhaustion",
                ),
                status=ExecutionStatus.FAILED,
                output="API call timed out during connection pool update",
                error="TimeoutError",
            )
        ]
    )
    monkeypatch.setattr(
        workflow_module,
        "execute_guarded_plan",
        lambda guarded: failed_execution,
    )

    state = handle_incident(
        "Payment service outage",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "execution_failure"


def test_diagnostic_reason_verification_inconclusive(stub_workflow_deps, monkeypatch) -> None:
    """Inconclusive post-remediation verification triggers 'verification_unresolved'."""
    stub_workflow_deps(
        Severity.CRITICAL,
        investigation=_investigation_result(findings=[_finding()]),
    )

    inconclusive_res = VerificationResult(
        status=VerificationStatus.INCONCLUSIVE,
        attempts=1,
        reasons=["telemetry gaps prevent confirming healthy state"],
        evidence=[],
    )
    monkeypatch.setattr(
        workflow_module,
        "verify_service_health",
        lambda *args, **kwargs: inconclusive_res,
    )

    state = handle_incident(
        "Payment service outage",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["next_step"] == "escalate"
    trace = state["trace"]
    assert trace is not None
    assert trace.diagnostic is not None
    assert trace.diagnostic.decision == "escalate"
    assert trace.diagnostic.trigger_type == "verification_unresolved"
    assert "inconclusive" in trace.diagnostic.primary_reason.lower()


def test_finalization_idempotency_and_timestamp_ordering() -> None:
    """Finalization is strictly idempotent and timestamps obey start_time <= end_time and duration >= 0."""
    import time

    tracer = OperationalTracer(incident_id="INC-IDEMP-01", correlation_id="CORR-IDEMP-02")
    with tracer.stage_span("assess_severity"):
        pass

    tracer.set_final_outcome(
        "resolve",
        diagnostic=workflow_module.DiagnosticReason(
            decision="resolve",
            primary_reason="Initial resolution",
            trigger_type="verification_recovered",
        ),
    )

    snap1 = tracer.get_trace()
    time.sleep(0.005)

    # Secondary attempt to overwrite final outcome must be ignored
    tracer.set_final_outcome(
        "escalate",
        diagnostic=workflow_module.DiagnosticReason(
            decision="escalate",
            primary_reason="Overwriting attempt",
            trigger_type="guardrail_rejection",
        ),
    )

    snap2 = tracer.get_trace()

    assert snap1.final_outcome == "resolve"
    assert snap2.final_outcome == "resolve"
    assert snap2.diagnostic.trigger_type == "verification_recovered"
    assert snap1.end_time == snap2.end_time
    assert snap1.total_duration_ms == snap2.total_duration_ms

    # Timestamp ordering check on stage spans
    for stage in snap2.stages:
        assert stage.start_time <= stage.end_time
        assert stage.duration_ms >= 0.0

