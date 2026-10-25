import pytest

from agentic_ai.agents.investigation import InvalidConclusionError
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
from agentic_ai.llm.rca import InvalidRootCauseAnalysisError
from agentic_ai.llm.remediation import InvalidRemediationPlanError
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.tools.mock_tools import (
    Scenario,
    get_mock_metrics,
    get_mock_pod_status,
)
from agentic_ai.workflows import incident as workflow_module
from agentic_ai.workflows.incident import handle_incident, route_by_severity


def _assessment(severity: Severity) -> SeverityAssessment:
    return SeverityAssessment(severity=severity, reasoning="test reasoning")


def _evidence(scenario: Scenario) -> list[Evidence]:
    """Real tool output, so the escalation policy has numbers to read."""
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
        investigation=None,
        error: Exception | None = None,
        rca_result=None,
        rca_error: Exception | None = None,
        plan_error: Exception | None = None,
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

        def fake_analyze_root_cause(result):
            if rca_error is not None:
                raise rca_error
            return rca_result if rca_result is not None else _analysis()

        monkeypatch.setattr(workflow_module, "analyze_root_cause", fake_analyze_root_cause)

        def fake_plan_remediation(description, analysis):
            if plan_error is not None:
                raise plan_error
            return RemediationPlan(
                steps=[
                    RemediationStep(
                        action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                        target_service="payment",
                        rationale="Pool is exhausted.",
                    )
                ]
            )

        monkeypatch.setattr(workflow_module, "plan_remediation", fake_plan_remediation)

    return _apply


# --- flow ---------------------------------------------------------------


def test_every_incident_is_investigated_before_routing(stub_dependencies) -> None:
    """Even a critical incident is investigated, so escalation carries evidence."""
    stub_dependencies(Severity.CRITICAL)

    state = handle_incident("Payment service is down")

    assert state["investigation"] is not None
    assert state["next_step"] == "escalate"


def test_high_severity_escalates(stub_dependencies) -> None:
    stub_dependencies(Severity.HIGH)

    assert handle_incident("Elevated errors")["next_step"] == "escalate"


def test_low_severity_with_complete_investigation_is_monitored(
    stub_dependencies,
) -> None:
    stub_dependencies(Severity.LOW)

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["investigation"].is_complete


def test_severity_assessment_is_carried_into_state(stub_dependencies) -> None:
    stub_dependencies(Severity.MEDIUM)

    state = handle_incident("Slow checkout")

    assert state["severity_assessment"].severity is Severity.MEDIUM


# --- degradation --------------------------------------------------------


def test_incomplete_investigation_escalates_even_when_severity_is_low(
    stub_dependencies,
) -> None:
    """An unexplained incident gets human eyes regardless of severity."""
    stub_dependencies(
        Severity.LOW,
        investigation=_result(TerminationReason.MAX_STEPS_REACHED),
    )

    assert handle_incident("Minor logging delay")["next_step"] == "escalate"


def test_failed_investigation_does_not_sink_the_workflow(
    stub_dependencies,
) -> None:
    """The incident still needs routing even if investigation failed."""
    stub_dependencies(
        Severity.LOW,
        error=InvalidConclusionError("model produced nothing usable"),
    )

    state = handle_incident("Minor logging delay")

    assert state["investigation"] is None
    assert "nothing usable" in state["investigation_error"]
    assert state["next_step"] == "escalate"


def test_unexpected_errors_are_not_swallowed(stub_dependencies) -> None:
    """Only investigation failures are handled; real bugs must surface."""
    stub_dependencies(Severity.LOW, error=RuntimeError("connection refused"))

    with pytest.raises(RuntimeError, match="connection refused"):
        handle_incident("Minor logging delay")


# --- routing in isolation -----------------------------------------------


def test_router_requires_a_severity_assessment() -> None:
    with pytest.raises(ValueError, match="Severity assessment is missing"):
        route_by_severity(
            {
                "incident_description": "x",
                "scenario": "connection_pool_exhaustion",
                "severity_assessment": None,
                "investigation": _result(),
                "investigation_error": None,
                "next_step": None,
            }
        )


# --- escalation on evidence ---------------------------------------------


def test_evidence_escalates_even_when_severity_looks_mild(
    stub_dependencies,
) -> None:
    """A mildly-worded report with alarming numbers must still escalate.

    Severity is assessed from the description before any data exists, so it
    cannot be the only input to the decision.
    """
    stub_dependencies(
        Severity.LOW,
        investigation=_result(scenario=Scenario.UPSTREAM_DEPENDENCY_FAILURE),
    )

    assert handle_incident("Occasional checkout hiccups")["next_step"] == "escalate"


def test_no_evidence_escalates(stub_dependencies) -> None:
    """An incident nobody could gather data on needs a human."""
    stub_dependencies(
        Severity.LOW,
        investigation=InvestigationResult(
            summary="Nothing gathered.",
            model_assessed_confidence=0.1,
            recommended_next_action="Escalate.",
            termination_reason=TerminationReason.COMPLETED,
        ),
    )

    assert handle_incident("Something odd")["next_step"] == "escalate"


# --- root cause analysis integration -------------------------------------


def test_rca_runs_on_the_escalation_path_when_findings_exist(
    stub_dependencies,
) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is not None
    assert state["root_cause_analysis"].root_cause == "Database connection pool exhaustion."
    assert state["root_cause_analysis_error"] is None


def test_rca_does_not_run_on_the_monitor_path(stub_dependencies) -> None:
    stub_dependencies(
        Severity.LOW,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["root_cause_analysis"] is None


def test_rca_is_skipped_when_investigation_produced_no_findings(
    stub_dependencies,
) -> None:
    """Nothing to synthesize - RCA must not be asked to invent a conclusion."""
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[]),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is None
    assert state["root_cause_analysis_error"] is None


def test_rca_is_skipped_when_investigation_itself_failed(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        error=InvalidConclusionError("model produced nothing usable"),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["investigation"] is None
    assert state["root_cause_analysis"] is None


def test_rca_failure_does_not_block_escalation(stub_dependencies) -> None:
    """A human is already being paged; a missing RCA should not stop that."""
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_error=InvalidRootCauseAnalysisError("model could not produce a valid analysis"),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is None
    assert "could not produce" in state["root_cause_analysis_error"]


def test_unexpected_rca_errors_are_not_swallowed(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_error=RuntimeError("connection refused"),
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        handle_incident("Payment service is down")


def test_routing_is_unaffected_by_rca_regardless_of_its_content(
    stub_dependencies,
) -> None:
    """Routing is decided before RCA runs - this is enforced by graph order,
    not by RCA's content, but a passing test here is a regression guard if
    that ever changes.
    """
    low_confidence_analysis = RootCauseAnalysis(
        root_cause="Unclear.",
        root_cause_confidence=0.05,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_result=low_confidence_analysis,
    )

    # Severity alone already routes this to escalate; a low-confidence RCA
    # conclusion must not change that, since RCA runs strictly after routing.
    assert handle_incident("Payment service is down")["next_step"] == "escalate"


# --- remediation planning integration ------------------------------------


def test_remediation_plan_is_produced_after_rca(stub_dependencies) -> None:
    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["remediation_plan"].steps[0].target_service == "payment"


def test_remediation_is_skipped_without_an_rca(stub_dependencies) -> None:
    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[]))

    state = handle_incident("Payment service is down")

    assert state["remediation_plan"] is None
    assert state["remediation_plan_error"] is None


def test_remediation_does_not_run_on_the_monitor_path(stub_dependencies) -> None:
    stub_dependencies(Severity.LOW, investigation=_result(findings=[_finding()]))

    assert handle_incident("Minor logging delay")["remediation_plan"] is None


def test_remediation_failure_does_not_block_escalation(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        plan_error=InvalidRemediationPlanError("no valid plan"),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["remediation_plan"] is None
    assert "no valid plan" in state["remediation_plan_error"]


def test_unexpected_remediation_errors_are_not_swallowed(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        plan_error=RuntimeError("connection refused"),
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        handle_incident("Payment service is down")


# --- long-term memory integration ----------------------------------------


def test_escalated_incident_is_remembered_as_learned(stub_dependencies) -> None:
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    assert state["memory_recorded"] is True
    (entry,) = load_memory()
    assert entry.source == "learned"
    assert entry.root_cause == "Database connection pool exhaustion."
    assert "increase_connection_pool" in entry.resolution


def test_same_incident_is_not_remembered_twice(stub_dependencies) -> None:
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    handle_incident("Payment service is down")
    second = handle_incident("Payment service is down")

    assert second["memory_recorded"] is False
    assert len(load_memory()) == 1


def test_nothing_is_remembered_without_an_rca_or_on_the_monitor_path(
    stub_dependencies,
) -> None:
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[]))
    handle_incident("Payment service is down")
    stub_dependencies(Severity.LOW, investigation=_result(findings=[_finding()]))
    handle_incident("Minor logging delay")

    assert load_memory() == []


def test_memory_write_failure_does_not_block_escalation(stub_dependencies, monkeypatch) -> None:
    def broken_remember(entry):
        raise OSError("disk full")

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))
    monkeypatch.setattr(workflow_module, "remember", broken_remember)

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["memory_recorded"] is False


# --- guardrails integration ----------------------------------------------


def test_guardrails_judge_the_plan_on_the_escalation_path(stub_dependencies) -> None:
    from agentic_ai.domain.guardrails import Disposition

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    (guarded,) = state["guarded_plan"].steps
    # increase_connection_pool is MEDIUM; payment was examined, so it passes.
    assert guarded.disposition == Disposition.ALLOWED
    assert state["next_step"] == "escalate"


def test_guardrails_do_not_run_without_a_plan(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        plan_error=InvalidRemediationPlanError("no valid plan"),
    )

    assert handle_incident("Payment service is down")["guarded_plan"] is None


# --- remediation execution integration ----------------------------------


def test_allowed_step_executes_in_workflow(stub_dependencies) -> None:
    from agentic_ai.domain.remediation import ExecutionStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    assert state["remediation_execution"] is not None
    (result,) = state["remediation_execution"].step_results
    assert result.status == ExecutionStatus.EXECUTED
    assert "[SIMULATED]" in result.output
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Executed: increase_connection_pool on payment" in entry.resolution
    assert "Recovery: unresolved" in entry.resolution


def test_approval_pending_step_is_not_executed_in_workflow(stub_dependencies, monkeypatch) -> None:
    from agentic_ai.domain.remediation import ExecutionStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda desc, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.ROLLBACK_DEPLOYMENT,
                    target_service="payment",
                    rationale="Roll back faulty deploy.",
                )
            ]
        ),
    )

    state = handle_incident("Payment service is down")

    assert state["remediation_execution"] is not None
    (result,) = state["remediation_execution"].step_results
    assert result.status == ExecutionStatus.SKIPPED
    assert "awaiting human approval" in result.output
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Skipped: rollback_deployment on payment" in entry.resolution
    assert "Recovery: skipped" in entry.resolution


def test_approval_rejected_step_is_not_executed_in_workflow(stub_dependencies, monkeypatch) -> None:
    from agentic_ai.domain.guardrails import guard_plan
    from agentic_ai.domain.remediation import ExecutionStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda desc, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.ROLLBACK_DEPLOYMENT,
                    target_service="payment",
                    rationale="Roll back deploy.",
                )
            ]
        ),
    )

    orig_guard_plan = guard_plan

    def reject_plan(plan, analysis, investigated):
        guarded = orig_guard_plan(plan, analysis, investigated)
        return guarded.reject(0, "lead_sre")

    monkeypatch.setattr(workflow_module, "guard_plan", reject_plan)

    state = handle_incident("Payment service is down")

    assert state["remediation_execution"] is not None
    (result,) = state["remediation_execution"].step_results
    assert result.status == ExecutionStatus.SKIPPED
    assert "rejected by lead_sre" in result.output
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Skipped: rollback_deployment on payment" in entry.resolution
    assert "Recovery: skipped" in entry.resolution


def test_blocked_step_never_reaches_mock_handler_in_workflow(
    stub_dependencies, monkeypatch
) -> None:
    from unittest.mock import Mock

    from agentic_ai.domain.remediation import ExecutionStatus
    from agentic_ai.tools import mock_remediation
    from agentic_ai.tools.incident_memory import load_memory

    low_confidence_analysis = RootCauseAnalysis(
        root_cause="Uncertain root cause.",
        root_cause_confidence=0.2,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_result=low_confidence_analysis,
    )
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda desc, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.FAILOVER,
                    target_service="payment",
                    rationale="Attempt failover.",
                )
            ]
        ),
    )

    spy_handler = Mock(wraps=mock_remediation.execute_mock_remediation_step)
    monkeypatch.setattr(mock_remediation, "execute_mock_remediation_step", spy_handler)

    state = handle_incident("Payment service is down")

    assert state["remediation_execution"] is not None
    (result,) = state["remediation_execution"].step_results
    assert result.status == ExecutionStatus.BLOCKED
    assert "blocked" in result.output.lower()
    spy_handler.assert_not_called()
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Blocked: failover on payment" in entry.resolution
    assert "Recovery: skipped" in entry.resolution


def test_execution_results_propagate_through_workflow_state(stub_dependencies) -> None:
    from agentic_ai.domain.remediation import RemediationExecution

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    assert "remediation_execution" in state
    assert isinstance(state["remediation_execution"], RemediationExecution)
    assert len(state["remediation_execution"].step_results) == 1
    assert state["remediation_execution"].completed_at.tzinfo is not None


def test_remediation_execution_does_not_run_on_monitor_path(stub_dependencies) -> None:
    stub_dependencies(
        Severity.LOW,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["remediation_execution"] is None


def test_accurate_incident_memory_summaries_with_multiple_steps(
    stub_dependencies, monkeypatch
) -> None:
    from agentic_ai.domain.remediation import ExecutionStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda desc, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.SCALE_UP,
                    target_service="payment",
                    rationale="Scale up.",
                ),
                RemediationStep(
                    action_type=RemediationActionType.ROLLBACK_DEPLOYMENT,
                    target_service="payment",
                    rationale="Roll back.",
                ),
            ]
        ),
    )

    state = handle_incident("Payment service is down")

    assert state["remediation_execution"] is not None
    results = state["remediation_execution"].step_results
    assert len(results) == 2
    assert results[0].status == ExecutionStatus.EXECUTED
    assert results[1].status == ExecutionStatus.SKIPPED

    (entry,) = load_memory()
    assert (
        "Executed: scale_up on payment; Skipped: rollback_deployment on payment" in entry.resolution
    )
    assert "Recovery: unresolved" in entry.resolution


def test_escalation_behavior_remains_unchanged(stub_dependencies) -> None:
    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"


# --- verification integration (Phase 12) ---------------------------------


def test_successful_recovery_routes_to_resolve_node(stub_dependencies) -> None:
    from agentic_ai.domain.verification import VerificationStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident(
        "Payment service is down",
        post_remediation_scenario=Scenario.HEALTHY,
    )

    assert state["verification"] is not None
    assert state["verification"].status == VerificationStatus.RECOVERED
    assert state["verification"].is_recovered is True
    assert state["next_step"] == "resolve"

    (entry,) = load_memory()
    assert "Recovery: verified recovered" in entry.resolution


def test_failed_recovery_routes_to_escalate(stub_dependencies) -> None:
    from agentic_ai.domain.verification import VerificationStatus
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident(
        "Payment service is down",
        post_remediation_scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["verification"] is not None
    assert state["verification"].status == VerificationStatus.UNRESOLVED
    assert state["verification"].is_recovered is False
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Recovery: unresolved" in entry.resolution


def test_inconclusive_verification_routes_to_escalate(stub_dependencies, monkeypatch) -> None:
    from agentic_ai.domain.verification import (
        VerificationResult,
        VerificationStatus,
    )
    from agentic_ai.tools.incident_memory import load_memory

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))
    monkeypatch.setattr(
        workflow_module,
        "verify_service_health",
        lambda service, scenario, attempts: VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            attempts=1,
            reasons=["could not read output from: get_mock_metrics"],
        ),
    )

    state = handle_incident("Payment service is down")

    assert state["verification"] is not None
    assert state["verification"].status == VerificationStatus.INCONCLUSIVE
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Recovery: inconclusive" in entry.resolution


def test_skipped_verification_when_remediation_blocked_routes_to_escalate(
    stub_dependencies, monkeypatch
) -> None:
    from agentic_ai.domain.verification import VerificationStatus
    from agentic_ai.tools.incident_memory import load_memory

    low_confidence_analysis = RootCauseAnalysis(
        root_cause="Uncertain.",
        root_cause_confidence=0.2,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_result=low_confidence_analysis,
    )
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda desc, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.FAILOVER,
                    target_service="payment",
                    rationale="Failover.",
                )
            ]
        ),
    )

    state = handle_incident("Payment service is down")

    assert state["verification"] is not None
    assert state["verification"].status == VerificationStatus.SKIPPED
    assert state["next_step"] == "escalate"

    (entry,) = load_memory()
    assert "Recovery: skipped" in entry.resolution


def test_never_automatically_reruns_remediation(stub_dependencies, monkeypatch) -> None:
    from unittest.mock import Mock

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    spy_executor = Mock(wraps=workflow_module.execute_guarded_plan)
    monkeypatch.setattr(workflow_module, "execute_guarded_plan", spy_executor)

    # Verification fails (unresolved)
    state = handle_incident(
        "Payment service is down",
        post_remediation_scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
    )

    assert state["next_step"] == "escalate"
    # Executor must have been called exactly once: no repeat loop!
    assert spy_executor.call_count == 1


def test_verification_results_propagate_in_workflow_state(
    stub_dependencies,
) -> None:
    from agentic_ai.domain.verification import VerificationResult

    stub_dependencies(Severity.CRITICAL, investigation=_result(findings=[_finding()]))

    state = handle_incident(
        "Payment service is down",
        post_remediation_scenario=Scenario.HEALTHY,
    )

    assert "verification" in state
    assert isinstance(state["verification"], VerificationResult)
    assert state["verification"].attempts == 1
    assert state["verification"].verified_at.tzinfo is not None


def test_monitor_path_bypasses_verification(stub_dependencies) -> None:
    stub_dependencies(
        Severity.LOW,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["verification"] is None
