"""Deterministic evaluation runner for the Incident Resolution Platform.

Executes evaluation cases (such as the 9 golden cases) through the incident
workflow deterministically, with zero live LLM dependencies and zero network access.
Uses controlled fixtures and stubs to evaluate Gate A, Gate B, guardrails,
remediation execution, and verification invariants.
"""

from collections.abc import Sequence
from unittest.mock import Mock, patch

from agentic_ai.domain.guardrails import Disposition
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.rca import (
    KnowledgeBaseConsistency,
    RootCauseAnalysis,
)
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationActionType,
    RemediationPlan,
    RemediationStep,
)
from agentic_ai.domain.verification import (
    VerificationStatus,
    evaluate_verification,
)
from agentic_ai.evaluation.dataset import GOLDEN_DATASET
from agentic_ai.evaluation.metrics import (
    check_all_invariants,
    check_case_invariants,
    compute_evaluation_metrics,
)
from agentic_ai.evaluation.models import (
    ActualExecution,
    CaseEvaluationResult,
    EvalCase,
    EvalCategory,
    EvaluationReport,
    GateADecision,
    GateBOutcome,
)
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.tools import mock_remediation, mock_verification
from agentic_ai.tools.mock_tools import get_mock_metrics, get_mock_pod_status
from agentic_ai.workflows import incident as workflow_module
from agentic_ai.workflows.incident import incident_workflow


def run_case(case: EvalCase) -> CaseEvaluationResult:
    """Execute a single evaluation case through the incident workflow.

    Deterministic and self-contained:
    - Injected fixtures provide pre-computed investigation, RCA, or remediation plans.
    - Observability fixtures (and any telemetry overrides) supply telemetry evidence.
    - Memory storage is spied so evaluation runs never mutate on-disk data.
    - Zero live LLM calls, zero network calls.
    """
    service = case.fixture.mock_service or "payment"

    # 1. Deterministic severity assessment stub
    def fake_assess_severity(description: str) -> SeverityAssessment:
        if case.fixture.injected_severity is not None:
            return SeverityAssessment(
                severity=case.fixture.injected_severity,
                reasoning=f"Assessed severity: {case.fixture.injected_severity.value}.",
            )
        if case.category == EvalCategory.NORMAL:
            return SeverityAssessment(
                severity=Severity.LOW,
                reasoning="Routine operational check; normal telemetry.",
            )
        return SeverityAssessment(
            severity=Severity.HIGH,
            reasoning="Default assessed severity: high impact.",
        )

    # 2. Deterministic investigation stub
    def fake_investigate(description: str, scenario: str | None = None) -> InvestigationResult:
        if case.fixture.injected_investigation is not None:
            return case.fixture.injected_investigation

        effective_scenario = scenario or case.initial_scenario

        if case.fixture.telemetry_override:
            metrics_raw = case.fixture.telemetry_override.get(
                "get_mock_metrics",
                get_mock_metrics(service, scenario=effective_scenario),
            )
            pods_raw = case.fixture.telemetry_override.get(
                "get_mock_pod_status",
                get_mock_pod_status(service, scenario=effective_scenario),
            )
        else:
            metrics_raw = get_mock_metrics(service, scenario=effective_scenario)
            pods_raw = get_mock_pod_status(service, scenario=effective_scenario)

        evidence = [
            Evidence(
                source_tool="get_mock_metrics",
                tool_arguments={"service": service},
                raw_output=metrics_raw,
                observation=f"Metrics telemetry for {service}",
            ),
            Evidence(
                source_tool="get_mock_pod_status",
                tool_arguments={"service": service},
                raw_output=pods_raw,
                observation=f"Pod status telemetry for {service}",
            ),
        ]

        if case.category == EvalCategory.NORMAL:
            findings: list[Finding] = []
        else:
            findings = [
                Finding(
                    statement=f"Telemetry indicates an incident on service '{service}'.",
                    model_assessed_confidence=0.9,
                )
            ]

        return InvestigationResult(
            summary=f"Automated evaluation investigation for {service}.",
            evidence=evidence,
            findings=findings,
            model_assessed_confidence=0.9,
            recommended_next_action="Proceed to root cause analysis.",
            termination_reason=TerminationReason.COMPLETED,
        )

    # 3. Deterministic RCA stub
    def fake_analyze_root_cause(result: InvestigationResult) -> RootCauseAnalysis:
        if case.fixture.injected_rca is not None:
            return case.fixture.injected_rca
        return RootCauseAnalysis(
            root_cause="Database connection pool exhaustion.",
            root_cause_confidence=0.9,
            knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
        )

    # 4. Deterministic remediation planning stub
    def fake_plan_remediation(description: str, analysis: RootCauseAnalysis) -> RemediationPlan:
        if case.fixture.injected_plan is not None:
            return case.fixture.injected_plan
        return RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                    target_service=service,
                    rationale="Mitigate connection pool exhaustion.",
                )
            ]
        )

    # 5. Deterministic verification tool stub
    def fake_verify_service_health(
        service: str,
        scenario: str,
        attempts: int = 1,
    ):
        if case.fixture.telemetry_override:
            metrics_raw = case.fixture.telemetry_override.get(
                "get_mock_metrics",
                get_mock_metrics(service, scenario=scenario),
            )
            pods_raw = case.fixture.telemetry_override.get(
                "get_mock_pod_status",
                get_mock_pod_status(service, scenario=scenario),
            )
            evidence = [
                Evidence(
                    source_tool="get_mock_metrics",
                    tool_arguments={"service": service},
                    raw_output=metrics_raw,
                    observation="Post-remediation metrics check",
                ),
                Evidence(
                    source_tool="get_mock_pod_status",
                    tool_arguments={"service": service},
                    raw_output=pods_raw,
                    observation="Post-remediation pod status check",
                ),
            ]
            return evaluate_verification(evidence, attempts=attempts)
        return mock_verification.verify_service_health(
            service,
            scenario=scenario,
            attempts=attempts,
        )

    # 6. Spies for memory and remediation execution count
    spy_remember = Mock(return_value=True)
    spy_execute_guarded_plan = Mock(wraps=mock_remediation.execute_guarded_plan)

    with (
        patch.object(workflow_module, "assess_severity", fake_assess_severity),
        patch.object(workflow_module, "investigate", fake_investigate),
        patch.object(workflow_module, "analyze_root_cause", fake_analyze_root_cause),
        patch.object(workflow_module, "plan_remediation", fake_plan_remediation),
        patch.object(workflow_module, "verify_service_health", fake_verify_service_health),
        patch.object(workflow_module, "remember", spy_remember),
        patch.object(workflow_module, "execute_guarded_plan", spy_execute_guarded_plan),
    ):
        final_state = incident_workflow.invoke(
            {
                "incident_description": case.incident_description,
                "scenario": case.initial_scenario,
                "post_remediation_scenario": case.post_remediation_scenario,
                "severity_assessment": None,
                "investigation": None,
                "investigation_error": None,
                "root_cause_analysis": None,
                "root_cause_analysis_error": None,
                "remediation_plan": None,
                "remediation_plan_error": None,
                "guarded_plan": None,
                "remediation_execution": None,
                "verification": None,
                "memory_recorded": False,
                "next_step": None,
            }
        )

    # Extract observed outcomes
    next_step = final_state.get("next_step")
    if next_step == "monitor":
        act_gate_a = GateADecision.NORMAL
        act_gate_b = GateBOutcome.MONITOR
    elif next_step == "resolve":
        act_gate_a = GateADecision.URGENT
        act_gate_b = GateBOutcome.RESOLVE
    elif next_step == "escalate":
        act_gate_a = GateADecision.URGENT
        act_gate_b = GateBOutcome.ESCALATE
    else:
        act_gate_a = None
        act_gate_b = None

    verification = final_state.get("verification")
    verification_status = verification.status if verification else None
    verification_ran = verification is not None and verification_status != VerificationStatus.SKIPPED

    remediation_execution = final_state.get("remediation_execution")
    executed_steps = (
        [
            res.step
            for res in remediation_execution.step_results
            if res.status == ExecutionStatus.EXECUTED
        ]
        if remediation_execution
        else []
    )

    guarded_plan = final_state.get("guarded_plan")
    step_dispositions = (
        {
            guarded.step.action_type.value: guarded.disposition
            for guarded in guarded_plan.steps
        }
        if guarded_plan
        else {}
    )

    unauthorized_executions = [
        step
        for step in executed_steps
        if step_dispositions.get(step.action_type.value)
        in (Disposition.BLOCKED, Disposition.REQUIRES_APPROVAL)
    ]

    severity = (
        final_state["severity_assessment"].severity
        if final_state.get("severity_assessment")
        else None
    )

    actual = ActualExecution(
        gate_a=act_gate_a,
        gate_b=act_gate_b,
        severity=severity,
        verification_status=verification_status,
        verification_ran=verification_ran,
        remediation_executed_count=spy_execute_guarded_plan.call_count,
        executed_steps=executed_steps,
        unauthorized_executions=unauthorized_executions,
        step_dispositions=step_dispositions,
    )

    case_result = CaseEvaluationResult(
        case_id=case.case_id,
        passed=True,
        expected=case.expected,
        actual=actual,
        violations=[],
    )

    # Invariant and expectation verification
    violations: list[str] = [
        str(v) for v in check_case_invariants(case_result)
    ]

    if act_gate_a != case.expected.gate_a:
        violations.append(
            f"Gate A mismatch: observed '{act_gate_a}', expected '{case.expected.gate_a}'."
        )

    if act_gate_b != case.expected.gate_b:
        violations.append(
            f"Gate B mismatch: observed '{act_gate_b}', expected '{case.expected.gate_b}'."
        )

    if verification_status != case.expected.verification_status:
        violations.append(
            f"Verification status mismatch: observed '{verification_status}', "
            f"expected '{case.expected.verification_status}'."
        )

    if case.expected.expected_dispositions:
        for action_name, expected_disp in case.expected.expected_dispositions.items():
            actual_disp = step_dispositions.get(action_name)
            if actual_disp != expected_disp:
                violations.append(
                    f"Disposition mismatch for '{action_name}': "
                    f"observed '{actual_disp}', expected '{expected_disp}'."
                )

    passed = (len(violations) == 0)

    return CaseEvaluationResult(
        case_id=case.case_id,
        passed=passed,
        expected=case.expected,
        actual=actual,
        violations=violations,
    )


def run_evaluation(
    cases: Sequence[EvalCase] = GOLDEN_DATASET,
) -> EvaluationReport:
    """Run evaluation across a sequence of evaluation cases (defaults to GOLDEN_DATASET).

    Produces structured per-case results and aggregate EvaluationMetrics.
    Overall pass/fail requires all cases to pass and all invariants to hold.
    """
    case_results = [run_case(case) for case in cases]
    metrics = compute_evaluation_metrics(case_results)
    all_violations = check_all_invariants(case_results)

    passed = (
        metrics.passed_cases == len(cases)
        and metrics.all_invariants_passed
        and metrics.unauthorized_execution_count == 0
        and metrics.repeat_remediation_violation_count == 0
        and metrics.invalid_resolution_count == 0
    )

    return EvaluationReport(
        passed=passed,
        metrics=metrics,
        case_results=case_results,
        violations=all_violations,
    )


__all__ = [
    "run_case",
    "run_evaluation",
]
