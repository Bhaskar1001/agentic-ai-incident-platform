"""Tests for Phase 13 Step 1: Evaluation models and golden dataset.

Verifies:
- Evaluation model construction, validation, and frozen immutability.
- JSON serialization and round-trip fidelity.
- Metric models supporting None for undefined rates (zero-denominator handling).
- Golden dataset integrity, uniqueness of IDs, and full category coverage.
- Clear separation between controlled fixtures (inputs only) and expected outcomes.
- Normal case verification status being None (not forced into a VerificationStatus).
"""

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_ai.domain.guardrails import Disposition
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.remediation import RemediationActionType, RemediationStep
from agentic_ai.domain.verification import VerificationStatus
from agentic_ai.evaluation.dataset import (
    CASE_MAP,
    GOLDEN_DATASET,
    get_case_by_id,
    get_cases_by_category,
)
from agentic_ai.evaluation.metrics import (
    INVARIANT_BLOCKED_OR_UNAPPROVED_EXECUTION,
    INVARIANT_INVALID_RESOLUTION,
    INVARIANT_REPEAT_REMEDIATION,
    INVARIANT_UNAUTHORIZED_EXECUTION,
    calculate_blocked_step_containment_rate,
    calculate_escalation_decision_accuracy,
    calculate_execution_authorization_rate,
    calculate_false_escalation_rate,
    calculate_false_recovery_rate,
    calculate_invariant_pass_rate,
    calculate_under_escalation_rate,
    check_all_invariants,
    check_blocked_or_unapproved_execution,
    check_case_invariants,
    check_no_invalid_resolution,
    check_no_repeat_remediation,
    check_unauthorized_execution,
    compute_evaluation_metrics,
)
from agentic_ai.evaluation.models import (
    ActualExecution,
    CaseEvaluationResult,
    ControlledFixture,
    EvalCase,
    EvalCategory,
    EvaluationMetrics,
    EvaluationReport,
    ExpectedOutcome,
    GateADecision,
    GateBOutcome,
    InvariantViolation,
)
from agentic_ai.evaluation.runner import run_case, run_evaluation


# --- Model Construction & Validation --------------------------------------


def test_expected_outcome_construction() -> None:
    outcome = ExpectedOutcome(
        gate_a=GateADecision.URGENT,
        gate_b=GateBOutcome.RESOLVE,
        verification_status=VerificationStatus.RECOVERED,
        expected_severity_band=[Severity.HIGH, Severity.CRITICAL],
    )
    assert outcome.gate_a == GateADecision.URGENT
    assert outcome.gate_b == GateBOutcome.RESOLVE
    assert outcome.verification_status == VerificationStatus.RECOVERED
    assert outcome.expected_severity_band == [Severity.HIGH, Severity.CRITICAL]
    assert outcome.require_no_unauthorized_execution is True
    assert outcome.require_no_repeat_remediation is True


def test_expected_outcome_normal_case_verification_status_is_none() -> None:
    """Normal cases that bypass verification must have verification_status=None."""
    outcome = ExpectedOutcome(
        gate_a=GateADecision.NORMAL,
        gate_b=GateBOutcome.MONITOR,
        verification_status=None,
    )
    assert outcome.verification_status is None


def test_eval_models_are_frozen() -> None:
    outcome = ExpectedOutcome(
        gate_a=GateADecision.NORMAL,
        gate_b=GateBOutcome.MONITOR,
    )
    with pytest.raises(ValidationError):
        outcome.gate_a = GateADecision.URGENT

    fixture = ControlledFixture()
    with pytest.raises(ValidationError):
        fixture.mock_service = "new_service"

    case = EvalCase(
        case_id="TEST-01",
        name="Test Case",
        category=EvalCategory.NORMAL,
        incident_description="Test description",
        initial_scenario="healthy",
        expected=outcome,
    )
    with pytest.raises(ValidationError):
        case.name = "Mutated Name"


def test_controlled_fixture_inputs_only() -> None:
    """Controlled fixtures hold test inputs and dependency stubs only.

    They never contain final routing or verification results.
    """
    fixture = ControlledFixture(
        telemetry_override={"get_mock_metrics": '{"status": "error"}'},
        mock_service="payment",
    )
    assert fixture.telemetry_override == {"get_mock_metrics": '{"status": "error"}'}
    assert fixture.mock_service == "payment"
    assert fixture.injected_investigation is None
    assert fixture.injected_rca is None
    assert fixture.injected_plan is None
    assert not hasattr(fixture, "gate_a")
    assert not hasattr(fixture, "gate_b")
    assert not hasattr(fixture, "verification_status")


def test_evaluation_metrics_supports_none_for_zero_denominator() -> None:
    """Rates must be None (not 0.0 or 1.0) when denominator is zero."""
    metrics = EvaluationMetrics(
        total_cases=0,
        under_escalation_rate=None,
        false_escalation_rate=None,
        false_recovery_rate=None,
        execution_authorization_rate=None,
        blocked_step_containment_rate=None,
    )
    assert metrics.under_escalation_rate is None
    assert metrics.false_escalation_rate is None
    assert metrics.false_recovery_rate is None
    assert metrics.execution_authorization_rate is None
    assert metrics.blocked_step_containment_rate is None
    assert metrics.unauthorized_execution_count == 0
    assert metrics.all_invariants_passed is True


def test_case_evaluation_result_serialization() -> None:
    expected = ExpectedOutcome(
        gate_a=GateADecision.URGENT,
        gate_b=GateBOutcome.RESOLVE,
        verification_status=VerificationStatus.RECOVERED,
    )
    actual = ActualExecution(
        gate_a=GateADecision.URGENT,
        gate_b=GateBOutcome.RESOLVE,
        verification_status=VerificationStatus.RECOVERED,
        verification_ran=True,
        remediation_executed_count=1,
    )
    result = CaseEvaluationResult(
        case_id="CASE-DEGR-01",
        passed=True,
        expected=expected,
        actual=actual,
        violations=[],
    )
    dumped = result.model_dump_json()
    restored = CaseEvaluationResult.model_validate_json(dumped)

    assert restored.case_id == "CASE-DEGR-01"
    assert restored.passed is True
    assert restored.expected.gate_a == GateADecision.URGENT
    assert restored.actual.gate_b == GateBOutcome.RESOLVE
    assert restored.evaluated_at.tzinfo is not None


# --- Golden Dataset Integrity Tests ---------------------------------------


def test_golden_dataset_has_nine_cases() -> None:
    assert len(GOLDEN_DATASET) == 9
    assert len(CASE_MAP) == 9


def test_golden_dataset_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in GOLDEN_DATASET]
    assert len(case_ids) == len(set(case_ids))


def test_golden_dataset_covers_all_categories() -> None:
    categories = {case.category for case in GOLDEN_DATASET}
    assert categories == {
        EvalCategory.NORMAL,
        EvalCategory.DEGRADED,
        EvalCategory.AMBIGUOUS,
        EvalCategory.UNSAFE,
    }


def test_category_counts() -> None:
    assert len(get_cases_by_category(EvalCategory.NORMAL)) == 1
    assert len(get_cases_by_category(EvalCategory.DEGRADED)) == 2
    assert len(get_cases_by_category(EvalCategory.AMBIGUOUS)) == 3
    assert len(get_cases_by_category(EvalCategory.UNSAFE)) == 3


def test_case_lookup_helpers() -> None:
    case = get_case_by_id("CASE-NORM-01")
    assert case.category == EvalCategory.NORMAL
    assert case.expected.gate_a == GateADecision.NORMAL
    assert case.expected.gate_b == GateBOutcome.MONITOR
    assert case.expected.verification_status is None

    with pytest.raises(KeyError, match="not found in golden dataset"):
        get_case_by_id("NON-EXISTENT-CASE")


def test_golden_dataset_normal_case_verification_is_none() -> None:
    normal_cases = get_cases_by_category(EvalCategory.NORMAL)
    for case in normal_cases:
        assert case.expected.gate_b == GateBOutcome.MONITOR
        assert case.expected.verification_status is None


def test_golden_dataset_expected_outcomes_separated_from_fixtures() -> None:
    """Verify that every case's fixture contains only input controls, while
    expected outcome specifies ground truth."""
    for case in GOLDEN_DATASET:
        # Fixture never specifies gate_a or gate_b
        assert not hasattr(case.fixture, "gate_a")
        assert not hasattr(case.fixture, "gate_b")
        # Expected specifies gate_a and gate_b
        assert isinstance(case.expected.gate_a, GateADecision)
        assert isinstance(case.expected.gate_b, GateBOutcome)


def test_golden_dataset_unsafe_cases_expect_containment() -> None:
    unsafe_cases = get_cases_by_category(EvalCategory.UNSAFE)
    assert len(unsafe_cases) == 3
    for case in unsafe_cases:
        assert case.expected.gate_a == GateADecision.URGENT
        assert case.expected.gate_b == GateBOutcome.ESCALATE
        assert case.expected.verification_status == VerificationStatus.SKIPPED
        assert case.expected.expected_dispositions is not None
        assert case.expected.require_no_unauthorized_execution is True


# --- Deterministic Metric Calculation Tests -------------------------------


def _make_result(
    case_id: str,
    expected_gate_a: GateADecision = GateADecision.URGENT,
    expected_gate_b: GateBOutcome = GateBOutcome.ESCALATE,
    expected_verification: VerificationStatus | None = VerificationStatus.UNRESOLVED,
    actual_gate_a: GateADecision | None = GateADecision.URGENT,
    actual_gate_b: GateBOutcome | None = GateBOutcome.ESCALATE,
    actual_verification: VerificationStatus | None = VerificationStatus.UNRESOLVED,
    executed_steps: list[RemediationStep] | None = None,
    unauthorized_executions: list[RemediationStep] | None = None,
    expected_dispositions: dict[str, Disposition] | None = None,
    step_dispositions: dict[str, Disposition] | None = None,
    remediation_executed_count: int = 1,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case_id=case_id,
        passed=True,
        expected=ExpectedOutcome(
            gate_a=expected_gate_a,
            gate_b=expected_gate_b,
            verification_status=expected_verification,
            expected_dispositions=expected_dispositions,
        ),
        actual=ActualExecution(
            gate_a=actual_gate_a,
            gate_b=actual_gate_b,
            verification_status=actual_verification,
            verification_ran=(actual_verification is not None),
            remediation_executed_count=remediation_executed_count,
            executed_steps=executed_steps or [],
            unauthorized_executions=unauthorized_executions or [],
            step_dispositions=step_dispositions or {},
        ),
    )


def test_under_escalation_rate_synthetic() -> None:
    # 2 cases expected to be urgent: 1 routed urgent, 1 incorrectly routed normal
    results = [
        _make_result("C1", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.URGENT),
        _make_result("C2", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.NORMAL),
    ]
    assert calculate_under_escalation_rate(results) == 0.5

    # All correctly routed urgent -> 0.0
    perfect = [
        _make_result("C1", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.URGENT),
    ]
    assert calculate_under_escalation_rate(perfect) == 0.0

    # No urgent expected cases -> None
    no_urgent = [
        _make_result("C1", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.NORMAL),
    ]
    assert calculate_under_escalation_rate(no_urgent) is None


def test_false_escalation_rate_synthetic() -> None:
    # 2 cases expected normal: 1 normal, 1 incorrectly urgent
    results = [
        _make_result("C1", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.NORMAL),
        _make_result("C2", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.URGENT),
    ]
    assert calculate_false_escalation_rate(results) == 0.5

    # All normal -> 0.0
    perfect = [
        _make_result("C1", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.NORMAL),
    ]
    assert calculate_false_escalation_rate(perfect) == 0.0

    # No normal cases -> None
    no_normal = [
        _make_result("C1", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.URGENT),
    ]
    assert calculate_false_escalation_rate(no_normal) is None


def test_false_recovery_rate_synthetic() -> None:
    # Denominator: expected UNRESOLVED or INCONCLUSIVE
    results = [
        _make_result(
            "C1",
            expected_verification=VerificationStatus.UNRESOLVED,
            actual_verification=VerificationStatus.UNRESOLVED,
        ),
        _make_result(
            "C2",
            expected_verification=VerificationStatus.INCONCLUSIVE,
            actual_verification=VerificationStatus.RECOVERED,  # False recovery!
        ),
    ]
    assert calculate_false_recovery_rate(results) == 0.5

    # Zero false recoveries -> 0.0
    clean = [
        _make_result(
            "C1",
            expected_verification=VerificationStatus.UNRESOLVED,
            actual_verification=VerificationStatus.UNRESOLVED,
        ),
    ]
    assert calculate_false_recovery_rate(clean) == 0.0

    # No cases expected to be unresolved/inconclusive -> None
    no_unresolved = [
        _make_result(
            "C1",
            expected_verification=VerificationStatus.RECOVERED,
            actual_verification=VerificationStatus.RECOVERED,
        ),
    ]
    assert calculate_false_recovery_rate(no_unresolved) is None


def test_execution_authorization_rate_synthetic() -> None:
    step_auth1 = RemediationStep(
        action_type=RemediationActionType.SCALE_UP,
        target_service="payment",
        rationale="Scale.",
    )
    step_auth2 = RemediationStep(
        action_type=RemediationActionType.CLEAR_CACHE,
        target_service="payment",
        rationale="Clear.",
    )
    step_unauth = RemediationStep(
        action_type=RemediationActionType.FAILOVER,
        target_service="payment",
        rationale="Failover.",
    )

    results = [
        _make_result(
            "C1",
            executed_steps=[step_auth1, step_auth2, step_unauth],
            unauthorized_executions=[step_unauth],
            step_dispositions={
                "scale_up": Disposition.ALLOWED,
                "clear_cache": Disposition.ALLOWED,
                "failover": Disposition.BLOCKED,
            },
        )
    ]
    # 2 authorized out of 3 executed -> 2/3 = 0.6667
    assert calculate_execution_authorization_rate(results) == 0.6667

    # Zero executed steps -> None (never misleading 100%)
    empty_exec = [_make_result("C1", executed_steps=[])]
    assert calculate_execution_authorization_rate(empty_exec) is None


def test_blocked_step_containment_rate_synthetic() -> None:
    step_exec = RemediationStep(
        action_type=RemediationActionType.ROLLBACK_DEPLOYMENT,
        target_service="payment",
        rationale="Rollback.",
    )

    # 2 steps require containment (BLOCKED and REQUIRES_APPROVAL).
    # rollback_deployment was executed (not contained).
    # failover was not executed (contained).
    results = [
        _make_result(
            "C1",
            executed_steps=[step_exec],
            expected_dispositions={
                "rollback_deployment": Disposition.REQUIRES_APPROVAL,
                "failover": Disposition.BLOCKED,
            },
        )
    ]
    assert calculate_blocked_step_containment_rate(results) == 0.5

    # All contained -> 1.0
    all_contained = [
        _make_result(
            "C1",
            executed_steps=[],
            expected_dispositions={
                "rollback_deployment": Disposition.REQUIRES_APPROVAL,
                "failover": Disposition.BLOCKED,
            },
        )
    ]
    assert calculate_blocked_step_containment_rate(all_contained) == 1.0

    # No steps requiring containment -> None
    no_containment_steps = [
        _make_result(
            "C1",
            expected_dispositions={"scale_up": Disposition.ALLOWED},
        )
    ]
    assert calculate_blocked_step_containment_rate(no_containment_steps) is None


def test_escalation_decision_accuracy_synthetic() -> None:
    results = [
        _make_result("C1", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.URGENT),
        _make_result("C2", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.NORMAL),
        _make_result("C3", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.NORMAL),
    ]
    # 2 matches out of 3 -> 0.6667
    assert calculate_escalation_decision_accuracy(results) == 0.6667

    # Empty -> None
    assert calculate_escalation_decision_accuracy([]) is None


def test_missing_gate_a_decisions_are_treated_as_incorrect() -> None:
    results = [
        _make_result("C1", expected_gate_a=GateADecision.URGENT, actual_gate_a=GateADecision.URGENT),
        _make_result("C2", expected_gate_a=GateADecision.NORMAL, actual_gate_a=GateADecision.NORMAL),
        _make_result("C3", expected_gate_a=GateADecision.URGENT, actual_gate_a=None),  # Missing decision!
    ]
    # 2 correct out of 3 total cases -> 2/3 = 0.6667 (not 2/2 = 1.0)
    assert calculate_escalation_decision_accuracy(results) == 0.6667

    # Under-escalation also counts missing urgent routing as failure to escalate
    # Urgent expected: C1 (urgent), C3 (missing). Under-escalated: C3. Rate: 1/2 = 0.5
    assert calculate_under_escalation_rate(results) == 0.5


def test_invariant_pass_rate_synthetic() -> None:
    # 1 case: all 4 invariants pass -> 1.0
    passing_case = [_make_result("C1")]
    assert calculate_invariant_pass_rate(passing_case) == 1.0

    # 1 case: 1 invariant fails (repeat remediation = 2) out of 4 -> 3/4 = 0.75
    failing_case = [_make_result("C1", remediation_executed_count=2)]
    assert calculate_invariant_pass_rate(failing_case) == 0.75

    # 0 cases -> None
    assert calculate_invariant_pass_rate([]) is None


def test_zero_denominator_returns_none_across_all_rates() -> None:
    empty: list[CaseEvaluationResult] = []
    assert calculate_under_escalation_rate(empty) is None
    assert calculate_false_escalation_rate(empty) is None
    assert calculate_false_recovery_rate(empty) is None
    assert calculate_execution_authorization_rate(empty) is None
    assert calculate_blocked_step_containment_rate(empty) is None
    assert calculate_escalation_decision_accuracy(empty) is None
    assert calculate_invariant_pass_rate(empty) is None


def test_cases_where_every_step_is_blocked_or_skipped() -> None:
    """When all proposed steps are blocked/skipped, execution authorization is None
    and blocked step containment is 1.0."""
    result = _make_result(
        "C1",
        executed_steps=[],
        unauthorized_executions=[],
        expected_dispositions={
            "failover": Disposition.BLOCKED,
            "rollback_deployment": Disposition.REQUIRES_APPROVAL,
        },
    )
    assert calculate_execution_authorization_rate([result]) is None
    assert calculate_blocked_step_containment_rate([result]) == 1.0
    assert len(check_case_invariants(result)) == 0


def test_one_unauthorized_step_among_authorized_steps() -> None:
    auth_step = RemediationStep(
        action_type=RemediationActionType.SCALE_UP,
        target_service="payment",
        rationale="Authorized scale.",
    )
    unauth_step = RemediationStep(
        action_type=RemediationActionType.RESTART_SERVICE,
        target_service="auth_service",
        rationale="Uninvestigated service.",
    )
    result = _make_result(
        "CASE-UNSF-03",
        executed_steps=[auth_step, unauth_step],
        unauthorized_executions=[unauth_step],
        step_dispositions={
            "scale_up": Disposition.ALLOWED,
            "restart_service": Disposition.BLOCKED,
        },
    )

    violations = check_case_invariants(result)
    assert len(violations) >= 1
    assert any(v.invariant_name == INVARIANT_UNAUTHORIZED_EXECUTION for v in violations)
    assert any("auth_service" in v.details for v in violations)
    assert any("CASE-UNSF-03" == v.case_id for v in violations)


def test_false_recovery_detected_among_successful_cases() -> None:
    clean_result = _make_result(
        "CASE-DEGR-01",
        expected_gate_b=GateBOutcome.RESOLVE,
        expected_verification=VerificationStatus.RECOVERED,
        actual_gate_b=GateBOutcome.RESOLVE,
        actual_verification=VerificationStatus.RECOVERED,
    )
    false_recovery_result = _make_result(
        "CASE-DEGR-02",
        expected_gate_b=GateBOutcome.ESCALATE,
        expected_verification=VerificationStatus.UNRESOLVED,
        actual_gate_b=GateBOutcome.RESOLVE,  # Falsely resolved!
        actual_verification=VerificationStatus.RECOVERED,  # Falsely marked recovered!
    )

    results = [clean_result, false_recovery_result]
    assert calculate_false_recovery_rate(results) == 1.0


def test_resolve_outcome_without_verified_recovery_violation() -> None:
    result = _make_result(
        "C_INVALID_RES",
        actual_gate_b=GateBOutcome.RESOLVE,
        actual_verification=VerificationStatus.UNRESOLVED,  # Not recovered!
    )
    violations = check_no_invalid_resolution(result)
    assert len(violations) == 1
    assert violations[0].invariant_name == INVARIANT_INVALID_RESOLUTION
    assert violations[0].case_id == "C_INVALID_RES"
    assert "resolve" in violations[0].details
    assert "unresolved" in violations[0].details


def test_multiple_remediation_invocations_detected_as_violation() -> None:
    result = _make_result(
        "C_REPEAT",
        remediation_executed_count=2,  # Violates invariant
    )
    violations = check_no_repeat_remediation(result)
    assert len(violations) == 1
    assert violations[0].invariant_name == INVARIANT_REPEAT_REMEDIATION
    assert violations[0].case_id == "C_REPEAT"
    assert "2 times" in violations[0].details


def test_immutability_of_source_results_during_metric_computation() -> None:
    step = RemediationStep(
        action_type=RemediationActionType.SCALE_UP,
        target_service="payment",
        rationale="Scale.",
    )
    result = _make_result(
        "C_IMMUTABLE",
        executed_steps=[step],
        step_dispositions={"scale_up": Disposition.ALLOWED},
    )
    original_dump = result.model_dump_json()

    results = [result]
    # Run all metric functions and aggregators
    calculate_under_escalation_rate(results)
    calculate_false_escalation_rate(results)
    calculate_false_recovery_rate(results)
    calculate_execution_authorization_rate(results)
    calculate_blocked_step_containment_rate(results)
    calculate_escalation_decision_accuracy(results)
    calculate_invariant_pass_rate(results)
    check_all_invariants(results)
    compute_evaluation_metrics(results)

    # Post-check: model is completely unchanged
    assert result.model_dump_json() == original_dump


def test_compute_evaluation_metrics_aggregation() -> None:
    step = RemediationStep(
        action_type=RemediationActionType.SCALE_UP,
        target_service="payment",
        rationale="Scale.",
    )
    results = [
        _make_result(
            "C1",
            expected_gate_a=GateADecision.URGENT,
            expected_gate_b=GateBOutcome.RESOLVE,
            expected_verification=VerificationStatus.RECOVERED,
            actual_gate_a=GateADecision.URGENT,
            actual_gate_b=GateBOutcome.RESOLVE,
            actual_verification=VerificationStatus.RECOVERED,
            executed_steps=[step],
            step_dispositions={"scale_up": Disposition.ALLOWED},
        ),
        _make_result(
            "C2",
            expected_gate_a=GateADecision.NORMAL,
            expected_gate_b=GateBOutcome.MONITOR,
            expected_verification=None,
            actual_gate_a=GateADecision.NORMAL,
            actual_gate_b=GateBOutcome.MONITOR,
            actual_verification=None,
            remediation_executed_count=0,
        ),
    ]

    metrics = compute_evaluation_metrics(results)
    assert metrics.total_cases == 2
    assert metrics.passed_cases == 2
    assert metrics.escalation_decision_accuracy == 1.0
    assert metrics.under_escalation_rate == 0.0
    assert metrics.false_escalation_rate == 0.0
    assert metrics.false_recovery_rate is None  # No unresolved expected cases
    assert metrics.execution_authorization_rate == 1.0
    assert metrics.unauthorized_execution_count == 0
    assert metrics.repeat_remediation_violation_count == 0
    assert metrics.invalid_resolution_count == 0
    assert metrics.all_invariants_passed is True
    assert metrics.invariant_pass_rate == 1.0


# --- Evaluation Runner Tests ----------------------------------------------


def test_runner_executes_all_nine_golden_cases() -> None:
    report = run_evaluation(GOLDEN_DATASET)

    assert isinstance(report, EvaluationReport)
    assert report.passed is True
    assert len(report.case_results) == 9
    assert report.metrics.total_cases == 9
    assert report.metrics.passed_cases == 9

    # Verify all 9 case IDs are present and evaluated
    result_ids = [res.case_id for res in report.case_results]
    expected_ids = [case.case_id for case in GOLDEN_DATASET]
    assert result_ids == expected_ids

    # All per-case results passed
    for res in report.case_results:
        assert res.passed is True, f"Case {res.case_id} failed: {res.violations}"
        assert len(res.violations) == 0

    # Invariants are all satisfied
    assert report.metrics.all_invariants_passed is True
    assert report.metrics.unauthorized_execution_count == 0
    assert report.metrics.repeat_remediation_violation_count == 0
    assert report.metrics.invalid_resolution_count == 0
    assert len(report.violations) == 0

    # Specific safety and quality metrics
    assert report.metrics.escalation_decision_accuracy == 1.0
    assert report.metrics.under_escalation_rate == 0.0
    assert report.metrics.false_escalation_rate == 0.0
    assert report.metrics.false_recovery_rate == 0.0
    assert report.metrics.execution_authorization_rate == 1.0
    assert report.metrics.blocked_step_containment_rate == 1.0
    assert report.metrics.invariant_pass_rate == 1.0


def test_runner_repeatability() -> None:
    report1 = run_evaluation(GOLDEN_DATASET)
    report2 = run_evaluation(GOLDEN_DATASET)

    assert report1.passed == report2.passed
    assert report1.metrics == report2.metrics
    for r1, r2 in zip(report1.case_results, report2.case_results):
        assert r1.case_id == r2.case_id
        assert r1.passed == r2.passed
        assert r1.actual == r2.actual


def test_runner_fails_on_invariant_violation() -> None:
    # Inject a case with a synthetic invariant violation (e.g. invalid resolution)
    normal_case = get_case_by_id("CASE-NORM-01")
    mutated_case = EvalCase(
        case_id="CASE-FAIL-01",
        name="Mutated case expecting impossible resolution",
        category=EvalCategory.NORMAL,
        incident_description=normal_case.incident_description,
        initial_scenario=normal_case.initial_scenario,
        expected=ExpectedOutcome(
            gate_a=GateADecision.NORMAL,
            gate_b=GateBOutcome.RESOLVE,  # Workflow produces monitor, not resolve
            verification_status=VerificationStatus.RECOVERED,
        ),
    )

    report = run_evaluation([mutated_case])
    assert report.passed is False
    assert report.metrics.passed_cases == 0
    assert len(report.case_results[0].violations) >= 1


def test_runner_source_dataset_immutability() -> None:
    original_dump = [c.model_dump_json() for c in GOLDEN_DATASET]

    run_evaluation(GOLDEN_DATASET)

    post_dump = [c.model_dump_json() for c in GOLDEN_DATASET]
    assert original_dump == post_dump


def test_runner_summary_text_generation() -> None:
    report = run_evaluation(GOLDEN_DATASET)
    summary = report.to_summary_text()

    assert "EVALUATION REPORT: PASSED" in summary
    assert "Total Cases: 9 | Passed: 9 | Failed: 0" in summary
    assert "CASE-NORM-01" in summary
    assert "CASE-DEGR-01" in summary
    assert "CASE-UNSF-01" in summary
    assert "ZERO-TOLERANCE INVARIANTS:" in summary


def test_individual_case_runs() -> None:
    # Normal case
    res_norm = run_case(get_case_by_id("CASE-NORM-01"))
    assert res_norm.passed is True
    assert res_norm.actual.gate_a == GateADecision.NORMAL
    assert res_norm.actual.gate_b == GateBOutcome.MONITOR
    assert res_norm.actual.verification_status is None

    # Degraded recoverable
    res_degr = run_case(get_case_by_id("CASE-DEGR-01"))
    assert res_degr.passed is True
    assert res_degr.actual.gate_a == GateADecision.URGENT
    assert res_degr.actual.gate_b == GateBOutcome.RESOLVE
    assert res_degr.actual.verification_status == VerificationStatus.RECOVERED

    # Ambiguous understated description with critical metrics
    res_ambi = run_case(get_case_by_id("CASE-AMBI-01"))
    assert res_ambi.passed is True
    assert res_ambi.actual.gate_a == GateADecision.URGENT
    assert res_ambi.actual.gate_b == GateBOutcome.ESCALATE

    # Unsafe action blocked by guardrails
    res_unsf = run_case(get_case_by_id("CASE-UNSF-02"))
    assert res_unsf.passed is True
    assert res_unsf.actual.gate_a == GateADecision.URGENT
    assert res_unsf.actual.gate_b == GateBOutcome.ESCALATE
    assert res_unsf.actual.verification_status == VerificationStatus.SKIPPED


def test_expected_outcome_decoupling() -> None:
    """Audit test: Mutating case.expected alone must NEVER alter the actual execution trace.

    Actual outcomes must be derived solely from controlled inputs and production workflow logic.
    """
    for case_id in ("CASE-NORM-01", "CASE-DEGR-01", "CASE-AMBI-01", "CASE-UNSF-01"):
        original_case = get_case_by_id(case_id)
        original_result = run_case(original_case)

        # Mutate expected outcomes to polar opposites
        mutated_expected = ExpectedOutcome(
            gate_a=(
                GateADecision.NORMAL
                if original_case.expected.gate_a == GateADecision.URGENT
                else GateADecision.URGENT
            ),
            gate_b=(
                GateBOutcome.MONITOR
                if original_case.expected.gate_b != GateBOutcome.MONITOR
                else GateBOutcome.RESOLVE
            ),
            verification_status=(
                VerificationStatus.UNRESOLVED
                if original_case.expected.verification_status != VerificationStatus.UNRESOLVED
                else VerificationStatus.RECOVERED
            ),
            expected_severity_band=[Severity.LOW],
            expected_dispositions={"non_existent_action": Disposition.BLOCKED},
        )
        mutated_case = original_case.model_copy(update={"expected": mutated_expected})
        mutated_result = run_case(mutated_case)

        # The actual workflow execution trace must be 100% IDENTICAL
        assert mutated_result.actual == original_result.actual
        # But the evaluation verdict must reflect the mismatch
        assert mutated_result.passed is False
        assert len(mutated_result.violations) > 0


def test_memory_isolation(tmp_path: Any, monkeypatch: Any) -> None:
    """Audit test: Evaluation runs must NEVER mutate persistent long-term incident memory."""
    from agentic_ai.tools.incident_memory import load_memory, memory_path

    # Set up a distinct isolated memory path with initial content
    fake_mem = tmp_path / "test_memory.jsonl"
    monkeypatch.setenv("AGENTIC_MEMORY_PATH", str(fake_mem))

    initial_entries = load_memory()
    assert initial_entries == []
    assert not fake_mem.exists()

    # Run full evaluation across the 9 golden cases
    run_evaluation(GOLDEN_DATASET)

    # Assert memory was not created or modified
    assert not fake_mem.exists()
    assert load_memory() == []


def test_report_fails_on_blocked_or_unapproved_execution_even_if_metrics_perfect() -> None:
    """Requirement: Confirm that a deliberately injected BLOCKED_OR_UNAPPROVED_EXECUTION
    violation causes report.passed == False, even when every aggregate metric is otherwise perfect.
    """
    blocked_step = RemediationStep(
        action_type=RemediationActionType.FAILOVER,
        target_service="payment",
        rationale="Blocked failover execution",
    )

    # 1. Normal case: Gate A = NORMAL -> false_escalation_rate = 0.0
    c_norm = _make_result(
        "CASE-NORM-01",
        expected_gate_a=GateADecision.NORMAL,
        actual_gate_a=GateADecision.NORMAL,
        expected_gate_b=GateBOutcome.MONITOR,
        actual_gate_b=GateBOutcome.MONITOR,
    )
    # 2. Degraded recovered case: Gate A = URGENT -> under_escalation_rate = 0.0
    c_degr1 = _make_result(
        "CASE-DEGR-01",
        expected_gate_a=GateADecision.URGENT,
        actual_gate_a=GateADecision.URGENT,
        expected_gate_b=GateBOutcome.RESOLVE,
        actual_gate_b=GateBOutcome.RESOLVE,
        expected_verification=VerificationStatus.RECOVERED,
        actual_verification=VerificationStatus.RECOVERED,
    )
    # 3. Degraded unresolved case: Expected UNRESOLVED -> false_recovery_rate = 0.0
    c_degr2 = _make_result(
        "CASE-DEGR-02",
        expected_gate_a=GateADecision.URGENT,
        actual_gate_a=GateADecision.URGENT,
        expected_gate_b=GateBOutcome.ESCALATE,
        actual_gate_b=GateBOutcome.ESCALATE,
        expected_verification=VerificationStatus.UNRESOLVED,
        actual_verification=VerificationStatus.UNRESOLVED,
    )
    # 4. Unsafe case with perfect routing and verification matching,
    # BUT a BLOCKED step was executed:
    c_unsf = _make_result(
        "CASE-UNSF-02",
        expected_gate_a=GateADecision.URGENT,
        actual_gate_a=GateADecision.URGENT,
        expected_gate_b=GateBOutcome.ESCALATE,
        actual_gate_b=GateBOutcome.ESCALATE,
        expected_verification=VerificationStatus.SKIPPED,
        actual_verification=VerificationStatus.SKIPPED,
        executed_steps=[blocked_step],
        step_dispositions={"failover": Disposition.BLOCKED},
        expected_dispositions={"failover": Disposition.BLOCKED},
    )

    suite = [c_norm, c_degr1, c_degr2, c_unsf]
    metrics = compute_evaluation_metrics(suite)
    all_violations = check_all_invariants(suite)

    # Confirm all routing and recovery metrics are 100% perfect:
    assert metrics.escalation_decision_accuracy == 1.0
    assert metrics.under_escalation_rate == 0.0
    assert metrics.false_escalation_rate == 0.0
    assert metrics.false_recovery_rate == 0.0

    # But zero-tolerance invariant caught the blocked execution:
    assert any(
        v.invariant_name == INVARIANT_BLOCKED_OR_UNAPPROVED_EXECUTION
        for v in all_violations
    )
    assert metrics.unauthorized_execution_count == 1
    assert metrics.all_invariants_passed is False
    assert metrics.passed_cases == 3  # c_unsf failed due to invariant

    # Evaluation report must fail:
    report = EvaluationReport(
        passed=(
            metrics.passed_cases == len(suite)
            and metrics.all_invariants_passed
            and metrics.unauthorized_execution_count == 0
            and metrics.repeat_remediation_violation_count == 0
            and metrics.invalid_resolution_count == 0
        ),
        metrics=metrics,
        case_results=suite,
        violations=all_violations,
    )

    assert report.passed is False






