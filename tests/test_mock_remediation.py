"""Tests for Phase 11 Step 2: deterministic mock remediation executor.

Verifies:
- All eight action types execute in simulation mode.
- Outputs are deterministic and state they are simulated.
- Unknown services and invalid inputs fail gracefully.
- Blocked steps are marked BLOCKED and never reach the executor.
- Pending or rejected approvals are marked SKIPPED and never reach the executor.
- Approved steps execute properly.
- Prohibited steps never reach a mock handler.
"""

from unittest.mock import Mock

import pytest

from agentic_ai.domain.guardrails import (
    ApprovalState,
    Disposition,
    GuardedPlan,
    GuardedStep,
    evaluate_step,
    guard_plan,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationActionType,
    RemediationPlan,
    RemediationStep,
    RiskTier,
)
from agentic_ai.tools.mock_remediation import (
    execute_guarded_plan,
    execute_mock_remediation_step,
)

SERVICES = {"payment"}


def _analysis(
    confidence: float = 0.9,
    consistency: KnowledgeBaseConsistency = KnowledgeBaseConsistency.SUPPORTED,
) -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Database pool saturated.",
        root_cause_confidence=confidence,
        knowledge_base_consistency=consistency,
    )


def _step(
    action: RemediationActionType | str = RemediationActionType.SCALE_UP,
    service: str = "payment",
) -> RemediationStep:
    action_type = (
        action
        if isinstance(action, RemediationActionType)
        else RemediationActionType(action)
    )
    return RemediationStep(
        action_type=action_type,
        target_service=service,
        rationale="Simulated test step.",
    )


# --- All 8 Action Types -----------------------------------------------------


@pytest.mark.parametrize("action_type", list(RemediationActionType))
def test_all_eight_actions_execute_successfully(action_type: RemediationActionType) -> None:
    step = _step(action=action_type)
    result = execute_mock_remediation_step(step)

    assert result.status == ExecutionStatus.EXECUTED
    assert result.error is None
    assert "[SIMULATED]" in result.output
    assert "payment" in result.output


def test_outputs_are_deterministic() -> None:
    step = _step(action=RemediationActionType.RESTART_SERVICE)
    res1 = execute_mock_remediation_step(step)
    res2 = execute_mock_remediation_step(step)

    assert res1.output == res2.output
    assert res1.status == res2.status == ExecutionStatus.EXECUTED


# --- Invalid / Unsupported Inputs -------------------------------------------


def test_unknown_target_service_fails_gracefully() -> None:
    step = RemediationStep(
        action_type=RemediationActionType.SCALE_UP,
        target_service="unknown_service",
        rationale="test",
    )
    result = execute_mock_remediation_step(step)

    assert result.status == ExecutionStatus.FAILED
    assert result.error is not None
    assert "unknown service" in result.output.lower()


def test_unsupported_action_type_fails_gracefully() -> None:
    # Construct a step bypassing pydantic enum validation for mock robustness
    step = RemediationStep.model_construct(
        action_type="unsupported_action",
        target_service="payment",
        rationale="test",
    )
    result = execute_mock_remediation_step(step)

    assert result.status == ExecutionStatus.FAILED
    assert result.error is not None
    assert "unsupported action" in result.output.lower()


# --- Safety Boundaries and Approval Enforcement in execute_guarded_plan -----


def test_allowed_low_risk_step_executes_in_guarded_plan() -> None:
    plan = RemediationPlan(steps=[_step(RemediationActionType.SCALE_UP)])
    guarded = guard_plan(plan, _analysis(), SERVICES)

    execution = execute_guarded_plan(guarded)

    assert len(execution.step_results) == 1
    assert execution.step_results[0].status == ExecutionStatus.EXECUTED
    assert "[SIMULATED]" in execution.step_results[0].output


def test_blocked_step_is_never_executed() -> None:
    # High risk action with low confidence RCA becomes CRITICAL -> BLOCKED
    plan = RemediationPlan(steps=[_step(RemediationActionType.FAILOVER)])
    guarded = guard_plan(plan, _analysis(confidence=0.2), SERVICES)
    assert guarded.steps[0].disposition == Disposition.BLOCKED

    spy_executor = Mock()
    execution = execute_guarded_plan(guarded, step_executor=spy_executor)

    assert len(execution.step_results) == 1
    assert execution.step_results[0].status == ExecutionStatus.BLOCKED
    assert "[SIMULATED]" in execution.step_results[0].output
    assert "blocked" in execution.step_results[0].output.lower()
    spy_executor.assert_not_called()


def test_pending_approval_step_is_skipped_and_never_executed() -> None:
    # High risk action needs human approval (PENDING)
    plan = RemediationPlan(steps=[_step(RemediationActionType.ROLLBACK_DEPLOYMENT)])
    guarded = guard_plan(plan, _analysis(), SERVICES)
    assert guarded.steps[0].disposition == Disposition.REQUIRES_APPROVAL
    assert guarded.steps[0].approval == ApprovalState.PENDING

    spy_executor = Mock()
    execution = execute_guarded_plan(guarded, step_executor=spy_executor)

    assert len(execution.step_results) == 1
    assert execution.step_results[0].status == ExecutionStatus.SKIPPED
    assert "awaiting human approval" in execution.step_results[0].output
    spy_executor.assert_not_called()


def test_rejected_approval_step_is_skipped_and_never_executed() -> None:
    plan = RemediationPlan(steps=[_step(RemediationActionType.ROLLBACK_DEPLOYMENT)])
    guarded = guard_plan(plan, _analysis(), SERVICES)
    rejected = guarded.reject(0, "oncall_engineer")
    assert rejected.steps[0].approval == ApprovalState.REJECTED

    spy_executor = Mock()
    execution = execute_guarded_plan(rejected, step_executor=spy_executor)

    assert len(execution.step_results) == 1
    assert execution.step_results[0].status == ExecutionStatus.SKIPPED
    assert "rejected by oncall_engineer" in execution.step_results[0].output
    spy_executor.assert_not_called()


def test_approved_step_executes_successfully() -> None:
    plan = RemediationPlan(steps=[_step(RemediationActionType.ROLLBACK_DEPLOYMENT)])
    guarded = guard_plan(plan, _analysis(), SERVICES)
    approved = guarded.approve(0, "lead_sre")
    assert approved.steps[0].approval == ApprovalState.APPROVED
    assert approved.steps[0].may_execute

    execution = execute_guarded_plan(approved)

    assert len(execution.step_results) == 1
    assert execution.step_results[0].status == ExecutionStatus.EXECUTED
    assert "rolled back" in execution.step_results[0].output.lower()


def test_prohibited_steps_never_reach_mock_handler_in_mixed_plan() -> None:
    # Plan with 4 steps:
    # 0: SCALE_UP (ALLOWED)
    # 1: ROLLBACK_DEPLOYMENT (REQUIRES_APPROVAL -> PENDING)
    # 2: FAILOVER with low confidence RCA (BLOCKED)
    # 3: ROTATE_CERTIFICATE (REQUIRES_APPROVAL -> REJECTED)
    step_allowed = _step(RemediationActionType.SCALE_UP)
    step_pending = _step(RemediationActionType.ROLLBACK_DEPLOYMENT)
    step_blocked = _step(RemediationActionType.FAILOVER)
    step_rejected = _step(RemediationActionType.ROTATE_CERTIFICATE)

    guarded_allowed = evaluate_step(step_allowed, _analysis(1.0), SERVICES)
    guarded_pending = evaluate_step(step_pending, _analysis(1.0), SERVICES)
    guarded_blocked = evaluate_step(step_blocked, _analysis(0.1), SERVICES)
    guarded_to_reject = evaluate_step(step_rejected, _analysis(1.0), SERVICES)

    plan = GuardedPlan(
        steps=[guarded_allowed, guarded_pending, guarded_blocked, guarded_to_reject]
    )
    plan_with_rejection = plan.reject(3, "security_team")

    mock_handler = Mock(wraps=execute_mock_remediation_step)
    execution = execute_guarded_plan(plan_with_rejection, step_executor=mock_handler)

    assert len(execution.step_results) == 4
    assert execution.step_results[0].status == ExecutionStatus.EXECUTED
    assert execution.step_results[1].status == ExecutionStatus.SKIPPED
    assert execution.step_results[2].status == ExecutionStatus.BLOCKED
    assert execution.step_results[3].status == ExecutionStatus.SKIPPED

    # Crucial assertion: the handler was called EXACTLY ONCE, for the allowed step only.
    assert mock_handler.call_count == 1
    mock_handler.assert_called_once_with(step_allowed)


def test_requires_approval_step_not_executed_if_approval_inconsistent() -> None:
    # Test safeguard: even if may_execute were somehow True, if approval is not APPROVED,
    # the step must not be executed.
    step = _step(RemediationActionType.ROLLBACK_DEPLOYMENT)
    # Fabricate an inconsistent state for boundary testing
    inconsistent_step = GuardedStep.model_construct(
        step=step,
        effective_tier=RiskTier.HIGH,
        disposition=Disposition.REQUIRES_APPROVAL,
        reasons=["test"],
        approval=ApprovalState.NOT_REQUIRED,  # Not APPROVED
    )
    plan = GuardedPlan(steps=[inconsistent_step])

    spy_executor = Mock()
    execution = execute_guarded_plan(plan, step_executor=spy_executor)

    assert execution.step_results[0].status == ExecutionStatus.SKIPPED
    spy_executor.assert_not_called()

