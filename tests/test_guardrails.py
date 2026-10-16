import pytest

from agentic_ai.domain.guardrails import (
    ApprovalState,
    Disposition,
    GuardrailError,
    evaluate_step,
    guard_plan,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationPlan, RemediationStep, RiskTier

SERVICES = {"payment"}


def _analysis(
    confidence: float = 0.9,
    consistency: KnowledgeBaseConsistency = KnowledgeBaseConsistency.SUPPORTED,
) -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="x",
        root_cause_confidence=confidence,
        knowledge_base_consistency=consistency,
    )


def _step(action: str, service: str = "payment") -> RemediationStep:
    return RemediationStep(action_type=action, target_service=service, rationale="r")


# --- tier policy -----------------------------------------------------------


def test_low_risk_step_is_allowed() -> None:
    guarded = evaluate_step(_step("scale_up"), _analysis(), SERVICES)
    assert guarded.disposition == Disposition.ALLOWED
    assert guarded.may_execute


def test_medium_risk_step_passing_validation_is_allowed() -> None:
    guarded = evaluate_step(_step("restart_service"), _analysis(), SERVICES)
    assert guarded.disposition == Disposition.ALLOWED
    assert guarded.effective_tier == RiskTier.MEDIUM


def test_medium_risk_step_failing_validation_needs_approval() -> None:
    guarded = evaluate_step(_step("restart_service"), _analysis(), set())
    assert guarded.disposition == Disposition.REQUIRES_APPROVAL
    assert guarded.approval == ApprovalState.PENDING
    assert not guarded.may_execute
    assert any("not examined" in r for r in guarded.reasons)


def test_high_risk_step_needs_approval_even_with_confident_analysis() -> None:
    guarded = evaluate_step(_step("rollback_deployment"), _analysis(0.99), SERVICES)
    assert guarded.disposition == Disposition.REQUIRES_APPROVAL
    assert not guarded.may_execute


# --- weak signals only raise risk ------------------------------------------


def test_low_confidence_raises_the_tier_by_one() -> None:
    guarded = evaluate_step(_step("scale_up"), _analysis(0.3), SERVICES)
    assert guarded.effective_tier == RiskTier.MEDIUM


def test_contradicted_precedent_raises_the_tier() -> None:
    analysis = _analysis(consistency=KnowledgeBaseConsistency.CONTRADICTED)
    guarded = evaluate_step(_step("restart_service"), analysis, SERVICES)
    assert guarded.effective_tier == RiskTier.HIGH
    assert guarded.disposition == Disposition.REQUIRES_APPROVAL


def test_high_risk_step_with_a_weak_analysis_is_blocked() -> None:
    guarded = evaluate_step(_step("failover"), _analysis(0.2), SERVICES)
    assert guarded.effective_tier == RiskTier.CRITICAL
    assert guarded.disposition == Disposition.BLOCKED
    assert not guarded.may_execute


def test_a_strong_analysis_never_lowers_the_tier() -> None:
    for action in ("scale_up", "restart_service", "rollback_deployment", "failover"):
        step = _step(action)
        guarded = evaluate_step(step, _analysis(1.0), SERVICES)
        assert guarded.effective_tier == step.risk_tier


def test_tier_is_capped_at_critical() -> None:
    analysis = _analysis(0.1, KnowledgeBaseConsistency.CONTRADICTED)
    guarded = evaluate_step(_step("failover"), analysis, SERVICES)
    assert guarded.effective_tier == RiskTier.CRITICAL


# --- approval gate ---------------------------------------------------------


def _pending_plan():
    plan = RemediationPlan(steps=[_step("scale_up"), _step("rollback_deployment")])
    return guard_plan(plan, _analysis(), SERVICES)


def test_approval_makes_a_pending_step_executable() -> None:
    plan = _pending_plan()
    assert plan.pending_approvals == [1]

    approved = plan.approve(1, "alice")

    assert approved.steps[1].may_execute
    assert approved.steps[1].decided_by == "alice"
    assert approved.pending_approvals == []


def test_approving_does_not_mutate_the_original_plan() -> None:
    plan = _pending_plan()
    plan.approve(1, "alice")
    assert not plan.steps[1].may_execute


def test_rejection_keeps_a_step_unexecutable() -> None:
    rejected = _pending_plan().reject(1, "alice")
    assert rejected.steps[1].approval == ApprovalState.REJECTED
    assert not rejected.steps[1].may_execute


def test_a_blocked_step_cannot_be_approved() -> None:
    plan = guard_plan(
        RemediationPlan(steps=[_step("failover")]), _analysis(0.2), SERVICES
    )
    with pytest.raises(GuardrailError, match="blocked"):
        plan.approve(0, "alice")


def test_an_allowed_step_has_nothing_to_approve() -> None:
    with pytest.raises(GuardrailError, match="not awaiting"):
        _pending_plan().approve(0, "alice")


def test_a_decision_cannot_be_made_twice() -> None:
    approved = _pending_plan().approve(1, "alice")
    with pytest.raises(GuardrailError, match="not awaiting"):
        approved.reject(1, "bob")


def test_an_approval_must_name_the_approver() -> None:
    with pytest.raises(GuardrailError, match="who"):
        _pending_plan().approve(1, "  ")


def test_an_out_of_range_index_is_rejected() -> None:
    with pytest.raises(GuardrailError, match="No step"):
        _pending_plan().approve(9, "alice")
