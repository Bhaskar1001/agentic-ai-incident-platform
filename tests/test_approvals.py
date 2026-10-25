"""Domain tests for persisted approvals: identity, transitions, and application."""

from datetime import datetime, timezone

import pytest

from agentic_ai.domain.approvals import (
    ApprovalDecisionError,
    ApprovalRecord,
    apply_persisted_decisions,
    check_decidable,
    decide,
    pending_requests_for,
    plan_fingerprint,
    step_fingerprint,
    validate_decision_inputs,
)
from agentic_ai.domain.guardrails import ApprovalState, Disposition
from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRemediationRecord
from tests.approval_helpers import HIGH_FAILOVER, HIGH_ROLLBACK, LOW, guarded

AT = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def _pending(*actions, incident_id="INC-1", confidence=0.9) -> list[ApprovalRecord]:
    return pending_requests_for(incident_id, guarded(*actions, confidence=confidence))


def _decided(record: ApprovalRecord, outcome: ApprovalState, who="alice") -> ApprovalRecord:
    return decide(record, outcome, who, at=AT)


# --- fingerprints -----------------------------------------------------------------


def test_step_fingerprint_depends_on_action_and_target_only() -> None:
    assert step_fingerprint("failover", "payment") == step_fingerprint("failover", "payment")
    assert step_fingerprint("failover", "payment") != step_fingerprint("failover", "other")
    assert step_fingerprint("failover", "payment") != step_fingerprint("scale_up", "payment")


def test_plan_fingerprint_is_order_and_content_sensitive() -> None:
    base = plan_fingerprint([LOW, HIGH_ROLLBACK])
    assert base == plan_fingerprint([LOW, HIGH_ROLLBACK])
    assert base != plan_fingerprint([HIGH_ROLLBACK, LOW])  # reordered
    assert base != plan_fingerprint([LOW])  # step removed
    assert base != plan_fingerprint([LOW, HIGH_ROLLBACK, HIGH_FAILOVER])  # step added
    assert base != plan_fingerprint([LOW, HIGH_FAILOVER])  # step replaced


def test_fingerprints_ignore_the_models_rationale_prose() -> None:
    """Re-planning to the same actions must reuse a decision, so prose is excluded."""
    a = pending_requests_for("INC-1", guarded(HIGH_ROLLBACK))[0]
    plan_b = guarded(HIGH_ROLLBACK)
    object.__setattr__(plan_b.steps[0].step, "rationale", "completely different wording")
    b = pending_requests_for("INC-1", plan_b)[0]
    assert (a.plan_fingerprint, a.step_fingerprint) == (b.plan_fingerprint, b.step_fingerprint)
    assert a.rationale != b.rationale


# --- the record itself ------------------------------------------------------------


def test_requests_exist_only_for_steps_that_need_a_human() -> None:
    # scale_up is ALLOWED; rollback needs approval; failover on a weak analysis is BLOCKED.
    plan = guarded(LOW, HIGH_ROLLBACK, confidence=0.9)
    requests = pending_requests_for("INC-1", plan)
    assert [r.step_index for r in requests] == [1]

    weak = guarded(HIGH_FAILOVER, confidence=0.2)
    assert weak.steps[0].disposition == Disposition.BLOCKED
    assert pending_requests_for("INC-1", weak) == []


def test_request_snapshots_what_the_approver_is_shown() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    assert request.status == ApprovalState.PENDING
    assert (request.action_type, request.target_service) == HIGH_ROLLBACK
    assert request.rationale == "because of rollback_deployment"
    assert request.decided_at is None and request.approver is None


def test_a_pending_record_cannot_carry_a_decision() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ValueError, match="pending"):
        ApprovalRecord.model_validate({**request.model_dump(), "approver": "alice"})


def test_a_decided_record_must_say_who_and_when() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ValueError, match="who"):
        ApprovalRecord.model_validate({**request.model_dump(), "status": ApprovalState.APPROVED})


def test_not_required_is_not_a_storable_status() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ValueError, match="storable"):
        ApprovalRecord.model_validate(
            {**request.model_dump(), "status": ApprovalState.NOT_REQUIRED}
        )


# --- the transition rule ------------------------------------------------------------


@pytest.mark.parametrize("outcome", [ApprovalState.APPROVED, ApprovalState.REJECTED])
def test_a_pending_approval_can_be_decided(outcome) -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    decided = decide(request, outcome, "  alice  ", "  looks fine  ", at=AT)
    assert decided.status == outcome
    assert decided.approver == "alice" and decided.decision_reason == "looks fine"
    assert decided.decided_at == AT
    assert request.status == ApprovalState.PENDING  # the original is not mutated


def test_a_blank_reason_is_stored_as_none() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    assert decide(request, ApprovalState.APPROVED, "a", "   ").decision_reason is None


@pytest.mark.parametrize("first", [ApprovalState.APPROVED, ApprovalState.REJECTED])
@pytest.mark.parametrize("second", [ApprovalState.APPROVED, ApprovalState.REJECTED])
def test_a_decision_is_never_overwritten_or_repeated(first, second) -> None:
    """All four pairs, including repeating the same decision: always an error."""
    (request,) = _pending(HIGH_ROLLBACK)
    done = _decided(request, first, "alice")

    with pytest.raises(ApprovalDecisionError, match="already"):
        decide(done, second, "bob")
    assert done.approver == "alice" and done.status == first


@pytest.mark.parametrize("bad", [ApprovalState.PENDING, ApprovalState.NOT_REQUIRED])
def test_only_approved_or_rejected_are_valid_outcomes(bad) -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ApprovalDecisionError):
        decide(request, bad, "alice")
    with pytest.raises(ValueError):
        validate_decision_inputs(bad, "alice")


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_decision_must_name_an_approver(blank) -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ApprovalDecisionError, match="who"):
        decide(request, ApprovalState.APPROVED, blank)
    with pytest.raises(ValueError):
        validate_decision_inputs(ApprovalState.APPROVED, blank)


# --- applying stored decisions to a fresh plan ---------------------------------------


def test_an_approved_step_becomes_executable() -> None:
    plan = guarded(HIGH_ROLLBACK)
    (request,) = pending_requests_for("INC-1", plan)

    applied = apply_persisted_decisions(plan, [_decided(request, ApprovalState.APPROVED)])

    assert applied.steps[0].may_execute
    assert applied.steps[0].decided_by == "alice"
    assert not plan.steps[0].may_execute  # the input plan is unchanged


def test_a_rejected_step_never_becomes_executable() -> None:
    plan = guarded(HIGH_ROLLBACK)
    (request,) = pending_requests_for("INC-1", plan)

    applied = apply_persisted_decisions(plan, [_decided(request, ApprovalState.REJECTED)])

    assert applied.steps[0].approval == ApprovalState.REJECTED
    assert not applied.steps[0].may_execute


def test_a_pending_approval_changes_nothing() -> None:
    plan = guarded(HIGH_ROLLBACK)
    pending = pending_requests_for("INC-1", plan)
    assert apply_persisted_decisions(plan, pending) == plan


def test_approving_one_step_does_not_approve_its_neighbour() -> None:
    plan = guarded(HIGH_ROLLBACK, HIGH_FAILOVER)
    first, second = pending_requests_for("INC-1", plan)

    applied = apply_persisted_decisions(plan, [_decided(first, ApprovalState.APPROVED)])

    assert applied.steps[0].may_execute and not applied.steps[1].may_execute
    assert applied.steps[1].approval == ApprovalState.PENDING


def test_an_approval_for_a_different_plan_is_ignored() -> None:
    old_plan = guarded(HIGH_ROLLBACK)
    (old,) = pending_requests_for("INC-1", old_plan)
    approved_for_old = _decided(old, ApprovalState.APPROVED)

    # The same step now sits in a different plan (a LOW step was added first).
    new_plan = guarded(LOW, HIGH_ROLLBACK)
    applied = apply_persisted_decisions(new_plan, [approved_for_old])

    assert applied == new_plan
    assert not applied.steps[1].may_execute


def test_an_approval_does_not_transfer_to_a_different_action() -> None:
    (rollback,) = pending_requests_for("INC-1", guarded(HIGH_ROLLBACK))
    approved = _decided(rollback, ApprovalState.APPROVED)

    failover_plan = guarded(HIGH_FAILOVER)
    assert apply_persisted_decisions(failover_plan, [approved]) == failover_plan


def test_an_approval_with_a_forged_step_fingerprint_is_ignored() -> None:
    plan = guarded(HIGH_ROLLBACK)
    (request,) = pending_requests_for("INC-1", plan)
    forged = ApprovalRecord.model_validate(
        {**_decided(request, ApprovalState.APPROVED).model_dump(), "step_fingerprint": "0" * 64}
    )
    assert apply_persisted_decisions(plan, [forged]) == plan


def test_approval_never_overrides_a_blocked_step() -> None:
    """HIGH on a weak analysis is CRITICAL, hence BLOCKED; no approval changes that."""
    strong = guarded(HIGH_FAILOVER, confidence=0.9)
    (request,) = pending_requests_for("INC-1", strong)
    approved = _decided(request, ApprovalState.APPROVED)

    weak = guarded(HIGH_FAILOVER, confidence=0.2)  # same plan fingerprint, now blocked
    assert weak.steps[0].disposition == Disposition.BLOCKED
    applied = apply_persisted_decisions(weak, [approved])

    assert applied == weak
    assert not applied.steps[0].may_execute


def test_allowed_steps_are_untouched() -> None:
    plan = guarded(LOW)
    assert plan.steps[0].may_execute
    assert apply_persisted_decisions(plan, []) == plan


# --- check_decidable ------------------------------------------------------------------


def _remediations_for(*actions, disposition="requires_approval") -> list[IncidentRemediationRecord]:
    return [
        IncidentRemediationRecord(
            step_order=i,
            action_type=a,
            target_service=t,
            disposition=disposition,
            execution_status="skipped",
        )
        for i, (a, t) in enumerate(actions)
    ]


def test_a_matching_open_pending_approval_is_decidable() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    check_decidable(
        request, IncidentLifecycleStatus.AWAITING_APPROVAL, _remediations_for(HIGH_ROLLBACK)
    )


def test_a_decided_approval_is_not_decidable() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    done = _decided(request, ApprovalState.APPROVED)
    with pytest.raises(ApprovalDecisionError, match="already approved by alice"):
        check_decidable(
            done, IncidentLifecycleStatus.AWAITING_APPROVAL, _remediations_for(HIGH_ROLLBACK)
        )


@pytest.mark.parametrize(
    "status",
    [
        IncidentLifecycleStatus.RESOLVED,
        IncidentLifecycleStatus.ESCALATED,
        IncidentLifecycleStatus.MONITORED,
        IncidentLifecycleStatus.CLOSED,
    ],
)
def test_a_closed_incident_accepts_no_decisions(status) -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ApprovalDecisionError, match="decisions are closed"):
        check_decidable(request, status, _remediations_for(HIGH_ROLLBACK))


def test_an_approval_for_a_replaced_plan_is_not_decidable() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    replaced = _remediations_for(HIGH_FAILOVER)
    with pytest.raises(ApprovalDecisionError, match="superseded"):
        check_decidable(request, IncidentLifecycleStatus.AWAITING_APPROVAL, replaced)


def test_a_step_that_no_longer_needs_approval_is_not_decidable() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    now_allowed = _remediations_for(HIGH_ROLLBACK, disposition="allowed")
    with pytest.raises(ApprovalDecisionError, match="superseded"):
        check_decidable(request, IncidentLifecycleStatus.AWAITING_APPROVAL, now_allowed)


def test_an_incident_with_no_remediations_has_nothing_to_approve() -> None:
    (request,) = _pending(HIGH_ROLLBACK)
    with pytest.raises(ApprovalDecisionError, match="superseded"):
        check_decidable(request, IncidentLifecycleStatus.AWAITING_APPROVAL, [])
