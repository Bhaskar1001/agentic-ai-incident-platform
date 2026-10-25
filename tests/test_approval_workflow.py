"""Workflow integration: persisted approvals at the execution boundary.

Each test drives the real sequence (submit, decide, resubmit) through the real planner,
guardrails, executor and verification, against every available backend. A "run" is
``handle_incident`` followed by ``persist_workflow_run``, exactly what ``POST /incidents``
does. Only the LLM-backed steps are controlled (see ``approval_helpers``).

Success criterion under test: a high-risk step cannot execute until its exact persisted
approval is APPROVED; rejected and pending decisions are enforced deterministically.
"""

import pytest

from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentLifecycleStatus
from agentic_ai.domain.remediation import ExecutionStatus
from agentic_ai.domain.verification import VerificationStatus
from agentic_ai.persistence.repository import ApprovalConflictError
from agentic_ai.persistence.runs import persist_workflow_run
from agentic_ai.tools.mock_tools import Scenario
from agentic_ai.workflows.incident import handle_incident
from tests.approval_helpers import (
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    LOW,
    install_pipeline_stubs,
    new_incident_id,
)

A, R = ApprovalState.APPROVED, ApprovalState.REJECTED
HEALTHY = Scenario.HEALTHY.value
STILL_BROKEN = Scenario.CONNECTION_POOL_EXHAUSTION.value


@pytest.fixture
def pipeline(monkeypatch):
    return install_pipeline_stubs(monkeypatch)


def run(store, incident_id, *, post=HEALTHY, lookup=None):
    """One submission: the workflow (reading approvals), then persistence."""
    state = handle_incident(
        "Payment service is degraded",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=post,
        incident_id=incident_id,
        approval_repository=lookup if lookup is not None else store,
    )
    return state, persist_workflow_run(store, state)


def statuses(state) -> list[ExecutionStatus]:
    return [r.status for r in state["remediation_execution"].step_results]


def stored_status(store, incident_id) -> IncidentLifecycleStatus:
    return store.get_incident(incident_id).status


def decide(store, incident_id, index, outcome, who="alice"):
    approval = store.list_approvals(incident_id)[index]
    return store.decide_approval(incident_id, approval.approval_id, outcome, who)


# --- a high-risk step waits ------------------------------------------------------------


def test_a_high_risk_step_is_persisted_pending_and_does_not_execute(incident_store, pipeline):
    incident_id = new_incident_id()

    state, persisted = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.SKIPPED]
    assert "awaiting human approval" in state["remediation_execution"].step_results[0].output
    assert state["verification"].status == VerificationStatus.SKIPPED
    assert state["next_step"] == "escalate"

    assert persisted.persisted and len(persisted.pending_approval_ids) == 1
    (approval,) = incident_store.list_approvals(incident_id)
    assert approval.status == ApprovalState.PENDING
    assert approval.approval_id == persisted.pending_approval_ids[0]
    assert (approval.action_type, approval.target_service) == HIGH_ROLLBACK
    assert approval.step_index == 0 and approval.rationale
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.AWAITING_APPROVAL


def test_resubmitting_without_a_decision_still_does_not_execute(incident_store, pipeline):
    incident_id = new_incident_id()
    _, first = run(incident_store, incident_id)

    state, second = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.SKIPPED]
    assert second.pending_approval_ids == first.pending_approval_ids  # same approval, not a new one
    assert len(incident_store.list_approvals(incident_id)) == 1
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.AWAITING_APPROVAL


# --- approve ------------------------------------------------------------------------------


def test_deciding_alone_executes_nothing_and_resolves_nothing(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)

    decide(incident_store, incident_id, 0, A)

    record = incident_store.get_incident(incident_id)
    assert record.status == IncidentLifecycleStatus.AWAITING_APPROVAL
    assert [r.execution_status for r in record.remediations] == ["skipped"]


def test_an_approved_step_executes_on_the_next_submission_and_is_then_verified(
    incident_store, pipeline
):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A, "alice")

    state, persisted = run(incident_store, incident_id, post=HEALTHY)

    assert statuses(state) == [ExecutionStatus.EXECUTED]
    assert state["verification"].status == VerificationStatus.RECOVERED
    assert state["next_step"] == "resolve"
    assert persisted.persisted and persisted.pending_approval_ids == []
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.RESOLVED
    # The audit trail survives the execution it authorized.
    (approval,) = incident_store.list_approvals(incident_id)
    assert approval.status == A and approval.approver == "alice"
    assert incident_store.get_incident(incident_id).remediations[0].execution_status == "executed"


def test_approval_does_not_imply_recovery(incident_store, pipeline):
    """Executing an approved step still has to pass deterministic verification."""
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A)

    state, _ = run(incident_store, incident_id, post=STILL_BROKEN)

    assert statuses(state) == [ExecutionStatus.EXECUTED]
    assert state["verification"].status == VerificationStatus.UNRESOLVED
    assert state["next_step"] == "escalate"
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.ESCALATED


# --- reject -----------------------------------------------------------------------------------


def test_a_rejected_step_never_executes(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, R, "bob")

    state, persisted = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.SKIPPED]
    assert "rejected by bob" in state["remediation_execution"].step_results[0].output
    assert state["next_step"] == "escalate"
    assert persisted.pending_approval_ids == []
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.ESCALATED
    assert incident_store.list_approvals(incident_id)[0].status == R


def test_a_rejection_survives_any_number_of_resubmissions(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, R)

    for _ in range(3):
        state, _ = run(incident_store, incident_id)
        assert statuses(state) == [ExecutionStatus.SKIPPED]
    assert len(incident_store.list_approvals(incident_id)) == 1


# --- approvals are exact ---------------------------------------------------------------------------


def test_approving_one_step_does_not_authorize_another(incident_store, pipeline):
    pipeline.actions = [HIGH_ROLLBACK, HIGH_FAILOVER]
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    assert [a.step_index for a in incident_store.list_approvals(incident_id)] == [0, 1]

    decide(incident_store, incident_id, 0, A)
    state, _ = run(incident_store, incident_id, post=HEALTHY)

    assert statuses(state) == [ExecutionStatus.EXECUTED, ExecutionStatus.SKIPPED]
    # The unapproved neighbour is still pending, and the incident (resolved by the first
    # step) no longer accepts decisions on it.
    leftover = incident_store.list_approvals(incident_id)[1]
    assert leftover.status == ApprovalState.PENDING
    with pytest.raises(ApprovalConflictError, match="closed"):
        incident_store.decide_approval(incident_id, leftover.approval_id, A, "alice")


def test_an_approval_does_not_carry_over_to_a_changed_plan(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)  # plan A: rollback, pending
    (old,) = incident_store.list_approvals(incident_id)

    pipeline.actions = [HIGH_FAILOVER]  # the model now proposes something else
    state, persisted = run(incident_store, incident_id)  # plan B

    assert statuses(state) == [ExecutionStatus.SKIPPED]  # B's step has no approval of its own
    assert len(persisted.pending_approval_ids) == 1
    approvals = incident_store.list_approvals(incident_id)
    assert len(approvals) == 2
    new = next(a for a in approvals if a.approval_id != old.approval_id)
    assert new.action_type == HIGH_FAILOVER[0]

    # The obsolete approval can no longer be decided; the current one can.
    with pytest.raises(ApprovalConflictError, match="superseded"):
        incident_store.decide_approval(incident_id, old.approval_id, A, "alice")
    incident_store.decide_approval(incident_id, new.approval_id, A, "alice")
    state, _ = run(incident_store, incident_id)
    assert statuses(state) == [ExecutionStatus.EXECUTED]


def test_an_approval_for_a_replaced_plan_never_authorizes_the_new_one(incident_store, pipeline):
    """The dangerous direction: approve plan A, then the plan changes underneath it."""
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A)  # approves rollback

    pipeline.actions = [HIGH_FAILOVER]  # the plan is replaced before the next run
    state, _ = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.SKIPPED]  # failover was NOT approved
    assert state["verification"].status == VerificationStatus.SKIPPED


def test_the_same_step_in_a_longer_plan_is_not_covered_by_an_old_approval(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A)  # rollback as the *only* step

    pipeline.actions = [LOW, HIGH_ROLLBACK]  # rollback again, but now second in a new plan
    state, _ = run(incident_store, incident_id, post=STILL_BROKEN)

    assert statuses(state) == [ExecutionStatus.EXECUTED, ExecutionStatus.SKIPPED]


def test_one_incidents_approval_never_authorizes_another_incident(incident_store, pipeline):
    approved_incident, other_incident = new_incident_id(), new_incident_id()
    run(incident_store, approved_incident)
    run(incident_store, other_incident)
    decide(incident_store, approved_incident, 0, A)

    state, _ = run(incident_store, other_incident)  # identical plan, different incident

    assert statuses(state) == [ExecutionStatus.SKIPPED]


# --- guardrails stay in charge ---------------------------------------------------------------------------


def test_approval_cannot_override_a_guardrail_block(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A)

    pipeline.confidence = 0.2  # a weak analysis raises HIGH to CRITICAL: blocked outright
    state, persisted = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.BLOCKED]
    assert persisted.pending_approval_ids == []  # nothing for a human to approve
    assert state["next_step"] == "escalate"


# --- unchanged behaviour ---------------------------------------------------------------------------------------------


def test_low_risk_steps_still_run_without_any_approval(incident_store, pipeline):
    pipeline.actions = [LOW]
    incident_id = new_incident_id()

    state, persisted = run(incident_store, incident_id)

    assert statuses(state) == [ExecutionStatus.EXECUTED]
    assert state["next_step"] == "resolve"
    assert persisted.persisted and persisted.pending_approval_ids == []
    assert incident_store.list_approvals(incident_id) == []
    assert stored_status(incident_store, incident_id) == IncidentLifecycleStatus.RESOLVED


def test_without_an_approval_repository_the_workflow_behaves_as_before(pipeline):
    state = handle_incident(
        "Payment service is degraded",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=HEALTHY,
    )
    assert statuses(state) == [ExecutionStatus.SKIPPED]
    assert state["approval_lookup_error"] is None


# --- failures ------------------------------------------------------------------------------------------------------------


class _Wrapped:
    """The real store with selected operations replaced; everything else delegates."""

    def __init__(self, store, **overrides):
        self._store = store
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides.get(name) or getattr(self._store, name)


def _boom(*args, **kwargs):
    raise RuntimeError("database down")


def test_an_unreachable_approval_store_fails_closed(incident_store, pipeline):
    incident_id = new_incident_id()
    run(incident_store, incident_id)
    decide(incident_store, incident_id, 0, A)  # approved in the real store

    state, _ = run(
        incident_store, incident_id, lookup=_Wrapped(incident_store, list_approvals=_boom)
    )

    assert statuses(state) == [ExecutionStatus.SKIPPED]  # an outage must not widen what runs
    assert "approval lookup failed (RuntimeError)" in state["approval_lookup_error"]
    assert state["next_step"] == "escalate"


def test_a_failed_registration_is_reported_and_a_resubmission_repairs_it(incident_store, pipeline):
    incident_id = new_incident_id()
    flaky = _Wrapped(incident_store, request_approvals=_boom)

    state = handle_incident(
        "Payment service is degraded",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        incident_id=incident_id,
        approval_repository=incident_store,
    )
    failed = persist_workflow_run(flaky, state)

    assert failed.persisted is False
    assert "record saved but approvals not registered" in failed.error
    assert incident_store.get_incident(incident_id) is not None  # the record did land
    assert incident_store.list_approvals(incident_id) == []  # but there is nothing to approve yet

    _, repaired = run(incident_store, incident_id)  # the same incident, submitted again
    assert repaired.persisted and len(repaired.pending_approval_ids) == 1


def test_a_failed_record_save_registers_nothing(incident_store, pipeline):
    incident_id = new_incident_id()
    state = handle_incident(
        "Payment service is degraded",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        incident_id=incident_id,
        approval_repository=incident_store,
    )

    failed = persist_workflow_run(_Wrapped(incident_store, save_incident=_boom), state)

    assert failed.persisted is False and "record not saved" in failed.error
    assert failed.pending_approval_ids == []
    assert incident_store.get_incident(incident_id) is None
    assert incident_store.list_approvals(incident_id) == []


def test_repeated_identical_submissions_register_one_approval(incident_store, pipeline):
    incident_id = new_incident_id()
    for _ in range(4):
        run(incident_store, incident_id)
    assert len(incident_store.list_approvals(incident_id)) == 1
