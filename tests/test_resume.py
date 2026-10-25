"""Resuming a stored plan, end to end.

Everything below the LLM is real: the stored snapshot, the guardrails, the approval
decisions, the execution ledger, the executor and the verification. ``forbid_llm`` makes any
attempt to plan, assess or analyse during resume fail the test, so "resume never asks the
model" is checked, not assumed. Runs against every available backend.

Success criterion: an operator can approve one step and then resume the *original* plan,
with no new plan generated, deterministic safety intact, and no step ever started twice.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentic_ai.domain import guardrails as guardrails_module
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentLifecycleStatus
from agentic_ai.domain.remediation import ExecutionStatus
from agentic_ai.domain.step_executions import ExecutionSource, StepExecutionStatus
from agentic_ai.persistence.repository import IncidentNotFoundError, IncidentPersistenceError
from agentic_ai.workflows.resume import ResumeRefusedError, ResumeResult, resume_incident
from tests.approval_helpers import (
    HEALTHY,
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    LOW,
    STILL_BROKEN,
    CountingExecutor,
    WrappedStore,
    boom,
    decide,
    forbid_llm,
    install_pipeline_stubs,
    new_incident_id,
    saved_incident,
    statuses,
    submit,
)
from tests.test_ledger_repository import _store_tampered_snapshot

A, R = ApprovalState.APPROVED, ApprovalState.REJECTED


class _ProcessDied(BaseException):
    """Not an Exception: stands in for the process being killed mid-action."""


@pytest.fixture
def pipeline(monkeypatch):
    return install_pipeline_stubs(monkeypatch)


def prepared(store, pipeline, *actions, post=HEALTHY) -> str:
    """Submit an incident whose plan is ``actions`` (default: one HIGH step) and return its id."""
    pipeline.actions = list(actions) or [HIGH_ROLLBACK]
    incident_id = new_incident_id()
    submit(store, incident_id, post=post)
    return incident_id


def refused(store, incident_id, **kwargs) -> ResumeRefusedError:
    with pytest.raises(ResumeRefusedError) as caught:
        resume_incident(store, incident_id, **kwargs)
    return caught.value


def ledger(store, incident_id):
    return store.list_step_executions(incident_id)


# --- the core promise ----------------------------------------------------------------------


def test_resume_executes_the_stored_plan_without_calling_any_llm(
    incident_store, pipeline, monkeypatch
):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A, "alice")
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    result = resume_incident(
        incident_store, incident_id, post_remediation_scenario=HEALTHY, base_executor=base
    )

    assert result.outcome == "resolve" and result.status == IncidentLifecycleStatus.RESOLVED
    (step,) = result.steps
    assert (step.action_type, step.target_service) == HIGH_ROLLBACK
    assert step.execution_status == "executed" and step.ran_now
    assert base.calls == [HIGH_ROLLBACK]
    assert result.verification.status.value == "recovered"
    assert result.persisted and result.error is None


def test_resume_runs_the_plan_that_was_stored_not_a_new_one(incident_store, pipeline, monkeypatch):
    """Even if the model would now propose something else, the stored plan is what runs."""
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    pipeline.actions = [HIGH_FAILOVER]  # what a fresh planner call would now say
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    result = resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )

    assert base.calls == [HIGH_ROLLBACK] and result.steps[0].action_type == "rollback_deployment"


def test_the_saved_record_reflects_the_resume(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)

    result = resume_incident(
        incident_store, incident_id, post_remediation_scenario=HEALTHY, requested_by="  op-7  "
    )

    record = incident_store.get_incident(incident_id)
    assert record.status == IncidentLifecycleStatus.RESOLVED and record.final_outcome == "resolve"
    assert record.resolved_at is not None
    assert [r.execution_status for r in record.remediations] == ["executed"]
    assert record.verification.status == "recovered"
    assert record.diagnostic.trigger_type == "verification_recovered"
    stage = record.stages[-1]
    assert stage.stage_name == "resume_remediation" and stage.status == "completed"
    assert stage.metadata["requested_by"] == "op-7"
    assert stage.metadata["snapshot_id"] == result.snapshot_id
    # The audit trail is untouched by it.
    assert incident_store.list_approvals(incident_id)[0].status == A


def test_recording_an_approval_never_executes_anything(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)

    decide(incident_store, incident_id, 0, A)

    assert ledger(incident_store, incident_id) == []
    record = incident_store.get_incident(incident_id)
    assert record.status == IncidentLifecycleStatus.AWAITING_APPROVAL


def test_every_step_resume_runs_is_recorded_as_a_resume(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)

    resume_incident(incident_store, incident_id, post_remediation_scenario=HEALTHY)

    (row,) = ledger(incident_store, incident_id)
    assert row.source == ExecutionSource.RESUME and row.status == StepExecutionStatus.SUCCEEDED


def test_approval_does_not_imply_recovery(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)

    result = resume_incident(incident_store, incident_id, post_remediation_scenario=STILL_BROKEN)

    assert result.steps[0].execution_status == "executed"
    assert result.verification.status.value == "unresolved"
    assert result.outcome == "escalate" and result.status == IncidentLifecycleStatus.ESCALATED
    assert incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.ESCALATED


# --- pending and rejected approvals block everything -----------------------------------------


def test_a_pending_approval_refuses_the_resume_and_executes_nothing(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)
    base = CountingExecutor()

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "approvals_pending" and "[0]" in error.message
    assert base.calls == [] and ledger(incident_store, incident_id) == []
    assert (
        incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.AWAITING_APPROVAL
    )


def test_a_rejected_approval_refuses_the_resume_and_executes_nothing(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, R, "bob")
    base = CountingExecutor()

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "approval_rejected"
    assert base.calls == [] and ledger(incident_store, incident_id) == []


def test_approving_one_step_does_not_approve_the_rest_of_the_plan(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline, HIGH_ROLLBACK, HIGH_FAILOVER)
    decide(incident_store, incident_id, 0, A)  # only the first
    base = CountingExecutor()

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "approvals_pending" and "[1]" in error.message
    assert base.calls == []  # all-or-nothing: the approved step does not run on its own


def test_a_rejection_anywhere_blocks_the_whole_plan(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline, HIGH_ROLLBACK, HIGH_FAILOVER)
    decide(incident_store, incident_id, 0, A)
    decide(incident_store, incident_id, 1, R)

    assert (
        refused(incident_store, incident_id, base_executor=CountingExecutor()).code
        == "approval_rejected"
    )
    assert ledger(incident_store, incident_id) == []


# --- guardrails are re-run, and still win ------------------------------------------------------


def test_an_approved_step_is_still_blocked_if_the_guardrails_now_say_so(
    incident_store, pipeline, monkeypatch
):
    """Resume re-runs the *current* policy on the stored inputs. If the policy has since
    tightened, an old approval cannot carry a step past it."""
    incident_id = prepared(incident_store, pipeline)  # stored confidence is 0.9
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    monkeypatch.setattr(
        guardrails_module, "LOW_CONFIDENCE_THRESHOLD", 0.95
    )  # 0.9 now counts as weak
    base = CountingExecutor()

    result = resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )

    (step,) = result.steps
    assert step.disposition == "blocked" and step.execution_status == "blocked"
    assert base.calls == [] and ledger(incident_store, incident_id) == []
    assert result.outcome == "escalate" and result.verification.status.value == "skipped"
    assert incident_store.list_approvals(incident_id)[0].status == A  # the approval itself stands


# --- the stored plan must be intact and current -----------------------------------------------------


def test_a_tampered_snapshot_is_refused(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    original = incident_store.get_plan_snapshot(
        incident_id, incident_store.list_approvals(incident_id)[0].plan_fingerprint
    )
    _store_tampered_snapshot(incident_store, original)
    base = CountingExecutor()

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "snapshot_integrity" and "checksum" in error.message
    assert base.calls == [] and ledger(incident_store, incident_id) == []


def test_an_incident_with_no_stored_plan_cannot_be_resumed(incident_store):
    """An incident saved before snapshots existed has approvals but no plan to resume."""
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    incident_store.decide_approval(record.incident_id, approval.approval_id, A, "alice")

    error = refused(incident_store, record.incident_id, base_executor=CountingExecutor())

    assert error.code == "no_plan_snapshot" and "resubmit" in error.message


def test_an_approval_for_a_replaced_plan_does_not_authorize_the_new_one(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)  # plan A: rollback
    decide(incident_store, incident_id, 0, A)  # approved...
    pipeline.actions = [HIGH_FAILOVER]
    submit(incident_store, incident_id)  # ...then the incident is re-run with plan B
    base = CountingExecutor()

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "approvals_pending"  # plan B has its own, still-pending approval
    assert base.calls == [] and ledger(incident_store, incident_id) == []


def test_resume_follows_the_incidents_current_plan_not_an_older_one(
    incident_store, pipeline, monkeypatch
):
    incident_id = prepared(incident_store, pipeline)  # plan A
    pipeline.actions = [HIGH_FAILOVER]
    submit(incident_store, incident_id)  # plan B replaces it
    # Approve plan B's step. Pick it by what it authorizes, not by list position: approvals
    # from two different plans submitted moments apart can share a timestamp.
    plan_b_approval = next(
        a for a in incident_store.list_approvals(incident_id) if a.action_type == "failover"
    )
    incident_store.decide_approval(incident_id, plan_b_approval.approval_id, A, "alice")
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    result = resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )

    assert base.calls == [HIGH_FAILOVER]  # B ran; A's snapshot is obsolete and never used
    assert result.steps[0].action_type == "failover"


def test_a_closed_incident_is_not_resumed(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline, LOW)  # low risk: resolves on submission
    assert incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.RESOLVED

    assert refused(incident_store, incident_id).code == "incident_closed"


def test_an_unknown_incident_is_not_found(incident_store):
    with pytest.raises(IncidentNotFoundError):
        resume_incident(incident_store, new_incident_id())


# --- repetition and concurrency ------------------------------------------------------------------------------


def test_resuming_again_after_success_executes_nothing_more(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor()
    resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )
    after_first = incident_store.get_incident(incident_id)

    error = refused(incident_store, incident_id, base_executor=base)

    assert error.code == "incident_closed"
    assert base.calls == [HIGH_ROLLBACK]  # still exactly one execution
    assert len(ledger(incident_store, incident_id)) == 1
    assert incident_store.get_incident(incident_id) == after_first


def test_concurrent_resumes_execute_the_step_exactly_once(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor(delay=0.15)  # long enough that the others arrive mid-resume
    attempts = 6
    barrier = threading.Barrier(attempts)

    def attempt(_):
        barrier.wait()
        try:
            return resume_incident(
                incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
            )
        except ResumeRefusedError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        outcomes = list(pool.map(attempt, range(attempts)))

    winners = [o for o in outcomes if isinstance(o, ResumeResult)]
    losers = [o for o in outcomes if isinstance(o, ResumeRefusedError)]
    assert len(winners) == 1 and len(losers) == attempts - 1
    assert {e.code for e in losers} <= {"resume_in_progress", "incident_closed"}
    assert base.calls == [HIGH_ROLLBACK]  # the action ran exactly once
    assert len(ledger(incident_store, incident_id)) == 1


def test_the_workflow_path_cannot_run_an_approved_step_twice(incident_store, pipeline):
    """Regression for Phase 17, where every resubmission after an approval re-executed the
    approved step because nothing recorded that it had already run."""
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)

    first, _ = submit(incident_store, incident_id)
    second, _ = submit(incident_store, incident_id)
    third, _ = submit(incident_store, incident_id)

    assert statuses(first) == [ExecutionStatus.EXECUTED]
    assert "previously executed" not in first["remediation_execution"].step_results[0].output
    for later in (second, third):
        assert statuses(later) == [ExecutionStatus.EXECUTED]
        assert "previously executed" in later["remediation_execution"].step_results[0].output
    (row,) = ledger(incident_store, incident_id)
    assert row.source == ExecutionSource.WORKFLOW  # one execution in total, ever


def test_the_workflow_and_resume_paths_cannot_both_run_a_step(
    incident_store, pipeline, monkeypatch
):
    """Whichever path runs a step first, the other reports it instead of re-running it."""
    incident_id = prepared(incident_store, pipeline, LOW, HIGH_ROLLBACK, post=STILL_BROKEN)
    # The LOW step already ran in the workflow's first run; the HIGH step is awaiting approval.
    assert [(r.step_index, r.source) for r in ledger(incident_store, incident_id)] == [
        (0, ExecutionSource.WORKFLOW)
    ]
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    result = resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )

    first, second = result.steps
    assert first.execution_status == "executed" and not first.ran_now
    assert "previously executed" in first.output
    assert second.execution_status == "executed" and second.ran_now
    assert base.calls == [HIGH_ROLLBACK]  # the LOW step was NOT run again
    assert [(r.step_index, r.source) for r in ledger(incident_store, incident_id)] == [
        (0, ExecutionSource.WORKFLOW),
        (1, ExecutionSource.RESUME),
    ]
    assert result.outcome == "resolve"


# --- failure and recovery ------------------------------------------------------------------------------------------


def test_a_failed_step_stops_the_rest_of_the_plan(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline, HIGH_ROLLBACK, HIGH_FAILOVER)
    decide(incident_store, incident_id, 0, A)
    decide(incident_store, incident_id, 1, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor(fail_actions={"rollback_deployment"})

    result = resume_incident(
        incident_store, incident_id, base_executor=base, post_remediation_scenario=HEALTHY
    )

    first, second = result.steps
    assert first.execution_status == "failed"
    assert second.execution_status == "skipped" and "earlier step did not succeed" in second.output
    assert base.calls == [HIGH_ROLLBACK]  # failover was never attempted
    assert [r.step_index for r in ledger(incident_store, incident_id)] == [0]
    assert result.outcome == "escalate" and result.status == IncidentLifecycleStatus.ESCALATED
    assert result.record.diagnostic.trigger_type == "execution_failure"


def test_a_crash_mid_step_leaves_an_unknown_outcome_that_is_never_replayed(
    incident_store, pipeline, monkeypatch
):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)

    with pytest.raises(_ProcessDied):  # the process dies after the claim, during the action
        resume_incident(
            incident_store,
            incident_id,
            base_executor=CountingExecutor(
                raise_on={"rollback_deployment"}, raise_with=_ProcessDied
            ),
        )

    (row,) = ledger(incident_store, incident_id)
    assert row.is_unknown  # started, never finished
    assert (
        incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.AWAITING_APPROVAL
    )
    with incident_store.resume_lock(incident_id) as lock_is_free:
        assert lock_is_free  # the crash did not leave the incident locked

    retry = CountingExecutor()
    error = refused(incident_store, incident_id, base_executor=retry)

    assert error.code == "unknown_outcome" and "[0]" in error.message
    assert "not replayed automatically" in error.message
    assert retry.calls == []  # the action is not started a second time


def test_steps_that_ran_but_were_not_saved_are_not_run_again(incident_store, pipeline, monkeypatch):
    """The process-failure case the ledger exists for: executed, then the save fails."""
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    first_run = CountingExecutor()

    broken_save = WrappedStore(incident_store, save_incident=boom)
    result = resume_incident(
        broken_save, incident_id, base_executor=first_run, post_remediation_scenario=HEALTHY
    )

    assert result.persisted is False and "incident record not saved" in result.error
    assert result.steps[0].execution_status == "executed"  # it really did run
    assert first_run.calls == [HIGH_ROLLBACK]
    assert (
        incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.AWAITING_APPROVAL
    )

    second_run = CountingExecutor()
    repaired = resume_incident(
        incident_store, incident_id, base_executor=second_run, post_remediation_scenario=HEALTHY
    )

    assert second_run.calls == []  # nothing ran again
    assert repaired.persisted and repaired.outcome == "resolve"
    assert not repaired.steps[0].ran_now and "previously executed" in repaired.steps[0].output
    assert incident_store.get_incident(incident_id).status == IncidentLifecycleStatus.RESOLVED


def test_an_unreachable_approval_store_fails_closed(incident_store, pipeline, monkeypatch):
    incident_id = prepared(incident_store, pipeline)
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    with pytest.raises(IncidentPersistenceError, match="approval lookup failed"):
        resume_incident(
            WrappedStore(incident_store, list_approvals=boom), incident_id, base_executor=base
        )

    assert base.calls == [] and ledger(incident_store, incident_id) == []


def test_an_incident_that_closes_during_the_resume_stops_the_remaining_steps(
    incident_store, pipeline, monkeypatch
):
    """Revalidation happens immediately before each step, not just once at the start."""
    incident_id = prepared(incident_store, pipeline, HIGH_ROLLBACK, HIGH_FAILOVER)
    decide(incident_store, incident_id, 0, A)
    decide(incident_store, incident_id, 1, A)
    forbid_llm(monkeypatch)
    base = CountingExecutor()

    class ClosesAfterFirstStep(CountingExecutor):
        def __call__(self, step):
            result = base(step)
            # Someone else closes the incident while this resume is mid-plan.
            record = incident_store.get_incident(incident_id)
            incident_store.save_incident(
                record.model_copy(update={"status": IncidentLifecycleStatus.ESCALATED})
            )
            return result

    result = resume_incident(
        incident_store,
        incident_id,
        base_executor=ClosesAfterFirstStep(),
        post_remediation_scenario=HEALTHY,
    )

    first, second = result.steps
    assert first.execution_status == "executed" and first.ran_now
    assert second.execution_status == "blocked" and "no longer open" in second.output
    assert base.calls == [HIGH_ROLLBACK]  # failover was refused on revalidation, not run
    assert [r.step_index for r in ledger(incident_store, incident_id)] == [0]
    # The save then runs into the closed record and reports it rather than overwriting it.
    assert result.persisted is False


# --- the snapshot is written once, and only when needed ------------------------------------------------------------------


def test_a_snapshot_is_stored_only_for_plans_awaiting_a_human(incident_store, pipeline):
    low_only = prepared(incident_store, pipeline, LOW)
    needs_human = prepared(incident_store, pipeline, HIGH_ROLLBACK)

    assert incident_store.list_approvals(low_only) == []
    fingerprint = incident_store.list_approvals(needs_human)[0].plan_fingerprint
    assert incident_store.get_plan_snapshot(needs_human, fingerprint) is not None


def test_presenting_the_same_plan_again_keeps_the_original_snapshot(incident_store, pipeline):
    incident_id = prepared(incident_store, pipeline)
    fingerprint = incident_store.list_approvals(incident_id)[0].plan_fingerprint
    original = incident_store.get_plan_snapshot(incident_id, fingerprint)

    submit(incident_store, incident_id)
    submit(incident_store, incident_id)

    assert incident_store.get_plan_snapshot(incident_id, fingerprint) == original


def test_changed_guardrail_inputs_are_a_new_snapshot_and_resume_uses_the_latest(
    incident_store, pipeline, monkeypatch
):
    incident_id = prepared(incident_store, pipeline)  # confidence 0.9
    fingerprint = incident_store.list_approvals(incident_id)[0].plan_fingerprint
    first = incident_store.get_plan_snapshot(incident_id, fingerprint)

    pipeline.confidence = 0.7  # still HIGH-needs-approval, but a different presentation
    submit(incident_store, incident_id)
    latest = incident_store.get_plan_snapshot(incident_id, fingerprint)

    assert latest.snapshot_id != first.snapshot_id and latest.root_cause_confidence == 0.7
    decide(incident_store, incident_id, 0, A)
    forbid_llm(monkeypatch)
    result = resume_incident(incident_store, incident_id, post_remediation_scenario=HEALTHY)
    assert result.snapshot_id == latest.snapshot_id
