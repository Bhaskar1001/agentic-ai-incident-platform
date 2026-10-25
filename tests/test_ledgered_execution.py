"""The ledgered executor: claim, run, complete, and every way that can go wrong.

Each failure is resolved in the safe direction. The invariant under test is that a step is
never *started* twice, and that anything uncertain is reported as uncertain rather than
retried.
"""

import pytest

from agentic_ai.domain.remediation import ExecutionStatus
from agentic_ai.domain.step_executions import ExecutionSource, StepExecutionStatus
from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from agentic_ai.persistence.ledgered_execution import LedgerStepExecutor
from agentic_ai.tools.mock_remediation import execute_guarded_plan
from tests.approval_helpers import (
    LOW,
    CountingExecutor,
    boom,
    guarded,
    new_incident_id,
)

RESTART = ("restart_service", "payment")  # MEDIUM; allowed because 'payment' was examined


class _DiedWithTheProcess(BaseException):
    """Not an Exception: stands in for the process being killed mid-action."""


def _run(store, incident_id, plan, base, **kwargs):
    executor = LedgerStepExecutor(
        store, incident_id, plan, ExecutionSource.WORKFLOW, base, **kwargs
    )
    return execute_guarded_plan(plan, step_executor=executor), executor


def _ledger(store, incident_id):
    return store.list_step_executions(incident_id)


def test_a_step_is_claimed_run_and_recorded() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    plan = guarded(LOW, RESTART)

    execution, executor = _run(store, incident_id, plan, base)

    assert [r.status for r in execution.step_results] == [ExecutionStatus.EXECUTED] * 2
    assert base.calls == [LOW, RESTART]
    records = _ledger(store, incident_id)
    assert [(r.step_index, r.status, r.source) for r in records] == [
        (0, StepExecutionStatus.SUCCEEDED, ExecutionSource.WORKFLOW),
        (1, StepExecutionStatus.SUCCEEDED, ExecutionSource.WORKFLOW),
    ]
    assert records[0].output and records[0].finished_at is not None
    assert executor.ran_steps == [0, 1]


def test_running_the_same_plan_again_runs_nothing_again() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    plan = guarded(LOW, RESTART)
    _run(store, incident_id, plan, base)

    second, executor = _run(store, incident_id, plan, base)

    assert base.calls == [LOW, RESTART]  # still only the first two calls
    assert [r.status for r in second.step_results] == [ExecutionStatus.EXECUTED] * 2
    assert all("previously executed" in r.output and "not run again" in r.output
               for r in second.step_results)  # fmt: skip
    assert executor.ran_steps == []  # reported on, not executed
    assert len(_ledger(store, incident_id)) == 2


def test_a_step_that_failed_is_not_retried_automatically() -> None:
    store, incident_id = InMemoryIncidentRepository(), new_incident_id()
    base = CountingExecutor(fail_actions={"scale_up"})
    plan = guarded(LOW)

    first, _ = _run(store, incident_id, plan, base)
    second, executor = _run(store, incident_id, plan, base)

    assert first.step_results[0].status == ExecutionStatus.FAILED
    assert second.step_results[0].status == ExecutionStatus.FAILED
    assert "previously failed" in second.step_results[0].output
    assert "not retried automatically" in second.step_results[0].output
    assert base.calls == [LOW] and executor.ran_steps == []
    assert _ledger(store, incident_id)[0].status == StepExecutionStatus.FAILED


def test_a_step_left_started_is_unknown_and_never_replayed() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    plan = guarded(LOW)
    with pytest.raises(_DiedWithTheProcess):  # the process dies after the claim, mid-action
        _run(
            store,
            incident_id,
            plan,
            CountingExecutor(raise_on={"scale_up"}, raise_with=_DiedWithTheProcess),
        )
    assert _ledger(store, incident_id)[0].is_unknown  # fmt: skip

    after_restart, executor = _run(store, incident_id, plan, base)

    (result,) = after_restart.step_results
    assert result.status == ExecutionStatus.UNKNOWN and result.error == "unknown_outcome"
    assert "outcome unknown" in result.output and "not replayed" in result.output
    assert base.calls == []  # the action was not started again
    assert executor.ran_steps == []


def test_an_executor_that_raises_leaves_the_outcome_unknown() -> None:
    store, incident_id = InMemoryIncidentRepository(), new_incident_id()
    base = CountingExecutor(raise_on={"scale_up"})

    execution, executor = _run(store, incident_id, guarded(LOW), base)

    (result,) = execution.step_results
    assert result.status == ExecutionStatus.UNKNOWN and "RuntimeError" in result.output
    assert _ledger(store, incident_id)[0].status == StepExecutionStatus.STARTED  # still unknown
    assert executor.ran_steps == []


def test_not_being_able_to_claim_means_not_running() -> None:
    store = InMemoryIncidentRepository()
    broken = type("Broken", (), {"claim_step_execution": boom})()
    base, incident_id = CountingExecutor(), new_incident_id()

    execution, executor = _run(broken, incident_id, guarded(LOW), base)

    (result,) = execution.step_results
    assert result.status == ExecutionStatus.SKIPPED and result.error == "ledger_unavailable"
    assert "failing closed" in result.output
    assert base.calls == [] and executor.ran_steps == []
    assert _ledger(store, incident_id) == []


def test_not_being_able_to_record_the_outcome_leaves_it_unknown_to_the_next_attempt() -> None:
    real = InMemoryIncidentRepository()
    incident_id, base = new_incident_id(), CountingExecutor()

    class CompletionFails:
        claim_step_execution = real.claim_step_execution
        complete_step_execution = staticmethod(boom)

    plan = guarded(LOW)
    first, _ = _run(CompletionFails(), incident_id, plan, base)

    assert first.step_results[0].status == ExecutionStatus.EXECUTED  # the real outcome is returned
    assert _ledger(real, incident_id)[0].is_unknown  # but the ledger could not be told

    second, _ = _run(real, incident_id, plan, base)
    assert second.step_results[0].status == ExecutionStatus.UNKNOWN
    assert base.calls == [LOW]  # never replayed


def test_a_refused_precheck_means_blocked_and_no_ledger_row() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()

    execution, executor = _run(
        store,
        incident_id,
        guarded(LOW),
        base,
        precheck=lambda step: "the incident is no longer open",
    )

    (result,) = execution.step_results
    assert result.status == ExecutionStatus.BLOCKED
    assert "refused on revalidation" in result.output and "no longer open" in result.output
    assert base.calls == [] and _ledger(store, incident_id) == []


def test_a_precheck_that_cannot_run_fails_closed() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()

    execution, _ = _run(store, incident_id, guarded(LOW), base, precheck=boom)

    assert execution.step_results[0].status == ExecutionStatus.BLOCKED
    assert "could not revalidate" in execution.step_results[0].output
    assert base.calls == []


def test_the_precheck_runs_before_every_step() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    seen = []

    def precheck(step):
        seen.append(step.action_type.value)

    _run(store, incident_id, guarded(LOW, RESTART), base, precheck=precheck)

    assert seen == ["scale_up", "restart_service"]


def test_stopping_after_a_failure_leaves_later_steps_unattempted() -> None:
    store, incident_id = InMemoryIncidentRepository(), new_incident_id()
    base = CountingExecutor(fail_actions={"scale_up"})

    execution, executor = _run(
        store, incident_id, guarded(LOW, RESTART), base, stop_after_problem=True
    )

    first, second = execution.step_results
    assert first.status == ExecutionStatus.FAILED
    assert (
        second.status == ExecutionStatus.SKIPPED and "earlier step did not succeed" in second.output
    )
    assert base.calls == [LOW]
    assert [r.step_index for r in _ledger(store, incident_id)] == [0]  # no row for the skipped step


def test_stopping_after_an_unknown_outcome_too() -> None:
    store, incident_id = InMemoryIncidentRepository(), new_incident_id()
    base = CountingExecutor(raise_on={"scale_up"})

    execution, _ = _run(store, incident_id, guarded(LOW, RESTART), base, stop_after_problem=True)

    assert [r.status for r in execution.step_results] == [
        ExecutionStatus.UNKNOWN,
        ExecutionStatus.SKIPPED,
    ]
    assert base.calls == [LOW]


def test_without_the_option_later_steps_still_run_after_a_failure() -> None:
    """The workflow path keeps its original behaviour; only resume stops early."""
    store, incident_id = InMemoryIncidentRepository(), new_incident_id()
    base = CountingExecutor(fail_actions={"scale_up"})

    execution, _ = _run(store, incident_id, guarded(LOW, RESTART), base)

    assert [r.status for r in execution.step_results] == [
        ExecutionStatus.FAILED,
        ExecutionStatus.EXECUTED,
    ]


def test_a_step_that_is_not_part_of_the_plan_is_not_run() -> None:
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    executor = LedgerStepExecutor(store, incident_id, guarded(LOW), ExecutionSource.WORKFLOW, base)

    stranger = guarded(RESTART).steps[0].step  # a step object that is not in the plan
    result = executor(stranger)

    assert result.status == ExecutionStatus.SKIPPED and result.error == "unknown_step"
    assert base.calls == []


def test_guardrails_still_decide_what_reaches_the_ledger() -> None:
    """A step awaiting approval is never passed to the executor, so it is never claimed."""
    store, incident_id, base = InMemoryIncidentRepository(), new_incident_id(), CountingExecutor()
    plan = guarded(LOW, ("rollback_deployment", "payment"))  # the second step needs approval

    execution, executor = _run(store, incident_id, plan, base)

    assert [r.status for r in execution.step_results] == [
        ExecutionStatus.EXECUTED,
        ExecutionStatus.SKIPPED,
    ]
    assert base.calls == [LOW] and [r.step_index for r in _ledger(store, incident_id)] == [0]
