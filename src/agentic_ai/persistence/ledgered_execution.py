"""Run remediation steps through the execution ledger.

``execute_guarded_plan`` decides *which* steps may run (guardrails and approvals) and calls
a step executor for each. This module supplies a step executor that wraps another one in the
ledger protocol (see ``domain/step_executions.py``):

    claim  -> run  -> complete

* **claim**: record intent as ``STARTED`` and commit it *before* acting. Whoever loses the
  claim runs nothing and is told what the earlier attempt left behind.
* **run**: call the real step executor.
* **complete**: record ``SUCCEEDED`` or ``FAILED``.

Every failure here is resolved in the safe direction, never towards running something twice:

====================================  =======================================================
Situation                             Result
====================================  =======================================================
Cannot record the claim               Not executed ("fails closed"), reported as skipped
Step already succeeded                Not executed again; reported as executed, marked previous
Step already failed                   Not retried automatically; reported as failed
Step started, never finished          Not replayed; reported ``UNKNOWN``
The action itself raises              Left started (it may have partly run); ``UNKNOWN``
Cannot record the completion          Outcome returned, ledger row stays started = unknown later
====================================  =======================================================

Two optional safeguards, used by the resume operation:

* ``precheck``: called for each step immediately *before* it is claimed. It returns a reason to
  refuse the step (for example "the incident is no longer open") or None. A refused step is
  reported as blocked and leaves no ledger row; a precheck that itself fails also refuses.
* ``stop_after_problem``: once a step ends failed or unknown, later steps are not attempted,
  because later steps in a remediation usually depend on earlier ones.

Used by both the workflow's execution step and the resume operation, so the two paths cannot
disagree about whether a step already ran.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from agentic_ai.domain.approvals import plan_fingerprint_of, step_fingerprint
from agentic_ai.domain.guardrails import GuardedPlan
from agentic_ai.domain.remediation import ExecutionStatus, RemediationStep, StepExecutionResult
from agentic_ai.domain.step_executions import (
    ExecutionSource,
    StepExecutionRecord,
    StepExecutionStatus,
    new_execution_id,
)
from agentic_ai.persistence.repository import ExecutionLedger
from agentic_ai.tools.mock_remediation import execute_mock_remediation_step

logger = logging.getLogger("agentic_ai.execution")

StepExecutor = Callable[[RemediationStep], StepExecutionResult]


def _when(record: StepExecutionRecord) -> str:
    moment = record.finished_at or record.started_at
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class LedgerStepExecutor:
    """A ``step_executor`` for ``execute_guarded_plan`` that claims each step first."""

    def __init__(
        self,
        ledger: ExecutionLedger,
        incident_id: str,
        guarded: GuardedPlan,
        source: ExecutionSource,
        base_executor: StepExecutor = execute_mock_remediation_step,
        *,
        precheck: Callable[[RemediationStep], str | None] | None = None,
        stop_after_problem: bool = False,
    ) -> None:
        self._ledger = ledger
        self._incident_id = incident_id
        self._source = source
        self._base = base_executor
        self._precheck = precheck
        self._stop_after_problem = stop_after_problem
        self._halted = False
        #: Positions of the steps this executor actually ran (claimed and executed), as
        #: opposed to steps it only reported on from an earlier attempt.
        self.ran_steps: list[int] = []
        self._plan_fingerprint = plan_fingerprint_of(guarded)
        # execute_guarded_plan hands us the very step objects held by ``guarded``, so object
        # identity recovers each step's position without trusting anything the step carries.
        self._index_of = {id(g.step): i for i, g in enumerate(guarded.steps)}

    def __call__(self, step: RemediationStep) -> StepExecutionResult:
        if self._halted:
            return self._result(
                step,
                ExecutionStatus.SKIPPED,
                "not attempted: an earlier step did not succeed",
                "earlier_step_failed",
            )

        result = self._execute(step)

        if self._stop_after_problem and result.status in (
            ExecutionStatus.FAILED,
            ExecutionStatus.UNKNOWN,
        ):
            self._halted = True
        return result

    def _execute(self, step: RemediationStep) -> StepExecutionResult:
        index = self._index_of.get(id(step))
        if index is None:
            return self._result(
                step,
                ExecutionStatus.SKIPPED,
                "execution not attempted: the step is not part of the plan being executed",
                "unknown_step",
            )

        if self._precheck is not None:
            try:
                refusal = self._precheck(step)
            except Exception as exc:
                refusal = f"could not revalidate ({type(exc).__name__}); failing closed"
            if refusal is not None:
                return self._result(
                    step,
                    ExecutionStatus.BLOCKED,
                    f"execution refused on revalidation: {refusal}",
                    refusal,
                )

        record = StepExecutionRecord(
            execution_id=new_execution_id(),
            incident_id=self._incident_id,
            plan_fingerprint=self._plan_fingerprint,
            step_index=index,
            step_fingerprint=step_fingerprint(step.action_type, step.target_service),
            action_type=getattr(step.action_type, "value", str(step.action_type)),
            target_service=step.target_service,
            source=self._source,
        )

        try:
            stored, claimed = self._ledger.claim_step_execution(record)
        except Exception as exc:
            logger.error(
                "Could not record execution intent (%s); step not run.", type(exc).__name__
            )
            return self._result(
                step,
                ExecutionStatus.SKIPPED,
                f"execution not attempted: could not record intent ({type(exc).__name__}); "
                "failing closed",
                "ledger_unavailable",
            )

        if not claimed:
            return self._from_existing(step, stored)

        try:
            result = self._base(step)
        except Exception as exc:
            # The action may have partly happened. Leave the claim as 'started', which is
            # exactly what "unknown" means, and do not replay it.
            logger.error("Executor raised %s; outcome is unknown.", type(exc).__name__)
            return self._result(
                step,
                ExecutionStatus.UNKNOWN,
                f"executor raised {type(exc).__name__}; whether the action took effect is unknown",
                "unknown_outcome",
            )

        self.ran_steps.append(index)
        succeeded = result.status == ExecutionStatus.EXECUTED
        try:
            self._ledger.complete_step_execution(
                self._incident_id,
                stored.execution_id,
                StepExecutionStatus.SUCCEEDED if succeeded else StepExecutionStatus.FAILED,
                result.output,
                result.error,
            )
        except Exception as exc:
            # The real outcome is returned, but the ledger still says 'started', so a later
            # attempt will treat this step as unknown and will not replay it.
            logger.error(
                "Could not record the outcome of '%s' (%s); the ledger will show it as unknown.",
                stored.execution_id,
                type(exc).__name__,
            )
        return result

    # -- what an earlier attempt left behind ----------------------------------------------

    def _from_existing(
        self, step: RemediationStep, existing: StepExecutionRecord
    ) -> StepExecutionResult:
        origin = f"{existing.source.value}, {_when(existing)}"
        if existing.status == StepExecutionStatus.SUCCEEDED:
            return self._result(
                step,
                ExecutionStatus.EXECUTED,
                f"[previously executed ({origin}); not run again] {existing.output or ''}".rstrip(),
                None,
            )
        if existing.status == StepExecutionStatus.FAILED:
            return self._result(
                step,
                ExecutionStatus.FAILED,
                f"[previously failed ({origin}); not retried automatically] "
                f"{existing.output or ''}".rstrip(),
                existing.error or "previously failed",
            )
        return self._result(
            step,
            ExecutionStatus.UNKNOWN,
            f"[outcome unknown] an attempt started ({origin}) but no result was ever recorded, "
            "or it is still running; it may or may not have taken effect, so it is not replayed",
            "unknown_outcome",
        )

    @staticmethod
    def _result(
        step: RemediationStep, status: ExecutionStatus, output: str, error: str | None
    ) -> StepExecutionResult:
        return StepExecutionResult(step=step, status=status, output=output, error=error)


__all__ = ["LedgerStepExecutor"]
