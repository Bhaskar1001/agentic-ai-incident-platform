"""Resume an incident's stored, approved remediation plan, without re-planning.

``POST /incidents`` runs the whole workflow, including an LLM planner whose output differs
from run to run. That made applying an approval awkward: resubmitting could produce a
different plan, which the approval (rightly) would not cover. This module is the explicit
alternative. It takes the plan that was *stored when it was presented for approval* and
carries it forward. **It never calls the planner or any LLM.**

What it does, in order
----------------------
1. Take the incident's resume lock (non-blocking). Only one resume runs at a time.
2. Load the incident. A closed incident is refused.
3. Load the stored plan snapshot for the incident's *current* plan. No snapshot, or one that
   fails its integrity check, is refused.
4. Re-run the deterministic guardrails on the snapshot's stored inputs (no model involved).
5. Apply the persisted approval decisions. If any required approval is still pending, or was
   rejected, refuse: nothing is executed. If the approval lookup itself fails, the error
   propagates and nothing is executed (fail closed).
6. Refuse if any step already has an unknown outcome from an earlier attempt.
7. Execute through the execution ledger: each step is claimed durably before it runs, so a step
   that already ran is never run again, and revalidated immediately before it is claimed.
   Execution stops at the first step that fails or whose outcome is unknown.
8. Run the existing deterministic verification, decide resolve or escalate with the existing
   routing rule, and save the result.

Safety boundary (unchanged): an approval authorizes only the exact step it covers, never
overrides a guardrail, and never guarantees the incident resolves. Recording an approval does
not call this; resuming is a separate, explicit act.

Honest limits: the ledger prevents this system from *starting* a step twice. It cannot make an
external action exactly-once; see ``domain/step_executions.py`` for what an unknown outcome
means. ``requested_by`` is an unauthenticated label.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from agentic_ai.domain.approvals import apply_persisted_decisions, plan_fingerprint
from agentic_ai.domain.guardrails import (
    ApprovalState,
    Disposition,
    GuardedPlan,
    evaluate_step,
    guard_plan,
)
from agentic_ai.domain.persistence import (
    IncidentLifecycleStatus,
    IncidentRecord,
    IncidentRemediationRecord,
    IncidentStageRecord,
)
from agentic_ai.domain.plan_snapshots import SnapshotIntegrityError
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationStep,
    StepExecutionResult,
)
from agentic_ai.domain.step_executions import ExecutionSource
from agentic_ai.domain.summaries import DiagnosticSummary, VerificationSummary
from agentic_ai.domain.verification import VerificationResult
from agentic_ai.persistence.ledgered_execution import LedgerStepExecutor
from agentic_ai.persistence.repository import (
    IncidentNotFoundError,
    IncidentPersistenceError,
    IncidentStore,
)
from agentic_ai.tools.mock_remediation import (
    execute_guarded_plan,
    execute_mock_remediation_step,
)
from agentic_ai.workflows.incident import route_after_verification, verify_remediation_node

logger = logging.getLogger("agentic_ai.resume")

ResumeRefusal = Literal[
    "resume_in_progress",
    "incident_closed",
    "no_remediation_plan",
    "no_plan_snapshot",
    "snapshot_integrity",
    "approval_rejected",
    "approvals_pending",
    "unknown_outcome",
]


class ResumeRefusedError(Exception):
    """The resume was refused before anything was executed. ``code`` says why."""

    def __init__(self, code: ResumeRefusal, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResumeStepReport:
    step_index: int
    action_type: str
    target_service: str
    disposition: str
    execution_status: str
    output: str
    error: str | None
    ran_now: bool
    """True only if *this* resume executed the step; False for a step it merely reported on
    (executed earlier, blocked, not attempted, or of unknown outcome)."""


@dataclass(frozen=True)
class ResumeResult:
    incident_id: str
    plan_fingerprint: str
    snapshot_id: str
    outcome: Literal["resolve", "escalate"]
    status: IncidentLifecycleStatus
    steps: list[ResumeStepReport]
    verification: VerificationResult
    persisted: bool
    error: str | None = None
    record: IncidentRecord | None = field(default=None, repr=False)


def _current_plan_fingerprint(record: IncidentRecord) -> str:
    ordered = sorted(record.remediations, key=lambda r: r.step_order)
    return plan_fingerprint((r.action_type, r.target_service) for r in ordered)


def _positions(guarded: GuardedPlan, disposition: Disposition, state: ApprovalState) -> list[int]:
    return [
        i
        for i, g in enumerate(guarded.steps)
        if g.disposition == disposition and g.approval == state
    ]


def _summarise(
    outcome: str, verification: VerificationResult, execution_results: list[StepExecutionResult]
) -> DiagnosticSummary:
    if outcome == "resolve":
        return DiagnosticSummary(
            decision="resolve",
            trigger_type="verification_recovered",
            primary_reason="Remediation resumed from the stored plan and verified: service "
            "health recovered.",
        )
    statuses = {r.status for r in execution_results}
    if ExecutionStatus.UNKNOWN in statuses:
        return DiagnosticSummary(
            decision="escalate",
            trigger_type="execution_failure",
            primary_reason="A resumed step has an unknown outcome and was not replayed; an "
            "operator must check whether it took effect.",
        )
    if ExecutionStatus.FAILED in statuses:
        return DiagnosticSummary(
            decision="escalate",
            trigger_type="execution_failure",
            primary_reason="A resumed remediation step failed; later steps were not attempted.",
        )
    if ExecutionStatus.BLOCKED in statuses:
        return DiagnosticSummary(
            decision="escalate",
            trigger_type="guardrail_rejection",
            primary_reason="A resumed step was blocked by the guardrails on revalidation.",
        )
    reasons = "; ".join(verification.reasons) if verification.reasons else verification.status.value
    return DiagnosticSummary(
        decision="escalate",
        trigger_type="verification_unresolved",
        primary_reason=f"Post-remediation verification did not recover the service: {reasons}.",
    )


def resume_incident(
    store: IncidentStore,
    incident_id: str,
    *,
    post_remediation_scenario: str | None = None,
    requested_by: str | None = None,
    base_executor: Callable[[RemediationStep], StepExecutionResult] = execute_mock_remediation_step,
) -> ResumeResult:
    """Resume ``incident_id``'s stored plan. See the module docstring for the full contract.

    Raises
    ------
    IncidentNotFoundError
        If the incident does not exist.
    ResumeRefusedError
        If it must not (or cannot safely) be resumed; nothing was executed.
    IncidentPersistenceError
        If storage fails before anything is executed (for example the approval lookup).
        Failures *after* execution are reported in ``ResumeResult.persisted``/``error``
        instead, because the executions already happened.
    """
    requested_by = (requested_by or "").strip() or None  # an unauthenticated label, tidied
    with store.resume_lock(incident_id) as acquired:
        if not acquired:
            raise ResumeRefusedError(
                "resume_in_progress", "another resume of this incident is already running."
            )
        return _resume_locked(
            store, incident_id, post_remediation_scenario, requested_by, base_executor
        )


def _resume_locked(
    store: IncidentStore,
    incident_id: str,
    post_remediation_scenario: str | None,
    requested_by: str | None,
    base_executor: Callable[[RemediationStep], StepExecutionResult],
) -> ResumeResult:
    started_at = datetime.now(timezone.utc)

    record = store.get_incident(incident_id)
    if record is None:
        raise IncidentNotFoundError(incident_id)
    if record.status.is_terminal:
        raise ResumeRefusedError(
            "incident_closed",
            f"the incident is already {record.status.value}; a closed incident is not resumed.",
        )
    if not record.remediations:
        raise ResumeRefusedError("no_remediation_plan", "the incident has no remediation plan.")

    # The plan the incident holds *now*. A snapshot of any other plan is obsolete.
    current_fingerprint = _current_plan_fingerprint(record)
    try:
        snapshot = store.get_plan_snapshot(incident_id, current_fingerprint)
    except SnapshotIntegrityError as exc:
        raise ResumeRefusedError(
            "snapshot_integrity", f"the stored plan failed its integrity check ({exc})"
        ) from exc
    if snapshot is None:
        raise ResumeRefusedError(
            "no_plan_snapshot",
            "no stored plan matches the incident's current plan (it predates plan snapshots, "
            "was never stored, or the plan was replaced); resubmit the incident instead.",
        )

    try:
        plan = snapshot.to_plan()
        analysis = snapshot.to_analysis()
    except SnapshotIntegrityError as exc:
        raise ResumeRefusedError("snapshot_integrity", str(exc)) from exc
    services = set(snapshot.investigated_services)

    # Re-run the deterministic guardrails on the stored inputs. No model is involved.
    guarded = guard_plan(plan, analysis, services)
    if len(guarded.steps) != len(record.remediations):
        raise ResumeRefusedError(
            "snapshot_integrity", "the stored plan and the incident's plan have different lengths."
        )

    # Fail closed: if the lookup fails for any reason, nothing has been executed and a
    # persistence error propagates. An unreachable approval store must not be readable as
    # "no approvals required".
    try:
        approvals = store.list_approvals(incident_id)
    except IncidentPersistenceError:
        raise
    except Exception as exc:
        raise IncidentPersistenceError(
            f"approval lookup failed ({type(exc).__name__}); nothing was executed"
        ) from exc
    guarded = apply_persisted_decisions(guarded, approvals)

    rejected = _positions(guarded, Disposition.REQUIRES_APPROVAL, ApprovalState.REJECTED)
    if rejected:
        raise ResumeRefusedError(
            "approval_rejected", f"step(s) {rejected} were rejected; the plan is not resumed."
        )
    pending = _positions(guarded, Disposition.REQUIRES_APPROVAL, ApprovalState.PENDING)
    if pending:
        raise ResumeRefusedError(
            "approvals_pending", f"step(s) {pending} still need an approval decision."
        )

    unknown = [
        e.step_index
        for e in store.list_step_executions(incident_id, current_fingerprint)
        if e.is_unknown
    ]
    if unknown:
        raise ResumeRefusedError(
            "unknown_outcome",
            f"step(s) {unknown} were started earlier but never recorded a result, so whether "
            "they took effect is unknown. They are not replayed automatically; an operator "
            "must check the target and resolve this by hand.",
        )

    def precheck(step: RemediationStep) -> str | None:
        """Immediately before each step is claimed: still allowed, incident still the same."""
        if evaluate_step(step, analysis, services).disposition == Disposition.BLOCKED:
            return "blocked by the guardrails"
        latest = store.get_incident(incident_id)
        if latest is None or latest.status.is_terminal:
            return "the incident is no longer open"
        if _current_plan_fingerprint(latest) != current_fingerprint:
            return "the incident's plan changed during the resume"
        return None

    executor = LedgerStepExecutor(
        store,
        incident_id,
        guarded,
        ExecutionSource.RESUME,
        base_executor,
        precheck=precheck,
        stop_after_problem=True,
    )
    execution = execute_guarded_plan(guarded, step_executor=executor)

    # The existing deterministic verification and routing, unchanged.
    verification = verify_remediation_node(
        {
            "remediation_execution": execution,
            "scenario": record.scenario,
            "post_remediation_scenario": post_remediation_scenario,
        }
    )["verification"]
    outcome: Literal["resolve", "escalate"] = route_after_verification(
        {"verification": verification}
    )
    status = (
        IncidentLifecycleStatus.RESOLVED
        if outcome == "resolve"
        else IncidentLifecycleStatus.ESCALATED
    )

    finished_at = datetime.now(timezone.utc)
    results = execution.step_results
    steps = [
        ResumeStepReport(
            step_index=i,
            action_type=getattr(g.step.action_type, "value", str(g.step.action_type)),
            target_service=g.step.target_service,
            disposition=g.disposition.value,
            execution_status=r.status.value,
            output=r.output,
            error=r.error,
            ran_now=i in executor.ran_steps,
        )
        for i, (g, r) in enumerate(zip(guarded.steps, results))
    ]

    updated = record.model_copy(
        update={
            "status": status,
            "final_outcome": outcome,
            "diagnostic": _summarise(outcome, verification, results),
            "verification": VerificationSummary(
                status=verification.status.value,
                attempts=verification.attempts,
                reasons=list(verification.reasons),
            ),
            "remediations": [
                IncidentRemediationRecord(
                    step_order=s.step_index,
                    action_type=s.action_type,
                    target_service=s.target_service,
                    disposition=s.disposition,
                    execution_status=s.execution_status,
                    parameters={},
                    output=s.output,
                )
                for s in steps
            ],
            "stages": [
                *record.stages,
                IncidentStageRecord(
                    stage_name="resume_remediation",
                    status="completed",
                    start_time=started_at,
                    end_time=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000.0,
                    metadata={
                        "snapshot_id": snapshot.snapshot_id,
                        "plan_fingerprint": current_fingerprint,
                        "steps_run_now": len(executor.ran_steps),
                        "requested_by": requested_by,  # an unauthenticated label
                    },
                ),
            ],
            "updated_at": finished_at,
            "resolved_at": finished_at,
        }
    )

    # The steps have already run (and are in the ledger), so a failure to save the incident
    # must not hide that. Report it; a second resume reads the ledger, executes nothing
    # again, and completes the save.
    persisted, error = True, None
    try:
        store.save_incident(updated)
    except Exception as exc:
        logger.error("Resume executed but the incident was not saved: %s", type(exc).__name__)
        persisted, error = False, f"incident record not saved ({type(exc).__name__})"

    return ResumeResult(
        incident_id=incident_id,
        plan_fingerprint=current_fingerprint,
        snapshot_id=snapshot.snapshot_id,
        outcome=outcome,
        status=status,
        steps=steps,
        verification=verification,
        persisted=persisted,
        error=error,
        record=updated if persisted else None,
    )


__all__ = [
    "ResumeRefusal",
    "ResumeRefusedError",
    "ResumeResult",
    "ResumeStepReport",
    "resume_incident",
]
