"""Persisted approvals for high-risk remediation steps.

``GuardedPlan.approve()/reject()`` already model the gate inside a single run.
This module is what lets a human decision survive between runs: an approval is
a stored fact about *one exact step of one exact plan*, and the workflow applies
it back onto a freshly guarded plan at the point where execution is authorized.

Identity, not just a flag
-------------------------
An approval is bound to ``(incident_id, plan_fingerprint, step_index)`` and also
carries a fingerprint of the step itself. The fingerprints cover what a human
actually authorizes (which action, on which target, at which position in which
plan) and deliberately exclude the model's rationale prose, which differs on
every run. So:

* re-planning to the same actions re-uses the decision;
* re-planning to different actions does not: an approval for plan A can never
  authorize a step of plan B, and approving step 1 never authorizes step 2.

Approval only ever substitutes for the human gate. A step the guardrails block
stays blocked, and the executor still re-checks every step itself.

Free of LLM, database and framework imports.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_ai.domain.guardrails import (
    ApprovalState,
    Disposition,
    GuardedPlan,
)
from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRemediationRecord


class ApprovalDecisionError(Exception):
    """A decision the lifecycle does not permit (bad outcome, or already decided)."""


_DECIDED = (ApprovalState.APPROVED, ApprovalState.REJECTED)
_STORED_STATES = (ApprovalState.PENDING, *_DECIDED)


def _text(value: Any) -> str:
    """Enum members and plain strings both fingerprint by their value."""
    return str(value.value) if hasattr(value, "value") else str(value)


def step_fingerprint(action_type: Any, target_service: str) -> str:
    """Identity of one action on one target (rationale deliberately excluded)."""
    return hashlib.sha256(f"{_text(action_type)}|{target_service}".encode()).hexdigest()


def plan_fingerprint(steps: Iterable[tuple[Any, str]]) -> str:
    """Identity of an ordered plan, from its ``(action_type, target_service)`` pairs.

    Position matters: swapping two steps, or inserting one, changes the plan.
    """
    lines = [f"{i}|{_text(action)}|{target}" for i, (action, target) in enumerate(steps)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def plan_fingerprint_of(guarded: GuardedPlan) -> str:
    return plan_fingerprint((g.step.action_type, g.step.target_service) for g in guarded.steps)


def new_approval_id() -> str:
    return f"APR-{uuid4().hex[:12]}"


class ApprovalRecord(BaseModel):
    """One stored approval: a request and, once made, its single decision."""

    approval_id: str
    incident_id: str
    plan_fingerprint: str
    step_index: int = Field(ge=0)
    step_fingerprint: str

    # Snapshot of what was proposed, so the audit trail shows what a human was
    # asked to approve even after the incident's remediation rows are replaced.
    action_type: str
    target_service: str
    rationale: str = ""

    status: ApprovalState = ApprovalState.PENDING
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: datetime | None = None
    approver: str | None = None
    decision_reason: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _lifecycle_is_coherent(self) -> ApprovalRecord:
        if self.status not in _STORED_STATES:
            raise ValueError(f"'{self.status.value}' is not a storable approval status.")
        decided = self.status in _DECIDED
        if decided and (self.decided_at is None or not (self.approver or "").strip()):
            raise ValueError("A decided approval must record who decided it and when.")
        if not decided and (self.decided_at is not None or self.approver is not None):
            raise ValueError("A pending approval cannot carry a decision.")
        return self

    @property
    def is_pending(self) -> bool:
        return self.status == ApprovalState.PENDING


def decide(
    record: ApprovalRecord,
    outcome: ApprovalState,
    approver: str,
    reason: str | None = None,
    *,
    at: datetime | None = None,
) -> ApprovalRecord:
    """The one transition rule: only a PENDING approval may be approved or rejected.

    A second decision is an error, never an overwrite and never a silent no-op:
    even repeating the same decision is refused, so the first decision (and its
    approver and timestamp) is the permanent record.
    """
    if outcome not in _DECIDED:
        raise ApprovalDecisionError("A decision must be 'approved' or 'rejected'.")
    if not approver or not approver.strip():
        raise ApprovalDecisionError("A decision must name who made it.")
    if record.status != ApprovalState.PENDING:
        by = f" by {record.approver}" if record.approver else ""
        raise ApprovalDecisionError(
            f"Approval '{record.approval_id}' is already {record.status.value}{by}."
        )
    cleaned_reason = reason.strip() if reason and reason.strip() else None
    return record.model_copy(
        update={
            "status": outcome,
            "decided_at": at or datetime.now(timezone.utc),
            "approver": approver.strip(),
            "decision_reason": cleaned_reason,
        }
    )


def validate_decision_inputs(outcome: ApprovalState, approver: str) -> None:
    """Reject malformed decision arguments before anything is looked up or locked."""
    if outcome not in _DECIDED:
        raise ValueError("A decision must be 'approved' or 'rejected'.")
    if not approver or not approver.strip():
        raise ValueError("A decision must name who made it.")


def check_decidable(
    approval: ApprovalRecord,
    incident_status: IncidentLifecycleStatus,
    remediations: Sequence[IncidentRemediationRecord],
) -> None:
    """Every reason a stored approval cannot be decided right now.

    Run by both repositories inside the same lock or transaction as the decision,
    so the plan cannot be replaced between the check and the write.

    1. It must still be pending (a decision is never overwritten or repeated).
    2. The incident must still be open: a terminal incident has no execution to
       authorize, so a late decision would be meaningless.
    3. It must still describe a real step: the incident's stored plan must match
       the approval's plan fingerprint, and that step must still exist, match,
       and still require approval. Otherwise the plan was replaced and this
       approval is obsolete.
    """
    if approval.status != ApprovalState.PENDING:
        by = f" by {approval.approver}" if approval.approver else ""
        raise ApprovalDecisionError(
            f"Approval '{approval.approval_id}' is already {approval.status.value}{by}."
        )
    if incident_status.is_terminal:
        raise ApprovalDecisionError(
            f"Incident '{approval.incident_id}' is already {incident_status.value}; "
            "decisions are closed."
        )

    ordered = sorted(remediations, key=lambda r: r.step_order)
    current = plan_fingerprint((r.action_type, r.target_service) for r in ordered)
    step = next((r for r in ordered if r.step_order == approval.step_index), None)
    if (
        current != approval.plan_fingerprint
        or step is None
        or step_fingerprint(step.action_type, step.target_service) != approval.step_fingerprint
        or step.disposition != Disposition.REQUIRES_APPROVAL.value
    ):
        raise ApprovalDecisionError(
            f"Approval '{approval.approval_id}' refers to a remediation plan that has been "
            "superseded, so it can no longer authorize anything."
        )


def pending_requests_for(incident_id: str, guarded: GuardedPlan) -> list[ApprovalRecord]:
    """PENDING approval records for every step that needs a human.

    Only ``REQUIRES_APPROVAL`` steps get one: ALLOWED steps never need it and
    BLOCKED steps can never have it. Storing is idempotent (an existing approval
    for the same step is kept, never reset), so passing steps that were already
    decided is safe.
    """
    plan_fp = plan_fingerprint_of(guarded)
    # One timestamp for the whole batch: the steps of a plan are requested together, and
    # listings break the tie by step_index rather than by a clock's resolution.
    requested_at = datetime.now(timezone.utc)
    return [
        ApprovalRecord(
            approval_id=new_approval_id(),
            incident_id=incident_id,
            requested_at=requested_at,
            plan_fingerprint=plan_fp,
            step_index=index,
            step_fingerprint=step_fingerprint(g.step.action_type, g.step.target_service),
            action_type=_text(g.step.action_type),
            target_service=g.step.target_service,
            rationale=g.step.rationale,
        )
        for index, g in enumerate(guarded.steps)
        if g.disposition == Disposition.REQUIRES_APPROVAL
    ]


def apply_persisted_decisions(
    guarded: GuardedPlan, approvals: Sequence[ApprovalRecord]
) -> GuardedPlan:
    """Fold stored decisions back onto a freshly guarded plan.

    Only a step that is still awaiting approval is touched, and only by an
    approval that matches it exactly (same plan, same position, same action and
    target). Everything else is left alone, so:

    * a BLOCKED step stays blocked, whatever was approved;
    * an approval for a different plan, or another step, is ignored;
    * pending approvals change nothing.

    Uses ``GuardedPlan.approve/reject``, so the existing gate rules still apply.
    """
    plan_fp = plan_fingerprint_of(guarded)
    result = guarded
    for index, g in enumerate(guarded.steps):
        if g.disposition != Disposition.REQUIRES_APPROVAL or g.approval != ApprovalState.PENDING:
            continue
        fingerprint = step_fingerprint(g.step.action_type, g.step.target_service)
        match = next(
            (
                a
                for a in approvals
                if a.plan_fingerprint == plan_fp
                and a.step_index == index
                and a.step_fingerprint == fingerprint
                and a.status in _DECIDED
            ),
            None,
        )
        if match is None:
            continue
        assert match.approver is not None  # guaranteed by the record's own validation
        if match.status == ApprovalState.APPROVED:
            result = result.approve(index, match.approver)
        else:
            result = result.reject(index, match.approver)
    return result


__all__ = [
    "ApprovalDecisionError",
    "ApprovalRecord",
    "apply_persisted_decisions",
    "check_decidable",
    "decide",
    "new_approval_id",
    "pending_requests_for",
    "plan_fingerprint",
    "plan_fingerprint_of",
    "step_fingerprint",
    "validate_decision_inputs",
]
