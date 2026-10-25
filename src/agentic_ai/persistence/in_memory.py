"""Thread-safe in-memory implementation of IncidentRepository (Phase 16).

Provides a deterministic, zero-dependency repository for unit tests, local
demonstrations, and offline validation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Sequence

from agentic_ai.domain.approvals import (
    ApprovalDecisionError,
    ApprovalRecord,
    check_decidable,
    decide,
    validate_decision_inputs,
)
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRecord
from agentic_ai.domain.plan_snapshots import PlanSnapshot
from agentic_ai.domain.step_executions import (
    StepExecutionError,
    StepExecutionRecord,
    StepExecutionStatus,
)
from agentic_ai.persistence.repository import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    ExecutionConflictError,
    ExecutionNotFoundError,
    IncidentConflictError,
    IncidentNotFoundError,
    IncidentRepository,
)


class InMemoryIncidentRepository:
    """Thread-safe in-memory store satisfying the IncidentRepository protocol."""

    def __init__(self, initial_records: Sequence[IncidentRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._storage: dict[str, IncidentRecord] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        # (incident, plan fingerprint, step index) -> approval_id: what makes
        # registering approvals idempotent, mirroring the PostgreSQL unique constraint.
        self._approval_keys: dict[tuple[str, str, int], str] = {}
        # Immutable plan snapshots per (incident, plan fingerprint), in the order stored.
        self._snapshots: dict[tuple[str, str], list[PlanSnapshot]] = {}
        # The execution ledger: one record per (incident, plan fingerprint, step index).
        self._executions: dict[tuple[str, str, int], StepExecutionRecord] = {}
        self._resume_locks: dict[str, threading.Lock] = {}
        if initial_records:
            for rec in initial_records:
                self.save_incident(rec)

    def save_incident(self, record: IncidentRecord) -> IncidentRecord:
        """Persist or update an incident record with terminal conflict protection.

        Parameters
        ----------
        record : IncidentRecord
            The incident record to store.

        Returns
        -------
        IncidentRecord
            The stored record.

        Raises
        ------
        IncidentConflictError
            If attempting to alter a terminal record's outcome, severity, or status.
        """
        with self._lock:
            existing = self._storage.get(record.incident_id)
            if existing is not None:
                # Check terminal record immutability rules
                if existing.status.is_terminal:
                    if not record.status.is_terminal:
                        raise IncidentConflictError(
                            f"Cannot revert terminal incident '{record.incident_id}' "
                            f"(status={existing.status.value}) to non-terminal status '{record.status.value}'."
                        )
                    if (
                        existing.final_outcome != record.final_outcome
                        or existing.severity != record.severity
                    ):
                        raise IncidentConflictError(
                            f"Cannot overwrite terminal incident '{record.incident_id}' with conflicting "
                            f"outcome ({existing.final_outcome} -> {record.final_outcome}) "
                            f"or severity ({existing.severity.value} -> {record.severity.value})."
                        )

            self._storage[record.incident_id] = record
            return record

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        """Retrieve an incident record by its unique ID."""
        with self._lock:
            return self._storage.get(incident_id)

    def get_by_correlation_id(self, correlation_id: str) -> list[IncidentRecord]:
        """Retrieve all incidents matching correlation_id, sorted by created_at DESC."""
        with self._lock:
            matching = [
                rec for rec in self._storage.values() if rec.correlation_id == correlation_id
            ]
            matching.sort(key=lambda r: (r.created_at, r.incident_id), reverse=True)
            return matching

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: IncidentLifecycleStatus | None = None,
    ) -> list[IncidentRecord]:
        """Retrieve a paginated slice of records, sorted by created_at DESC."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        with self._lock:
            records = list(self._storage.values())
            if status is not None:
                records = [r for r in records if r.status == status]

            records.sort(key=lambda r: (r.created_at, r.incident_id), reverse=True)
            return records[offset : offset + limit]

    def count(self, status: IncidentLifecycleStatus | None = None) -> int:
        """Return the count of incidents, optionally filtered by status."""
        with self._lock:
            if status is None:
                return len(self._storage)
            return sum(1 for r in self._storage.values() if r.status == status)

    # -- approvals ------------------------------------------------------------

    def request_approvals(self, approvals: Sequence[ApprovalRecord]) -> list[ApprovalRecord]:
        """Store pending approvals idempotently; an existing one is kept untouched."""
        with self._lock:
            # Validate every incident first so a bad batch stores nothing, like a
            # rolled-back transaction would.
            for approval in approvals:
                if approval.incident_id not in self._storage:
                    raise IncidentNotFoundError(approval.incident_id)

            stored: list[ApprovalRecord] = []
            for approval in approvals:
                key = (approval.incident_id, approval.plan_fingerprint, approval.step_index)
                existing_id = self._approval_keys.get(key)
                if existing_id is not None:
                    stored.append(self._approvals[existing_id])
                    continue
                self._approvals[approval.approval_id] = approval
                self._approval_keys[key] = approval.approval_id
                stored.append(approval)
            return stored

    def get_approval(self, incident_id: str, approval_id: str) -> ApprovalRecord | None:
        with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None or approval.incident_id != incident_id:
                return None
            return approval

    def list_approvals(
        self, incident_id: str, status: ApprovalState | None = None
    ) -> list[ApprovalRecord]:
        with self._lock:
            matching = [
                a
                for a in self._approvals.values()
                if a.incident_id == incident_id and (status is None or a.status == status)
            ]
            matching.sort(key=lambda a: (a.requested_at, a.step_index, a.approval_id))
            return matching

    def list_pending_approvals(self, limit: int = 50) -> list[ApprovalRecord]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._lock:
            pending = [a for a in self._approvals.values() if a.is_pending]
            pending.sort(key=lambda a: (a.requested_at, a.approval_id))
            return pending[:limit]

    def decide_approval(
        self,
        incident_id: str,
        approval_id: str,
        outcome: ApprovalState,
        approver: str,
        reason: str | None = None,
    ) -> ApprovalRecord:
        validate_decision_inputs(outcome, approver)
        with self._lock:
            incident = self._storage.get(incident_id)
            if incident is None:
                raise IncidentNotFoundError(incident_id)
            approval = self.get_approval(incident_id, approval_id)
            if approval is None:
                raise ApprovalNotFoundError(incident_id, approval_id)

            try:
                check_decidable(approval, incident.status, incident.remediations)
                decided = decide(approval, outcome, approver, reason)
            except ApprovalDecisionError as exc:
                raise ApprovalConflictError(str(exc)) from exc

            self._approvals[approval_id] = decided
            return decided

    # -- plan snapshots ---------------------------------------------------------

    def save_plan_snapshot(self, snapshot: PlanSnapshot) -> PlanSnapshot:
        with self._lock:
            if snapshot.incident_id not in self._storage:
                raise IncidentNotFoundError(snapshot.incident_id)
            key = (snapshot.incident_id, snapshot.plan_fingerprint)
            existing = self._snapshots.setdefault(key, [])
            for stored in existing:
                if stored.checksum == snapshot.checksum:
                    return stored  # the same presentation again: keep the original
            existing.append(snapshot)
            return snapshot

    def get_plan_snapshot(self, incident_id: str, plan_fingerprint: str) -> PlanSnapshot | None:
        with self._lock:
            stored = self._snapshots.get((incident_id, plan_fingerprint))
            if not stored:
                return None
            # The most recently *stored*, by insertion order, not by timestamp: two
            # presentations inside one clock tick must not tie.
            latest = stored[-1]
        latest.verify()  # raises SnapshotIntegrityError rather than returning a bad plan
        return latest

    # -- execution ledger -------------------------------------------------------

    def claim_step_execution(self, record: StepExecutionRecord) -> tuple[StepExecutionRecord, bool]:
        key = (record.incident_id, record.plan_fingerprint, record.step_index)
        with self._lock:
            existing = self._executions.get(key)
            if existing is not None:
                return existing, False
            self._executions[key] = record
            return record, True

    def complete_step_execution(
        self,
        incident_id: str,
        execution_id: str,
        status: StepExecutionStatus,
        output: str | None = None,
        error: str | None = None,
    ) -> StepExecutionRecord:
        with self._lock:
            for key, record in self._executions.items():
                if record.incident_id == incident_id and record.execution_id == execution_id:
                    try:
                        finished = record.finish(status, output, error)
                    except StepExecutionError as exc:
                        raise ExecutionConflictError(str(exc)) from exc
                    self._executions[key] = finished
                    return finished
            raise ExecutionNotFoundError(incident_id, execution_id)

    def list_step_executions(
        self, incident_id: str, plan_fingerprint: str | None = None
    ) -> list[StepExecutionRecord]:
        with self._lock:
            matching = [
                r
                for r in self._executions.values()
                if r.incident_id == incident_id
                and (plan_fingerprint is None or r.plan_fingerprint == plan_fingerprint)
            ]
            matching.sort(key=lambda r: (r.step_index, r.started_at))
            return matching

    # -- resume lock ------------------------------------------------------------

    @contextmanager
    def resume_lock(self, incident_id: str) -> Iterator[bool]:
        with self._lock:
            lock = self._resume_locks.setdefault(incident_id, threading.Lock())
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    def clear(self) -> None:
        """Reset internal storage (useful for test fixtures)."""
        with self._lock:
            self._storage.clear()
            self._approvals.clear()
            self._approval_keys.clear()
            self._snapshots.clear()
            self._executions.clear()
            self._resume_locks.clear()


# Ensure implementation strictly satisfies protocol at runtime
assert issubclass(InMemoryIncidentRepository, IncidentRepository) or isinstance(
    InMemoryIncidentRepository(), IncidentRepository
)

__all__ = ["InMemoryIncidentRepository"]
