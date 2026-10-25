"""PostgreSQL implementation of ``IncidentRepository``.

Takes a ``connect`` callable rather than a pool or a URL so the transaction
boundary is explicit and the class can be exercised against a fake connection:
each public method opens one connection, does its work in one transaction, and
closes it. That is deliberately simple (no pooling). Add a pool when concurrent
load makes connection setup a measured cost, not before.

Behavioural parity with ``InMemoryIncidentRepository`` is intentional, including
its conflict rules and its ordering (``created_at`` descending, ties broken by
``incident_id`` descending). One consequence worth knowing: like the in-memory
version, ``save_incident`` overwrites ``created_at`` with the incoming record's
value. Callers that re-save a record should pass the original ``created_at``.

The SQL is written for the schema in ``migrations/001_initial_incident_schema.sql``
and uses PostgreSQL features (``JSONB``, ``ANY``, ``FOR UPDATE``, ``ON CONFLICT``),
so it can only be verified against a real server; see the ``DATABASE_URL``-gated
test in ``tests/test_postgres_repository.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import ValidationError

from agentic_ai.domain.approvals import (
    ApprovalDecisionError,
    ApprovalRecord,
    check_decidable,
    decide,
    validate_decision_inputs,
)
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.persistence import (
    IncidentLifecycleStatus,
    IncidentRecord,
    IncidentRemediationRecord,
    IncidentStageRecord,
)
from agentic_ai.domain.plan_snapshots import PlanSnapshot, SnapshotIntegrityError, SnapshotStep
from agentic_ai.domain.rca import KnowledgeBaseConsistency
from agentic_ai.domain.step_executions import (
    ExecutionSource,
    StepExecutionError,
    StepExecutionRecord,
    StepExecutionStatus,
)
from agentic_ai.domain.summaries import DiagnosticSummary, VerificationSummary
from agentic_ai.persistence.repository import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    ExecutionConflictError,
    ExecutionNotFoundError,
    IncidentConflictError,
    IncidentNotFoundError,
    IncidentPersistenceError,
)

_INCIDENT_COLUMNS = (
    "incident_id, correlation_id, scenario, incident_description, severity, status, "
    "final_outcome, diagnostic_decision, diagnostic_trigger, diagnostic_reason, "
    "verification_status, verification_attempts, verification_reasons, "
    "total_duration_ms, created_at, updated_at, resolved_at"
)

_UPSERT_INCIDENT = f"""
INSERT INTO incidents ({_INCIDENT_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
ON CONFLICT (incident_id) DO UPDATE SET
    correlation_id = EXCLUDED.correlation_id,
    scenario = EXCLUDED.scenario,
    incident_description = EXCLUDED.incident_description,
    severity = EXCLUDED.severity,
    status = EXCLUDED.status,
    final_outcome = EXCLUDED.final_outcome,
    diagnostic_decision = EXCLUDED.diagnostic_decision,
    diagnostic_trigger = EXCLUDED.diagnostic_trigger,
    diagnostic_reason = EXCLUDED.diagnostic_reason,
    verification_status = EXCLUDED.verification_status,
    verification_attempts = EXCLUDED.verification_attempts,
    verification_reasons = EXCLUDED.verification_reasons,
    total_duration_ms = EXCLUDED.total_duration_ms,
    created_at = EXCLUDED.created_at,
    updated_at = EXCLUDED.updated_at,
    resolved_at = EXCLUDED.resolved_at
"""

_INSERT_STAGE = """
INSERT INTO incident_stages
    (incident_id, stage_name, status, start_time, end_time, duration_ms,
     error_category, error_message, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
"""

_INSERT_REMEDIATION = """
INSERT INTO incident_remediations
    (incident_id, step_order, action_type, target_service, disposition,
     execution_status, parameters, output)
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
"""

_SELECT_ORDER = "ORDER BY created_at DESC, incident_id DESC"

_APPROVAL_COLUMNS = (
    "approval_id, incident_id, plan_fingerprint, step_index, step_fingerprint, "
    "action_type, target_service, rationale, status, requested_at, decided_at, "
    "approver, decision_reason"
)

_INSERT_APPROVAL = f"""
INSERT INTO incident_approvals ({_APPROVAL_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (incident_id, plan_fingerprint, step_index) DO NOTHING
"""

# Per incident: oldest first, steps of one plan in order. The global queue has no
# step_index in its ordering so that it matches the partial index exactly.
_APPROVAL_ORDER = "ORDER BY requested_at, step_index, approval_id"
_PENDING_QUEUE_ORDER = "ORDER BY requested_at, approval_id"

# A closed incident has no execution left to authorize, so its approvals are not decidable.
_TERMINAL_STATUSES = [s.value for s in IncidentLifecycleStatus if s.is_terminal]

_SNAPSHOT_COLUMNS = (
    "snapshot_id, incident_id, plan_fingerprint, snapshot_version, checksum, content, created_at"
)

_INSERT_SNAPSHOT = f"""
INSERT INTO incident_plan_snapshots ({_SNAPSHOT_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (incident_id, plan_fingerprint, checksum) DO NOTHING
"""

_EXECUTION_COLUMNS = (
    "execution_id, incident_id, plan_fingerprint, step_index, step_fingerprint, action_type, "
    "target_service, status, source, started_at, finished_at, output, error"
)

_CLAIM_EXECUTION = f"""
INSERT INTO incident_step_executions ({_EXECUTION_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (incident_id, plan_fingerprint, step_index) DO NOTHING
"""


def _pending_filter(
    action_type: str | None, target_service: str | None, include_closed: bool
) -> tuple[str, list[Any]]:
    """WHERE clause for the cross-incident pending queue (keeps the partial-index predicate)."""
    clauses = ["status = 'pending'"]
    params: list[Any] = []
    if action_type is not None:
        clauses.append("action_type = %s")
        params.append(action_type)
    if target_service is not None:
        clauses.append("target_service = %s")
        params.append(target_service)
    if not include_closed:
        clauses.append(
            "EXISTS (SELECT 1 FROM incidents i "
            "WHERE i.incident_id = incident_approvals.incident_id AND i.status <> ALL(%s))"
        )
        params.append(_TERMINAL_STATUSES)
    return " AND ".join(clauses), params


def _json_value(value: Any) -> Any:
    """JSONB comes back parsed from psycopg; tolerate drivers that return text."""
    if isinstance(value, (str, bytes)):
        return json.loads(value)
    return value


class PostgresIncidentRepository:
    """``IncidentRepository`` backed by PostgreSQL."""

    def __init__(self, connect: Callable[[], Any]) -> None:
        self._connect = connect

    @classmethod
    def from_url(cls, database_url: str) -> PostgresIncidentRepository:
        """Build a repository that connects with ``psycopg`` on every operation."""
        import psycopg

        return cls(lambda: psycopg.connect(database_url))

    # -- writes ---------------------------------------------------------------

    def save_incident(self, record: IncidentRecord) -> IncidentRecord:
        def work(cur: Any) -> None:
            cur.execute(
                "SELECT status, final_outcome, severity FROM incidents "
                "WHERE incident_id = %s FOR UPDATE",
                (record.incident_id,),
            )
            existing = cur.fetchone()
            if existing is not None:
                self._check_terminal_conflict(record, existing)

            diagnostic = record.diagnostic
            verification = record.verification
            cur.execute(
                _UPSERT_INCIDENT,
                (
                    record.incident_id,
                    record.correlation_id,
                    record.scenario,
                    record.incident_description,
                    record.severity.value,
                    record.status.value,
                    record.final_outcome,
                    diagnostic.decision if diagnostic else None,
                    diagnostic.trigger_type if diagnostic else None,
                    diagnostic.primary_reason if diagnostic else None,
                    verification.status if verification else None,
                    verification.attempts if verification else 0,
                    json.dumps(verification.reasons if verification else []),
                    record.total_duration_ms,
                    record.created_at,
                    record.updated_at,
                    record.resolved_at,
                ),
            )

            # Children are replaced wholesale: the record is the source of truth.
            cur.execute(
                "DELETE FROM incident_stages WHERE incident_id = %s",
                (record.incident_id,),
            )
            cur.execute(
                "DELETE FROM incident_remediations WHERE incident_id = %s",
                (record.incident_id,),
            )
            for stage in record.stages:
                cur.execute(
                    _INSERT_STAGE,
                    (
                        record.incident_id,
                        stage.stage_name,
                        stage.status,
                        stage.start_time,
                        stage.end_time,
                        stage.duration_ms,
                        stage.error_category,
                        stage.error_message,
                        json.dumps(stage.metadata),
                    ),
                )
            for step in record.remediations:
                cur.execute(
                    _INSERT_REMEDIATION,
                    (
                        record.incident_id,
                        step.step_order,
                        step.action_type,
                        step.target_service,
                        step.disposition,
                        step.execution_status,
                        json.dumps(step.parameters),
                        step.output,
                    ),
                )

        self._in_transaction(work)
        return record

    @staticmethod
    def _check_terminal_conflict(record: IncidentRecord, existing: Any) -> None:
        """Same rules as the in-memory repository: a terminal record is frozen."""
        existing_status = IncidentLifecycleStatus(existing[0])
        existing_outcome = existing[1]
        existing_severity = Severity(existing[2])
        if not existing_status.is_terminal:
            return

        if not record.status.is_terminal:
            raise IncidentConflictError(
                f"Cannot revert terminal incident '{record.incident_id}' "
                f"(status={existing_status.value}) to non-terminal status "
                f"'{record.status.value}'."
            )
        if existing_outcome != record.final_outcome or existing_severity != record.severity:
            raise IncidentConflictError(
                f"Cannot overwrite terminal incident '{record.incident_id}' with conflicting "
                f"outcome ({existing_outcome} -> {record.final_outcome}) "
                f"or severity ({existing_severity.value} -> {record.severity.value})."
            )

    # -- reads ----------------------------------------------------------------

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        def work(cur: Any) -> IncidentRecord | None:
            cur.execute(
                f"SELECT {_INCIDENT_COLUMNS} FROM incidents WHERE incident_id = %s",
                (incident_id,),
            )
            rows = cur.fetchall()
            records = self._hydrate(cur, rows)
            return records[0] if records else None

        return self._in_transaction(work)

    def get_by_correlation_id(self, correlation_id: str) -> list[IncidentRecord]:
        def work(cur: Any) -> list[IncidentRecord]:
            cur.execute(
                f"SELECT {_INCIDENT_COLUMNS} FROM incidents "
                f"WHERE correlation_id = %s {_SELECT_ORDER}",
                (correlation_id,),
            )
            return self._hydrate(cur, cur.fetchall())

        return self._in_transaction(work)

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: IncidentLifecycleStatus | None = None,
    ) -> list[IncidentRecord]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        def work(cur: Any) -> list[IncidentRecord]:
            if status is None:
                cur.execute(
                    f"SELECT {_INCIDENT_COLUMNS} FROM incidents {_SELECT_ORDER} LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            else:
                cur.execute(
                    f"SELECT {_INCIDENT_COLUMNS} FROM incidents WHERE status = %s "
                    f"{_SELECT_ORDER} LIMIT %s OFFSET %s",
                    (status.value, limit, offset),
                )
            return self._hydrate(cur, cur.fetchall())

        return self._in_transaction(work)

    def count(self, status: IncidentLifecycleStatus | None = None) -> int:
        def work(cur: Any) -> int:
            if status is None:
                cur.execute("SELECT COUNT(*) FROM incidents")
            else:
                cur.execute("SELECT COUNT(*) FROM incidents WHERE status = %s", (status.value,))
            return int(cur.fetchone()[0])

        return self._in_transaction(work)

    # -- approvals ------------------------------------------------------------

    def request_approvals(self, approvals: Sequence[ApprovalRecord]) -> list[ApprovalRecord]:
        def work(cur: Any) -> list[ApprovalRecord]:
            stored: list[ApprovalRecord] = []
            for approval in approvals:
                # KEY SHARE: the incident cannot be deleted while we attach to it.
                cur.execute(
                    "SELECT 1 FROM incidents WHERE incident_id = %s FOR KEY SHARE",
                    (approval.incident_id,),
                )
                if cur.fetchone() is None:
                    raise IncidentNotFoundError(approval.incident_id)

                cur.execute(
                    _INSERT_APPROVAL,
                    (
                        approval.approval_id,
                        approval.incident_id,
                        approval.plan_fingerprint,
                        approval.step_index,
                        approval.step_fingerprint,
                        approval.action_type,
                        approval.target_service,
                        approval.rationale,
                        approval.status.value,
                        approval.requested_at,
                        approval.decided_at,
                        approval.approver,
                        approval.decision_reason,
                    ),
                )
                # Re-read: if the step already had an approval, that one wins untouched.
                cur.execute(
                    f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals "
                    "WHERE incident_id = %s AND plan_fingerprint = %s AND step_index = %s",
                    (approval.incident_id, approval.plan_fingerprint, approval.step_index),
                )
                stored.append(self._row_to_approval(cur.fetchone()))
            return stored

        return self._in_transaction(work)

    def get_approval(self, incident_id: str, approval_id: str) -> ApprovalRecord | None:
        def work(cur: Any) -> ApprovalRecord | None:
            cur.execute(
                f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals "
                "WHERE approval_id = %s AND incident_id = %s",
                (approval_id, incident_id),
            )
            row = cur.fetchone()
            return None if row is None else self._row_to_approval(row)

        return self._in_transaction(work)

    def list_approvals(
        self, incident_id: str, status: ApprovalState | None = None
    ) -> list[ApprovalRecord]:
        def work(cur: Any) -> list[ApprovalRecord]:
            if status is None:
                cur.execute(
                    f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals "
                    f"WHERE incident_id = %s {_APPROVAL_ORDER}",
                    (incident_id,),
                )
            else:
                cur.execute(
                    f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals "
                    f"WHERE incident_id = %s AND status = %s {_APPROVAL_ORDER}",
                    (incident_id, status.value),
                )
            return [self._row_to_approval(r) for r in cur.fetchall()]

        return self._in_transaction(work)

    def list_pending_approvals(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        action_type: str | None = None,
        target_service: str | None = None,
        include_closed: bool = False,
    ) -> list[ApprovalRecord]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        where, params = _pending_filter(action_type, target_service, include_closed)

        def work(cur: Any) -> list[ApprovalRecord]:
            cur.execute(
                f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals WHERE {where} "
                f"{_PENDING_QUEUE_ORDER} LIMIT %s OFFSET %s",
                (*params, limit, offset),
            )
            return [self._row_to_approval(r) for r in cur.fetchall()]

        return self._in_transaction(work)

    def count_pending_approvals(
        self,
        *,
        action_type: str | None = None,
        target_service: str | None = None,
        include_closed: bool = False,
    ) -> int:
        where, params = _pending_filter(action_type, target_service, include_closed)

        def work(cur: Any) -> int:
            cur.execute(f"SELECT COUNT(*) FROM incident_approvals WHERE {where}", tuple(params))
            return int(cur.fetchone()[0])

        return self._in_transaction(work)

    def decide_approval(
        self,
        incident_id: str,
        approval_id: str,
        outcome: ApprovalState,
        approver: str,
        reason: str | None = None,
    ) -> ApprovalRecord:
        validate_decision_inputs(outcome, approver)

        def work(cur: Any) -> ApprovalRecord:
            # FOR SHARE on the incident blocks a concurrent save_incident (which takes
            # FOR UPDATE on the same row), so the plan we check cannot be replaced
            # before our write commits. Concurrent decisions share this lock freely.
            cur.execute(
                "SELECT status FROM incidents WHERE incident_id = %s FOR SHARE",
                (incident_id,),
            )
            incident_row = cur.fetchone()
            if incident_row is None:
                raise IncidentNotFoundError(incident_id)

            # FOR UPDATE serializes competing decisions: the loser waits, then re-reads
            # the winner's committed row and fails the pending check below.
            cur.execute(
                f"SELECT {_APPROVAL_COLUMNS} FROM incident_approvals "
                "WHERE approval_id = %s AND incident_id = %s FOR UPDATE",
                (approval_id, incident_id),
            )
            approval_row = cur.fetchone()
            if approval_row is None:
                raise ApprovalNotFoundError(incident_id, approval_id)
            approval = self._row_to_approval(approval_row)

            cur.execute(
                "SELECT step_order, action_type, target_service, disposition, "
                "execution_status, parameters, output FROM incident_remediations "
                "WHERE incident_id = %s ORDER BY step_order",
                (incident_id,),
            )
            remediations = [
                IncidentRemediationRecord(
                    step_order=r[0],
                    action_type=r[1],
                    target_service=r[2],
                    disposition=r[3],
                    execution_status=r[4],
                    parameters=_json_value(r[5]) or {},
                    output=r[6],
                )
                for r in cur.fetchall()
            ]

            try:
                check_decidable(approval, IncidentLifecycleStatus(incident_row[0]), remediations)
                decided = decide(approval, outcome, approver, reason)
            except ApprovalDecisionError as exc:
                raise ApprovalConflictError(str(exc)) from exc

            # Conditional on still being pending: a second line of defence behind the lock.
            cur.execute(
                "UPDATE incident_approvals SET status = %s, decided_at = %s, approver = %s, "
                "decision_reason = %s WHERE approval_id = %s AND status = 'pending'",
                (
                    decided.status.value,
                    decided.decided_at,
                    decided.approver,
                    decided.decision_reason,
                    approval_id,
                ),
            )
            if cur.rowcount != 1:
                raise ApprovalConflictError(
                    f"Approval '{approval_id}' was decided by a concurrent request."
                )
            return decided

        return self._in_transaction(work)

    @staticmethod
    def _row_to_approval(row: Any) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row[0],
            incident_id=row[1],
            plan_fingerprint=row[2],
            step_index=row[3],
            step_fingerprint=row[4],
            action_type=row[5],
            target_service=row[6],
            rationale=row[7],
            status=ApprovalState(row[8]),
            requested_at=row[9],
            decided_at=row[10],
            approver=row[11],
            decision_reason=row[12],
        )

    # -- plan snapshots ---------------------------------------------------------

    def save_plan_snapshot(self, snapshot: PlanSnapshot) -> PlanSnapshot:
        content = {
            "steps": [s.model_dump() for s in snapshot.steps],
            "root_cause": snapshot.root_cause,
            "root_cause_confidence": snapshot.root_cause_confidence,
            "knowledge_base_consistency": snapshot.knowledge_base_consistency.value,
            "investigated_services": snapshot.investigated_services,
        }

        def work(cur: Any) -> PlanSnapshot:
            cur.execute(
                "SELECT 1 FROM incidents WHERE incident_id = %s FOR KEY SHARE",
                (snapshot.incident_id,),
            )
            if cur.fetchone() is None:
                raise IncidentNotFoundError(snapshot.incident_id)
            cur.execute(
                _INSERT_SNAPSHOT,
                (
                    snapshot.snapshot_id,
                    snapshot.incident_id,
                    snapshot.plan_fingerprint,
                    snapshot.version,
                    snapshot.checksum,
                    json.dumps(content),
                    snapshot.created_at,
                ),
            )
            # Re-read: if this exact presentation was already stored, the original wins.
            cur.execute(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM incident_plan_snapshots "
                "WHERE incident_id = %s AND plan_fingerprint = %s AND checksum = %s",
                (snapshot.incident_id, snapshot.plan_fingerprint, snapshot.checksum),
            )
            return self._row_to_snapshot(cur.fetchone())

        return self._in_transaction(work)

    def get_plan_snapshot(self, incident_id: str, plan_fingerprint: str) -> PlanSnapshot | None:
        def work(cur: Any) -> PlanSnapshot | None:
            cur.execute(
                f"SELECT {_SNAPSHOT_COLUMNS} FROM incident_plan_snapshots "
                "WHERE incident_id = %s AND plan_fingerprint = %s "
                "ORDER BY snapshot_seq DESC LIMIT 1",
                (incident_id, plan_fingerprint),
            )
            row = cur.fetchone()
            if row is None:
                return None
            snapshot = self._row_to_snapshot(row)
            snapshot.verify()  # never hand back a plan that fails its own checks
            return snapshot

        return self._in_transaction(work)

    @staticmethod
    def _row_to_snapshot(row: Any) -> PlanSnapshot:
        try:
            content = _json_value(row[5])
            return PlanSnapshot(
                snapshot_id=row[0],
                incident_id=row[1],
                plan_fingerprint=row[2],
                version=row[3],
                steps=[SnapshotStep(**step) for step in content["steps"]],
                root_cause=content.get("root_cause", ""),
                root_cause_confidence=content["root_cause_confidence"],
                knowledge_base_consistency=KnowledgeBaseConsistency(
                    content["knowledge_base_consistency"]
                ),
                investigated_services=list(content["investigated_services"]),
                checksum=row[4],
                created_at=row[6],
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise SnapshotIntegrityError("Stored plan snapshot content is malformed.") from exc

    # -- execution ledger ---------------------------------------------------------

    def claim_step_execution(self, record: StepExecutionRecord) -> tuple[StepExecutionRecord, bool]:
        def work(cur: Any) -> tuple[StepExecutionRecord, bool]:
            # A concurrent claim blocks on the unique index until the first one commits,
            # then does nothing: exactly one caller gets rowcount 1 for a given step.
            cur.execute(
                _CLAIM_EXECUTION,
                (
                    record.execution_id,
                    record.incident_id,
                    record.plan_fingerprint,
                    record.step_index,
                    record.step_fingerprint,
                    record.action_type,
                    record.target_service,
                    record.status.value,
                    record.source.value,
                    record.started_at,
                    record.finished_at,
                    record.output,
                    record.error,
                ),
            )
            claimed = cur.rowcount == 1
            cur.execute(
                f"SELECT {_EXECUTION_COLUMNS} FROM incident_step_executions "
                "WHERE incident_id = %s AND plan_fingerprint = %s AND step_index = %s",
                (record.incident_id, record.plan_fingerprint, record.step_index),
            )
            return self._row_to_execution(cur.fetchone()), claimed

        return self._in_transaction(work)

    def complete_step_execution(
        self,
        incident_id: str,
        execution_id: str,
        status: StepExecutionStatus,
        output: str | None = None,
        error: str | None = None,
    ) -> StepExecutionRecord:
        def work(cur: Any) -> StepExecutionRecord:
            cur.execute(
                f"SELECT {_EXECUTION_COLUMNS} FROM incident_step_executions "
                "WHERE incident_id = %s AND execution_id = %s FOR UPDATE",
                (incident_id, execution_id),
            )
            row = cur.fetchone()
            if row is None:
                raise ExecutionNotFoundError(incident_id, execution_id)
            try:
                finished = self._row_to_execution(row).finish(status, output, error)
            except StepExecutionError as exc:
                raise ExecutionConflictError(str(exc)) from exc

            cur.execute(
                "UPDATE incident_step_executions SET status = %s, finished_at = %s, "
                "output = %s, error = %s WHERE execution_id = %s AND status = 'started'",
                (finished.status.value, finished.finished_at, output, error, execution_id),
            )
            if cur.rowcount != 1:
                raise ExecutionConflictError(
                    f"Execution '{execution_id}' was finished by a concurrent request."
                )
            return finished

        return self._in_transaction(work)

    def list_step_executions(
        self, incident_id: str, plan_fingerprint: str | None = None
    ) -> list[StepExecutionRecord]:
        def work(cur: Any) -> list[StepExecutionRecord]:
            if plan_fingerprint is None:
                cur.execute(
                    f"SELECT {_EXECUTION_COLUMNS} FROM incident_step_executions "
                    "WHERE incident_id = %s ORDER BY step_index, started_at",
                    (incident_id,),
                )
            else:
                cur.execute(
                    f"SELECT {_EXECUTION_COLUMNS} FROM incident_step_executions "
                    "WHERE incident_id = %s AND plan_fingerprint = %s "
                    "ORDER BY step_index, started_at",
                    (incident_id, plan_fingerprint),
                )
            return [self._row_to_execution(r) for r in cur.fetchall()]

        return self._in_transaction(work)

    @staticmethod
    def _row_to_execution(row: Any) -> StepExecutionRecord:
        return StepExecutionRecord(
            execution_id=row[0],
            incident_id=row[1],
            plan_fingerprint=row[2],
            step_index=row[3],
            step_fingerprint=row[4],
            action_type=row[5],
            target_service=row[6],
            status=StepExecutionStatus(row[7]),
            source=ExecutionSource(row[8]),
            started_at=row[9],
            finished_at=row[10],
            output=row[11],
            error=row[12],
        )

    # -- resume lock ----------------------------------------------------------------

    @contextmanager
    def resume_lock(self, incident_id: str) -> Iterator[bool]:
        """A session-level advisory lock, so a crashed resumer cannot leave it held.

        It lives on its own connection for the whole ``with`` block: the operations done
        inside use their own short transactions on other connections.
        """
        key = f"resume:{incident_id}"
        connection = self._connect()
        acquired = False
        try:
            try:
                cursor = connection.cursor()
                try:
                    cursor.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,))
                    acquired = bool(cursor.fetchone()[0])
                finally:
                    cursor.close()
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise IncidentPersistenceError(f"Could not take the resume lock: {exc}") from exc
            yield acquired
        finally:
            try:
                if acquired:
                    cursor = connection.cursor()
                    try:
                        cursor.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,))
                    finally:
                        cursor.close()
                    connection.commit()
            except Exception:
                # Best effort: closing the connection below releases a session lock anyway,
                # and a failure to unlock must not mask the outcome of the resume itself.
                pass
            finally:
                connection.close()

    # -- internals ------------------------------------------------------------

    def _in_transaction(self, work: Callable[[Any], Any]) -> Any:
        """Run ``work(cursor)`` in one transaction: commit on success, else roll back.

        Domain errors (conflicts, bad arguments) pass through untouched. Anything
        else from the driver is wrapped so callers see one persistence error type
        and never a raw driver exception.
        """
        connection = self._connect()
        try:
            cursor = connection.cursor()
            try:
                result = work(cursor)
            finally:
                cursor.close()
            connection.commit()
            return result
        except (IncidentPersistenceError, SnapshotIntegrityError, ValueError):
            # Domain outcomes (conflict, not found) and bad arguments are not driver
            # failures: roll back and let the caller see them as they are.
            connection.rollback()
            raise
        except Exception as exc:
            connection.rollback()
            raise IncidentPersistenceError(f"Database operation failed: {exc}") from exc
        finally:
            connection.close()

    def _hydrate(self, cur: Any, rows: list[Any]) -> list[IncidentRecord]:
        """Turn incident rows into records, loading all children in two queries."""
        if not rows:
            return []

        ids = [row[0] for row in rows]
        stages_by_incident: dict[str, list[IncidentStageRecord]] = {i: [] for i in ids}
        remediations_by_incident: dict[str, list[IncidentRemediationRecord]] = {i: [] for i in ids}

        cur.execute(
            "SELECT incident_id, stage_name, status, start_time, end_time, duration_ms, "
            "error_category, error_message, metadata FROM incident_stages "
            "WHERE incident_id = ANY(%s) ORDER BY stage_id",
            (ids,),
        )
        for r in cur.fetchall():
            stages_by_incident[r[0]].append(
                IncidentStageRecord(
                    stage_name=r[1],
                    status=r[2],
                    start_time=r[3],
                    end_time=r[4],
                    duration_ms=r[5],
                    error_category=r[6],
                    error_message=r[7],
                    metadata=_json_value(r[8]) or {},
                )
            )

        cur.execute(
            "SELECT incident_id, step_order, action_type, target_service, disposition, "
            "execution_status, parameters, output FROM incident_remediations "
            "WHERE incident_id = ANY(%s) ORDER BY incident_id, step_order",
            (ids,),
        )
        for r in cur.fetchall():
            remediations_by_incident[r[0]].append(
                IncidentRemediationRecord(
                    step_order=r[1],
                    action_type=r[2],
                    target_service=r[3],
                    disposition=r[4],
                    execution_status=r[5],
                    parameters=_json_value(r[6]) or {},
                    output=r[7],
                )
            )

        return [
            self._row_to_record(row, stages_by_incident[row[0]], remediations_by_incident[row[0]])
            for row in rows
        ]

    @staticmethod
    def _row_to_record(
        row: Any,
        stages: list[IncidentStageRecord],
        remediations: list[IncidentRemediationRecord],
    ) -> IncidentRecord:
        diagnostic = None
        if row[7] is not None:
            diagnostic = DiagnosticSummary(
                decision=row[7], trigger_type=row[8] or "", primary_reason=row[9] or ""
            )
        verification = None
        if row[10] is not None:
            verification = VerificationSummary(
                status=row[10],
                attempts=row[11],
                reasons=list(_json_value(row[12]) or []),
            )
        return IncidentRecord(
            incident_id=row[0],
            correlation_id=row[1],
            scenario=row[2],
            incident_description=row[3],
            severity=Severity(row[4]),
            status=IncidentLifecycleStatus(row[5]),
            final_outcome=row[6],
            diagnostic=diagnostic,
            verification=verification,
            stages=stages,
            remediations=remediations,
            total_duration_ms=row[13],
            created_at=row[14],
            updated_at=row[15],
            resolved_at=row[16],
        )


__all__ = ["PostgresIncidentRepository"]
