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
from collections.abc import Callable
from typing import Any

from agentic_ai.domain.incident import Severity
from agentic_ai.domain.persistence import (
    IncidentLifecycleStatus,
    IncidentRecord,
    IncidentRemediationRecord,
    IncidentStageRecord,
)
from agentic_ai.domain.summaries import DiagnosticSummary, VerificationSummary
from agentic_ai.persistence.repository import (
    IncidentConflictError,
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
        except (IncidentConflictError, ValueError):
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
