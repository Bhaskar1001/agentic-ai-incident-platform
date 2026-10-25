"""Contract tests for plan snapshots, the execution ledger, the resume lock and the pending
queue. Each runs against every available backend; the PostgreSQL leg needs a real server and
is skipped (visibly) without one. PostgreSQL-only tests at the end check the schema itself.
"""

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from agentic_ai.config import Settings
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.domain.plan_snapshots import SnapshotIntegrityError, build_snapshot
from agentic_ai.domain.step_executions import (
    ExecutionSource,
    StepExecutionRecord,
    StepExecutionStatus,
    new_execution_id,
)
from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from agentic_ai.persistence.repository import (
    ExecutionConflictError,
    ExecutionLedger,
    ExecutionNotFoundError,
    IncidentNotFoundError,
    PlanSnapshotRepository,
    ResumeLocking,
)
from tests.approval_helpers import (
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    analysis,
    guarded,
    new_incident_id,
    saved_incident,
    state_for,
)
from tests.test_approval_repository import pg_connection  # noqa: F401  (a pytest fixture)

SUCCEEDED, FAILED = StepExecutionStatus.SUCCEEDED, StepExecutionStatus.FAILED


def _snapshot_for(record, plan, *, confidence=0.9, services=("payment",)):
    return build_snapshot(record.incident_id, plan, analysis(confidence), list(services))


def _claim_record(
    incident_id: str, *, step_index=0, plan_fp="a" * 64, source=ExecutionSource.WORKFLOW
):
    return StepExecutionRecord(
        execution_id=new_execution_id(),
        incident_id=incident_id,
        plan_fingerprint=plan_fp,
        step_index=step_index,
        step_fingerprint="b" * 64,
        action_type="rollback_deployment",
        target_service="payment",
        source=source,
    )


def test_the_store_satisfies_the_new_protocols(incident_store) -> None:
    assert isinstance(incident_store, PlanSnapshotRepository)
    assert isinstance(incident_store, ExecutionLedger)
    assert isinstance(incident_store, ResumeLocking)


# =============================================================================
# Plan snapshots
# =============================================================================


def test_a_snapshot_round_trips(incident_store) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    snap = _snapshot_for(record, plan)

    saved = incident_store.save_plan_snapshot(snap)
    loaded = incident_store.get_plan_snapshot(record.incident_id, snap.plan_fingerprint)

    assert saved == snap and loaded == snap
    loaded.verify()


def test_saving_the_same_presentation_again_keeps_the_original(incident_store) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    first = incident_store.save_plan_snapshot(_snapshot_for(record, plan))

    again = incident_store.save_plan_snapshot(_snapshot_for(record, plan))  # new id, same content

    assert again == first and again.snapshot_id == first.snapshot_id


def test_the_same_plan_with_different_inputs_is_a_new_snapshot_and_the_latest_wins(
    incident_store,
) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    strong = incident_store.save_plan_snapshot(_snapshot_for(record, plan, confidence=0.9))

    weak = _snapshot_for(record, plan, confidence=0.3)
    weak = weak.model_copy(update={"created_at": strong.created_at + timedelta(seconds=5)})
    incident_store.save_plan_snapshot(weak)

    assert incident_store.get_plan_snapshot(record.incident_id, strong.plan_fingerprint) == weak


def test_an_unknown_plan_has_no_snapshot(incident_store) -> None:
    record, _, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    assert incident_store.get_plan_snapshot(record.incident_id, "f" * 64) is None
    assert incident_store.get_plan_snapshot(new_incident_id(), "f" * 64) is None


def test_a_snapshot_needs_an_existing_incident(incident_store) -> None:
    plan = guarded(HIGH_ROLLBACK)
    orphan = build_snapshot(new_incident_id(), plan, analysis(), ["payment"])
    with pytest.raises(IncidentNotFoundError):
        incident_store.save_plan_snapshot(orphan)


def test_one_incidents_snapshot_is_not_visible_through_another(incident_store) -> None:
    mine, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    other, _, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    snap = incident_store.save_plan_snapshot(_snapshot_for(mine, plan))

    assert incident_store.get_plan_snapshot(other.incident_id, snap.plan_fingerprint) is None


def _store_tampered_snapshot(store, snap) -> None:
    """Get a snapshot into storage whose content does not match its checksum, as a database
    edit would. (PostgreSQL forbids UPDATE of a snapshot, so a bad *new* row is inserted.)"""
    later = snap.created_at + timedelta(hours=1)
    if isinstance(store, InMemoryIncidentRepository):
        bad = snap.model_copy(update={"root_cause_confidence": 0.01, "created_at": later})
        store._snapshots[(snap.incident_id, snap.plan_fingerprint)].append(bad)
        return

    import psycopg

    content = {
        "steps": [s.model_dump() for s in snap.steps],
        "root_cause": snap.root_cause,
        "root_cause_confidence": 0.01,  # altered
        "knowledge_base_consistency": snap.knowledge_base_consistency.value,
        "investigated_services": snap.investigated_services,
    }
    with psycopg.connect(Settings.from_env().get_database_url_value()) as conn:
        conn.execute(
            "INSERT INTO incident_plan_snapshots (snapshot_id, incident_id, plan_fingerprint, "
            "snapshot_version, checksum, content, created_at) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s)",
            (
                f"SNP-bad{uuid.uuid4().hex[:8]}",
                snap.incident_id,
                snap.plan_fingerprint,
                snap.version,
                "e" * 64,
                json.dumps(content),
                later,
            ),
        )


def test_a_snapshot_that_fails_its_checks_is_never_returned(incident_store) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    snap = incident_store.save_plan_snapshot(_snapshot_for(record, plan))
    _store_tampered_snapshot(incident_store, snap)

    with pytest.raises(SnapshotIntegrityError, match="checksum"):
        incident_store.get_plan_snapshot(record.incident_id, snap.plan_fingerprint)


# =============================================================================
# Execution ledger
# =============================================================================


def test_the_first_claim_wins_and_the_second_learns_what_exists(incident_store) -> None:
    incident_id = new_incident_id()
    first, claimed = incident_store.claim_step_execution(_claim_record(incident_id))
    second, claimed_again = incident_store.claim_step_execution(
        _claim_record(incident_id, source=ExecutionSource.RESUME)
    )

    assert claimed is True and claimed_again is False
    assert second == first  # the loser is handed the winner's record, not its own
    assert second.source == ExecutionSource.WORKFLOW and second.is_unknown


def test_a_claim_does_not_need_the_incident_record_to_exist(incident_store) -> None:
    """Deliberate: a workflow run executes before its record is saved, and a record of
    something that really happened must survive a failed save."""
    stored, claimed = incident_store.claim_step_execution(_claim_record(new_incident_id()))
    assert claimed and stored.status == StepExecutionStatus.STARTED


def test_different_steps_and_plans_are_claimed_independently(incident_store) -> None:
    incident_id = new_incident_id()
    claims = [
        incident_store.claim_step_execution(_claim_record(incident_id, step_index=0))[1],
        incident_store.claim_step_execution(_claim_record(incident_id, step_index=1))[1],
        incident_store.claim_step_execution(_claim_record(incident_id, plan_fp="c" * 64))[1],
    ]
    assert claims == [True, True, True]


@pytest.mark.parametrize("outcome", [SUCCEEDED, FAILED])
def test_completing_records_the_outcome(incident_store, outcome) -> None:
    incident_id = new_incident_id()
    stored, _ = incident_store.claim_step_execution(_claim_record(incident_id))

    done = incident_store.complete_step_execution(
        incident_id, stored.execution_id, outcome, "the output", "an error"
    )

    assert done.status == outcome and done.output == "the output" and done.error == "an error"
    assert done.finished_at is not None and not done.is_unknown
    assert incident_store.list_step_executions(incident_id) == [done]


@pytest.mark.parametrize("second", [SUCCEEDED, FAILED])
def test_a_finished_execution_is_never_rewritten(incident_store, second) -> None:
    incident_id = new_incident_id()
    stored, _ = incident_store.claim_step_execution(_claim_record(incident_id))
    first = incident_store.complete_step_execution(incident_id, stored.execution_id, SUCCEEDED, "x")

    with pytest.raises(ExecutionConflictError, match="never rewritten"):
        incident_store.complete_step_execution(incident_id, stored.execution_id, second, "y")

    assert incident_store.list_step_executions(incident_id) == [first]


def test_completing_something_unknown_is_not_found(incident_store) -> None:
    incident_id = new_incident_id()
    stored, _ = incident_store.claim_step_execution(_claim_record(incident_id))
    with pytest.raises(ExecutionNotFoundError):
        incident_store.complete_step_execution(incident_id, "EXE-nope", SUCCEEDED)
    with pytest.raises(ExecutionNotFoundError):  # right id, wrong incident
        incident_store.complete_step_execution(new_incident_id(), stored.execution_id, SUCCEEDED)


def test_listing_is_ordered_and_can_be_limited_to_one_plan(incident_store) -> None:
    incident_id = new_incident_id()
    for index in (2, 0, 1):
        incident_store.claim_step_execution(_claim_record(incident_id, step_index=index))
    incident_store.claim_step_execution(_claim_record(incident_id, plan_fp="c" * 64))

    everything = incident_store.list_step_executions(incident_id)
    one_plan = incident_store.list_step_executions(incident_id, "a" * 64)

    assert [r.step_index for r in one_plan] == [0, 1, 2]
    assert len(everything) == 4
    assert incident_store.list_step_executions(new_incident_id()) == []


def test_concurrent_claims_have_exactly_one_winner(incident_store) -> None:
    incident_id = new_incident_id()
    attempts = 8
    barrier = threading.Barrier(attempts)

    def attempt(i: int) -> bool:
        barrier.wait()
        source = ExecutionSource.RESUME if i % 2 else ExecutionSource.WORKFLOW
        return incident_store.claim_step_execution(_claim_record(incident_id, source=source))[1]

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        results = list(pool.map(attempt, range(attempts)))

    assert results.count(True) == 1 and results.count(False) == attempts - 1
    assert len(incident_store.list_step_executions(incident_id)) == 1


# =============================================================================
# Resume lock
# =============================================================================


def test_only_one_holder_at_a_time_and_it_is_released_afterwards(incident_store) -> None:
    incident_id = new_incident_id()
    with incident_store.resume_lock(incident_id) as first:
        assert first is True
        with incident_store.resume_lock(incident_id) as second:
            assert second is False  # refused immediately, not queued
    with incident_store.resume_lock(incident_id) as again:
        assert again is True


def test_the_lock_is_released_when_the_holder_fails(incident_store) -> None:
    incident_id = new_incident_id()
    with pytest.raises(RuntimeError):
        with incident_store.resume_lock(incident_id) as held:
            assert held
            raise RuntimeError("holder failed")
    with incident_store.resume_lock(incident_id) as after:
        assert after is True


def test_locks_are_per_incident(incident_store) -> None:
    with incident_store.resume_lock(new_incident_id()) as a:
        with incident_store.resume_lock(new_incident_id()) as b:
            assert a is True and b is True


def test_a_lock_is_held_across_threads(incident_store) -> None:
    incident_id = new_incident_id()
    acquired, release, other_result = threading.Event(), threading.Event(), []

    def holder() -> None:
        with incident_store.resume_lock(incident_id) as got:
            assert got
            acquired.set()
            release.wait(timeout=10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(timeout=10)
    with incident_store.resume_lock(incident_id) as contender:
        other_result.append(contender)
    release.set()
    thread.join(timeout=10)

    assert other_result == [False]


# =============================================================================
# PostgreSQL schema (needs a real server)
# =============================================================================


def _raw(conn, sql, params=()):
    return conn.execute(sql, params)


def _snapshot_row(conn, incident_id: str, *, checksum="d" * 64):
    _raw(
        conn,
        "INSERT INTO incident_plan_snapshots (snapshot_id, incident_id, plan_fingerprint, "
        "snapshot_version, checksum, content) VALUES (%s, %s, %s, 1, %s, '{}'::jsonb)",
        (f"SNP-{uuid.uuid4().hex[:10]}", incident_id, "a" * 64, checksum),
    )


def _real_incident(conn) -> str:
    from agentic_ai.persistence.postgres import PostgresIncidentRepository

    store = PostgresIncidentRepository.from_url(conn.info.dsn)
    record, _, _ = saved_incident(store, HIGH_ROLLBACK)
    return record.incident_id


@pytest.fixture
def pg_clean(pg_connection):  # noqa: F811
    yield pg_connection
    pg_connection.execute("DELETE FROM incident_plan_snapshots WHERE incident_id LIKE 'TEST-APR-%'")
    pg_connection.execute(
        "DELETE FROM incident_step_executions WHERE incident_id LIKE 'TEST-APR-%'"
    )


def test_a_stored_snapshot_cannot_be_updated(pg_clean) -> None:
    import psycopg

    incident_id = _real_incident(pg_clean)
    _snapshot_row(pg_clean, incident_id)

    with pytest.raises(psycopg.Error, match="immutable"):
        _raw(pg_clean, "UPDATE incident_plan_snapshots SET checksum = %s WHERE incident_id = %s",
             ("0" * 64, incident_id))  # fmt: skip


def test_a_snapshot_is_unique_per_incident_plan_and_checksum(pg_clean) -> None:
    import psycopg

    incident_id = _real_incident(pg_clean)
    _snapshot_row(pg_clean, incident_id)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _snapshot_row(pg_clean, incident_id)
    _snapshot_row(pg_clean, incident_id, checksum="9" * 64)  # a different presentation is fine


def test_an_incident_with_stored_plans_cannot_be_deleted(pg_clean) -> None:
    import psycopg

    incident_id = _real_incident(pg_clean)
    _snapshot_row(pg_clean, incident_id)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _raw(pg_clean, "DELETE FROM incidents WHERE incident_id = %s", (incident_id,))


def _execution_row(conn, incident_id: str, **overrides):
    values = {
        "execution_id": f"EXE-{uuid.uuid4().hex[:10]}",
        "incident_id": incident_id,
        "plan_fingerprint": "a" * 64,
        "step_index": 0,
        "step_fingerprint": "b" * 64,
        "action_type": "failover",
        "target_service": "payment",
        "status": "started",
        "source": "workflow",
        **overrides,
    }
    _raw(
        conn,
        f"INSERT INTO incident_step_executions ({', '.join(values)}) "
        f"VALUES ({', '.join(['%s'] * len(values))})",
        list(values.values()),
    )
    return values["execution_id"]


def test_the_ledger_allows_one_row_per_step_whatever_the_incident_record(pg_clean) -> None:
    import psycopg

    incident_id = new_incident_id()  # no incident row at all: there is deliberately no FK
    _execution_row(pg_clean, incident_id)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _execution_row(pg_clean, incident_id)
    _execution_row(pg_clean, incident_id, step_index=1)


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "maybe"},
        {"source": "somewhere"},
        {"step_index": -1},
        {"status": "succeeded"},  # finished with no finish time
    ],
)
def test_the_ledger_rejects_malformed_rows(pg_clean, bad) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        _execution_row(pg_clean, new_incident_id(), **bad)


def test_a_finished_ledger_row_cannot_be_updated_but_a_started_one_can(pg_clean) -> None:
    import psycopg

    incident_id = new_incident_id()
    started = _execution_row(pg_clean, incident_id)
    _raw(pg_clean, "UPDATE incident_step_executions SET status = 'succeeded', finished_at = now() "
         "WHERE execution_id = %s", (started,))  # fmt: skip

    with pytest.raises(psycopg.Error, match="cannot change"):
        _raw(pg_clean, "UPDATE incident_step_executions SET status = 'failed' "
             "WHERE execution_id = %s", (started,))  # fmt: skip


def test_migration_003_can_be_run_again_without_error(pg_clean) -> None:
    from pathlib import Path

    import agentic_ai.persistence as persistence_package

    sql = (
        Path(persistence_package.__file__).parent
        / "migrations"
        / "003_plan_snapshots_and_execution_ledger.sql"
    ).read_text(encoding="utf-8")
    pg_clean.execute(sql)
    pg_clean.execute(sql)


def test_the_lock_dies_with_its_session(pg_clean) -> None:
    """A crashed resumer cannot leave an incident locked: the advisory lock belongs to the
    database session, so killing the session frees it."""
    from agentic_ai.persistence.postgres import PostgresIncidentRepository

    store = PostgresIncidentRepository.from_url(pg_clean.info.dsn)
    incident_id = new_incident_id()

    with store.resume_lock(incident_id) as held:
        assert held
        holders = [
            row[0]
            for row in pg_clean.execute(
                "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted"
            ).fetchall()
        ]
        assert len(holders) == 1
        pg_clean.execute("SELECT pg_terminate_backend(%s)", (holders[0],))  # the 'crash'
        with store.resume_lock(incident_id) as taken_over:
            assert taken_over is True  # the dead session's lock is gone


def test_the_latest_snapshot_is_the_last_one_stored_even_when_timestamps_tie(
    incident_store,
) -> None:
    """'Latest' must not depend on clock resolution. Resume re-runs the guardrails on the latest
    snapshot's inputs, so picking the older one by accident would use the wrong inputs.

    Repeated across many plans: ordering by (timestamp, random id) would fail this almost
    certainly, ordering by storage order never does.
    """
    for _ in range(12):
        record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
        first = _snapshot_for(record, plan, confidence=0.9)
        second = _snapshot_for(record, plan, confidence=0.3).model_copy(
            update={"created_at": first.created_at}  # exactly the same instant
        )

        incident_store.save_plan_snapshot(first)
        incident_store.save_plan_snapshot(second)

        latest = incident_store.get_plan_snapshot(record.incident_id, first.plan_fingerprint)
        assert latest.snapshot_id == second.snapshot_id and latest.root_cause_confidence == 0.3
