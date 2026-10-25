"""Approval repository tests.

Three layers, kept distinct about what they prove:

* Contract tests (``incident_store``) run against the in-memory repository and, when
  ``DATABASE_URL`` points at a real server, PostgreSQL. This is the evidence that the
  two behave identically and that the SQL is actually valid.
* PostgreSQL-only tests check the schema itself (constraints, foreign key, indexes,
  migration re-runs). They skip without a server.
* Fake-connection tests check the PostgreSQL class's transaction handling and SQL
  shape. A fake accepts any SQL string, so these prove the Python logic only.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentic_ai.config import Settings
from agentic_ai.domain.approvals import ApprovalRecord, pending_requests_for
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRecord
from agentic_ai.persistence.postgres import PostgresIncidentRepository
from agentic_ai.persistence.repository import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    ApprovalRepository,
    IncidentNotFoundError,
    IncidentPersistenceError,
    IncidentStore,
)
from tests.approval_helpers import (
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    LOW,
    explain,
    seed_realistic_approvals,
    guarded,
    new_incident_id,
    saved_incident,
    state_for,
)

A, R = ApprovalState.APPROVED, ApprovalState.REJECTED


def _only(approvals: list[ApprovalRecord]) -> ApprovalRecord:
    assert len(approvals) == 1
    return approvals[0]


# =============================================================================
# Contract: every backend
# =============================================================================


def test_backends_satisfy_the_protocols(incident_store) -> None:
    assert isinstance(incident_store, ApprovalRepository)
    assert isinstance(incident_store, IncidentStore)


def test_pending_approvals_are_registered_for_the_exact_step(incident_store) -> None:
    _, _, approvals = saved_incident(incident_store, LOW, HIGH_ROLLBACK)

    approval = _only(approvals)  # the LOW step needs none
    assert approval.status == ApprovalState.PENDING
    assert approval.step_index == 1
    assert (approval.action_type, approval.target_service) == HIGH_ROLLBACK
    assert incident_store.list_approvals(approval.incident_id) == [approval]


def test_requesting_for_a_missing_incident_is_refused_and_stores_nothing(incident_store) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK)
    real = pending_requests_for(record.incident_id, guarded(HIGH_FAILOVER))
    orphan = pending_requests_for(new_incident_id(), plan)

    with pytest.raises(IncidentNotFoundError):
        incident_store.request_approvals([*real, *orphan])

    # The whole batch rolled back: not even the valid half was stored.
    assert [a.action_type for a in incident_store.list_approvals(record.incident_id)] == [
        HIGH_ROLLBACK[0]
    ]


def test_registering_twice_keeps_one_approval(incident_store) -> None:
    record, plan, first = saved_incident(incident_store, HIGH_ROLLBACK)

    again = incident_store.request_approvals(pending_requests_for(record.incident_id, plan))

    assert again == first  # same id: the new request was discarded, the old one returned
    assert len(incident_store.list_approvals(record.incident_id)) == 1


def test_registering_again_never_resets_a_decision(incident_store) -> None:
    record, plan, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    decided = incident_store.decide_approval(record.incident_id, approval.approval_id, A, "alice")

    again = incident_store.request_approvals(pending_requests_for(record.incident_id, plan))

    assert again == [decided]
    assert again[0].status == A and again[0].approver == "alice"


def test_get_is_scoped_to_the_owning_incident(incident_store) -> None:
    mine, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    other, _, _ = saved_incident(incident_store, HIGH_ROLLBACK)

    assert incident_store.get_approval(mine.incident_id, approval.approval_id) == approval
    assert incident_store.get_approval(other.incident_id, approval.approval_id) is None
    assert incident_store.get_approval(mine.incident_id, "APR-nope") is None


def test_listing_filters_by_status_and_orders_oldest_first(incident_store) -> None:
    record, plan, _ = saved_incident(incident_store, HIGH_ROLLBACK, HIGH_FAILOVER)
    first, second = incident_store.list_approvals(record.incident_id)
    assert (first.step_index, second.step_index) == (0, 1)

    incident_store.decide_approval(record.incident_id, first.approval_id, A, "alice")

    pending = incident_store.list_approvals(record.incident_id, ApprovalState.PENDING)
    approved = incident_store.list_approvals(record.incident_id, ApprovalState.APPROVED)
    assert [a.step_index for a in pending] == [1]
    assert [a.step_index for a in approved] == [0]
    assert incident_store.list_approvals(record.incident_id, ApprovalState.REJECTED) == []
    assert incident_store.list_approvals(new_incident_id()) == []


def test_pending_queue_spans_incidents_oldest_first_and_honours_the_limit(incident_store) -> None:
    base = datetime.now(timezone.utc) - timedelta(hours=1)
    minutes_ago = {"newest": 10, "middle": 20, "oldest": 30}  # smaller = more recent
    registered: dict[str, ApprovalRecord] = {}
    for label, minutes in minutes_ago.items():
        incident_id = new_incident_id()
        plan = guarded(HIGH_ROLLBACK)
        incident_store.save_incident(
            IncidentRecord.from_workflow_state(state_for(incident_id, plan))
        )
        (request,) = pending_requests_for(incident_id, plan)
        # Requested in this order of *time*, not of insertion, so ordering is observable.
        (stored,) = incident_store.request_approvals(
            [request.model_copy(update={"requested_at": base - timedelta(minutes=minutes)})]
        )
        registered[label] = stored

    ours = {a.approval_id for a in registered.values()}
    queue = [a for a in incident_store.list_pending_approvals(limit=500) if a.approval_id in ours]

    assert [a.approval_id for a in queue] == [
        registered["oldest"].approval_id,
        registered["middle"].approval_id,
        registered["newest"].approval_id,
    ]
    assert len(incident_store.list_pending_approvals(limit=1)) == 1
    with pytest.raises(ValueError):
        incident_store.list_pending_approvals(limit=0)


def test_decided_approvals_leave_the_pending_queue(incident_store) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    incident_store.decide_approval(record.incident_id, approval.approval_id, R, "alice")

    queue = incident_store.list_pending_approvals(limit=500)
    assert approval.approval_id not in {a.approval_id for a in queue}


@pytest.mark.parametrize("outcome", [A, R])
def test_a_decision_records_who_when_and_why(incident_store, outcome) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    before = datetime.now(timezone.utc)

    decided = incident_store.decide_approval(
        record.incident_id, approval.approval_id, outcome, " alice ", " ticket 42 "
    )

    assert decided.status == outcome
    assert decided.approver == "alice" and decided.decision_reason == "ticket 42"
    assert decided.decided_at is not None and decided.decided_at >= before - timedelta(seconds=1)
    assert incident_store.get_approval(record.incident_id, approval.approval_id) == decided
    # What was proposed is preserved, whatever happened to it afterwards.
    assert (decided.action_type, decided.step_index) == (HIGH_ROLLBACK[0], 0)


@pytest.mark.parametrize("first", [A, R])
@pytest.mark.parametrize("second", [A, R])
def test_a_second_decision_is_refused_and_the_first_stands(incident_store, first, second) -> None:
    """All four pairs, including repeating the same decision."""
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    winner = incident_store.decide_approval(
        record.incident_id, approval.approval_id, first, "alice", "first"
    )

    with pytest.raises(ApprovalConflictError, match="already"):
        incident_store.decide_approval(
            record.incident_id, approval.approval_id, second, "mallory", "second"
        )

    assert incident_store.get_approval(record.incident_id, approval.approval_id) == winner


def test_unknown_things_are_not_found(incident_store) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)

    with pytest.raises(IncidentNotFoundError):
        incident_store.decide_approval(new_incident_id(), approval.approval_id, A, "alice")
    with pytest.raises(ApprovalNotFoundError):
        incident_store.decide_approval(record.incident_id, "APR-nope", A, "alice")


def test_an_approval_cannot_be_decided_through_another_incident(incident_store) -> None:
    mine, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    other, _, _ = saved_incident(incident_store, HIGH_ROLLBACK)

    with pytest.raises(ApprovalNotFoundError):
        incident_store.decide_approval(other.incident_id, approval.approval_id, A, "alice")

    assert incident_store.get_approval(mine.incident_id, approval.approval_id).is_pending


@pytest.mark.parametrize("approver", ["", "   "])
def test_a_blank_approver_is_refused_before_anything_changes(incident_store, approver) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    with pytest.raises(ValueError):
        incident_store.decide_approval(record.incident_id, approval.approval_id, A, approver)
    assert incident_store.get_approval(record.incident_id, approval.approval_id).is_pending


def test_pending_is_not_a_decision(incident_store) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    with pytest.raises(ValueError):
        incident_store.decide_approval(
            record.incident_id, approval.approval_id, ApprovalState.PENDING, "alice"
        )


def test_a_closed_incident_accepts_no_decisions(incident_store) -> None:
    record, plan, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    incident_store.save_incident(
        IncidentRecord.from_workflow_state(state_for(record.incident_id, plan, "resolve"))
    )

    with pytest.raises(ApprovalConflictError, match="closed"):
        incident_store.decide_approval(record.incident_id, approval.approval_id, A, "alice")
    assert incident_store.get_approval(record.incident_id, approval.approval_id).is_pending


def test_an_approval_for_a_replaced_plan_cannot_be_decided(incident_store) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    # The incident is re-run and now carries a different plan; it is still open.
    incident_store.save_incident(
        IncidentRecord.from_workflow_state(state_for(record.incident_id, guarded(HIGH_FAILOVER)))
    )

    with pytest.raises(ApprovalConflictError, match="superseded"):
        incident_store.decide_approval(record.incident_id, approval.approval_id, A, "alice")
    assert incident_store.get_approval(record.incident_id, approval.approval_id).is_pending


def test_resaving_an_incident_keeps_its_approvals(incident_store) -> None:
    """Remediation rows are replaced wholesale on re-save; approvals must survive that."""
    record, plan, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    decided = incident_store.decide_approval(record.incident_id, approval.approval_id, A, "alice")

    incident_store.save_incident(
        IncidentRecord.from_workflow_state(state_for(record.incident_id, plan, "escalate"))
    )

    assert incident_store.list_approvals(record.incident_id) == [decided]


def test_concurrent_decisions_have_exactly_one_winner(incident_store) -> None:
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)
    attempts = 8
    barrier = threading.Barrier(attempts)

    def attempt(i: int):
        outcome = A if i % 2 == 0 else R
        barrier.wait()
        try:
            return incident_store.decide_approval(
                record.incident_id, approval.approval_id, outcome, f"approver-{i}"
            )
        except ApprovalConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=attempts) as pool:
        results = list(pool.map(attempt, range(attempts)))

    winners = [r for r in results if isinstance(r, ApprovalRecord)]
    losers = [r for r in results if isinstance(r, ApprovalConflictError)]
    assert len(winners) == 1 and len(losers) == attempts - 1

    final = incident_store.get_approval(record.incident_id, approval.approval_id)
    assert final == winners[0]  # the stored decision is the winner's, untouched by the losers


# =============================================================================
# PostgreSQL schema (needs a real server)
# =============================================================================


@pytest.fixture
def pg_connection():
    database_url = Settings.from_env().get_database_url_value()
    if not database_url:
        pytest.skip("DATABASE_URL not set; the PostgreSQL schema is unverified.")
    try:
        import psycopg

        from agentic_ai.persistence.migrator import MigrationRunner

        with psycopg.connect(database_url) as conn:
            MigrationRunner().apply_migrations(conn)
    except Exception as exc:
        pytest.skip(f"Could not reach or migrate the configured PostgreSQL: {exc}")

    conn = psycopg.connect(database_url, autocommit=True)
    yield conn
    conn.execute("DELETE FROM incident_approvals WHERE incident_id LIKE 'TEST-APR-%'")
    conn.execute("DELETE FROM incidents WHERE incident_id LIKE 'TEST-APR-%'")
    conn.close()


def _insert_approval(conn, incident_id: str, **overrides) -> None:
    values = {
        "approval_id": f"APR-{new_incident_id()[-10:]}",
        "incident_id": incident_id,
        "plan_fingerprint": "a" * 64,
        "step_index": 0,
        "step_fingerprint": "b" * 64,
        "action_type": "failover",
        "target_service": "payment",
        "status": "pending",
        "decided_at": None,
        "approver": None,
        **overrides,
    }
    columns = ", ".join(values)
    placeholders = ", ".join(["%s"] * len(values))
    conn.execute(
        f"INSERT INTO incident_approvals ({columns}) VALUES ({placeholders})",
        list(values.values()),
    )


def _real_incident(conn) -> str:
    incident_id = new_incident_id()
    PostgresIncidentRepository(lambda: __import__("psycopg").connect(conn.info.dsn)).save_incident(
        IncidentRecord.from_workflow_state(state_for(incident_id, guarded(HIGH_ROLLBACK)))
    )
    return incident_id


def test_schema_rejects_an_approval_for_an_unknown_incident(pg_connection) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_approval(pg_connection, new_incident_id())


def test_schema_rejects_an_invalid_status(pg_connection) -> None:
    import psycopg

    incident_id = _real_incident(pg_connection)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_approval(pg_connection, incident_id, status="maybe")


def test_schema_demands_a_decider_and_time_for_a_decision(pg_connection) -> None:
    import psycopg

    incident_id = _real_incident(pg_connection)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_approval(pg_connection, incident_id, status="approved")  # no approver, no time
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_approval(
            pg_connection,
            incident_id,
            plan_fingerprint="c" * 64,
            approver="alice",  # a pending approval cannot already have an approver
        )


def test_schema_allows_only_one_approval_per_step_of_a_plan(pg_connection) -> None:
    import psycopg

    incident_id = _real_incident(pg_connection)
    _insert_approval(pg_connection, incident_id)
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_approval(pg_connection, incident_id)
    _insert_approval(pg_connection, incident_id, step_index=1)  # another step is fine


def test_schema_rejects_a_negative_step_index(pg_connection) -> None:
    import psycopg

    incident_id = _real_incident(pg_connection)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_approval(pg_connection, incident_id, step_index=-1)


def test_an_incident_with_approval_history_cannot_be_deleted(pg_connection) -> None:
    import psycopg

    incident_id = _real_incident(pg_connection)
    _insert_approval(pg_connection, incident_id)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        pg_connection.execute("DELETE FROM incidents WHERE incident_id = %s", (incident_id,))


def test_the_status_constraint_now_admits_awaiting_approval(pg_connection) -> None:
    incident_id = new_incident_id()
    record = IncidentRecord.from_workflow_state(state_for(incident_id, guarded(HIGH_ROLLBACK)))
    assert record.status == IncidentLifecycleStatus.AWAITING_APPROVAL
    repo = PostgresIncidentRepository(lambda: __import__("psycopg").connect(pg_connection.info.dsn))
    repo.save_incident(record)
    assert repo.get_incident(incident_id).status == IncidentLifecycleStatus.AWAITING_APPROVAL


def test_pending_approvals_are_served_by_the_partial_index(pg_connection) -> None:
    indexes = dict(
        pg_connection.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'incident_approvals'"
        ).fetchall()
    )
    assert "idx_incident_approvals_pending" in indexes
    assert "WHERE" in indexes["idx_incident_approvals_pending"]
    assert "idx_incident_approvals_incident_status" in indexes

    # On a realistic table (many decided approvals, few pending) the queue query is served by
    # the partial index, which also supplies the ordering, so no sort is needed. (A near-empty
    # table proves nothing: PostgreSQL rightly prefers a bitmap scan and a sort over a handful
    # of rows, and which it picks depends on table statistics.)
    seed_realistic_approvals(pg_connection)
    plan = "\n".join(
        explain(
            pg_connection,
            "SELECT approval_id FROM incident_approvals WHERE status = 'pending' "
            "ORDER BY requested_at, approval_id LIMIT 50",
        )
    )
    assert "idx_incident_approvals_pending" in plan
    assert "Sort" not in plan


def test_migration_002_can_be_run_again_without_error(pg_connection) -> None:
    sql = (
        Path(__import__("agentic_ai.persistence", fromlist=["x"]).__file__).parent
        / "migrations"
        / "002_incident_approvals.sql"
    ).read_text(encoding="utf-8")
    pg_connection.execute(sql)
    pg_connection.execute(sql)


# =============================================================================
# PostgresIncidentRepository transaction handling (fake connection)
# =============================================================================


class _Cursor:
    def __init__(self, conn: "_Conn") -> None:
        self.conn = conn
        self.rowcount = 1
        self._rows: list = []

    def execute(self, sql, params=None) -> None:
        sql = " ".join(sql.split())
        self.conn.executed.append(sql)
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("driver exploded")
        self._rows = list(self.conn.respond(sql, params))
        if sql.startswith("UPDATE incident_approvals"):
            self.rowcount = self.conn.update_rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self) -> None:
        pass


class _Conn:
    def __init__(self, respond=None, fail_on=None, update_rowcount=1) -> None:
        self.respond = respond or (lambda sql, params: [])
        self.fail_on = fail_on
        self.update_rowcount = update_rowcount
        self.executed: list[str] = []
        self.commits = self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _decide_responder(approval: ApprovalRecord, incident_status="awaiting_approval"):
    def respond(sql: str, params):
        if sql.startswith("SELECT status FROM incidents"):
            return [(incident_status,)]
        if "FROM incident_approvals" in sql and sql.endswith("FOR UPDATE"):
            return [
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
                )
            ]
        if "FROM incident_remediations" in sql:
            return [
                (0, HIGH_ROLLBACK[0], HIGH_ROLLBACK[1], "requires_approval", "skipped", {}, None)
            ]
        return []

    return respond


def _pending_approval() -> ApprovalRecord:
    return pending_requests_for("INC-F", guarded(HIGH_ROLLBACK))[0]


def test_a_decision_locks_checks_then_writes_conditionally_in_one_transaction() -> None:
    approval = _pending_approval()
    conn = _Conn(_decide_responder(approval))

    decided = PostgresIncidentRepository(lambda: conn).decide_approval(
        "INC-F", approval.approval_id, A, "alice", "ok"
    )

    assert decided.status == A
    kinds = [
        "SELECT status FROM incidents WHERE incident_id = %s FOR SHARE",
        "FOR UPDATE",
        "FROM incident_remediations",
        "UPDATE incident_approvals",
    ]
    positions = [next(i for i, sql in enumerate(conn.executed) if k in sql) for k in kinds]
    assert positions == sorted(positions)
    assert "AND status = 'pending'" in conn.executed[positions[-1]]
    assert conn.commits == 1 and conn.rollbacks == 0 and conn.closed


def test_a_refused_decision_rolls_back_without_writing() -> None:
    approval = _pending_approval()
    conn = _Conn(_decide_responder(approval, incident_status="resolved"))

    with pytest.raises(ApprovalConflictError, match="closed"):
        PostgresIncidentRepository(lambda: conn).decide_approval(
            "INC-F", approval.approval_id, A, "a"
        )

    assert not any(sql.startswith("UPDATE") for sql in conn.executed)
    assert conn.rollbacks == 1 and conn.commits == 0 and conn.closed


def test_losing_the_conditional_update_is_a_conflict_not_a_success() -> None:
    approval = _pending_approval()
    conn = _Conn(_decide_responder(approval), update_rowcount=0)

    with pytest.raises(ApprovalConflictError, match="concurrent"):
        PostgresIncidentRepository(lambda: conn).decide_approval(
            "INC-F", approval.approval_id, A, "a"
        )

    assert conn.rollbacks == 1 and conn.commits == 0


def test_a_missing_approval_row_is_not_found() -> None:
    conn = _Conn(lambda sql, params: [("awaiting_approval",)] if "FROM incidents" in sql else [])
    with pytest.raises(ApprovalNotFoundError):
        PostgresIncidentRepository(lambda: conn).decide_approval("INC-F", "APR-x", A, "a")
    assert conn.rollbacks == 1


def test_a_driver_failure_while_deciding_is_wrapped_and_rolled_back() -> None:
    approval = _pending_approval()
    conn = _Conn(_decide_responder(approval), fail_on="UPDATE incident_approvals")

    with pytest.raises(IncidentPersistenceError, match="driver exploded"):
        PostgresIncidentRepository(lambda: conn).decide_approval(
            "INC-F", approval.approval_id, A, "a"
        )

    assert conn.rollbacks == 1 and conn.commits == 0 and conn.closed


def test_a_driver_failure_while_registering_stores_nothing() -> None:
    conn = _Conn(lambda sql, params: [(1,)], fail_on="INSERT INTO incident_approvals")
    with pytest.raises(IncidentPersistenceError):
        PostgresIncidentRepository(lambda: conn).request_approvals([_pending_approval()])
    assert conn.rollbacks == 1 and conn.commits == 0


def test_registering_checks_the_incident_exists_inside_the_transaction() -> None:
    conn = _Conn()  # SELECT 1 ... returns nothing
    with pytest.raises(IncidentNotFoundError):
        PostgresIncidentRepository(lambda: conn).request_approvals([_pending_approval()])
    assert not any(sql.startswith("INSERT") for sql in conn.executed)
    assert conn.rollbacks == 1


def test_a_bad_decision_never_opens_a_connection() -> None:
    opened: list[_Conn] = []

    def connect() -> _Conn:
        opened.append(_Conn())
        return opened[-1]

    repo = PostgresIncidentRepository(connect)
    with pytest.raises(ValueError):
        repo.decide_approval("INC-F", "APR-x", A, "  ")
    with pytest.raises(ValueError):
        repo.decide_approval("INC-F", "APR-x", ApprovalState.PENDING, "alice")
    assert opened == []
