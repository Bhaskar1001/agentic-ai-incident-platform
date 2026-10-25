"""Tests for PostgresIncidentRepository.

Two layers, deliberately kept apart:

* Unit tests drive the repository through a fake DB-API connection. They prove
  the Python logic (conflict rules, transaction handling, row mapping) but
  cannot prove the SQL is valid; a fake accepts any string.
* A contract test runs one scenario against both the in-memory repository and,
  when ``DATABASE_URL`` points at a real server, PostgreSQL. That is the only
  place the SQL is actually exercised, and it skips (it does not pass) without
  a server.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agentic_ai.config import Settings
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.persistence import (
    IncidentLifecycleStatus,
    IncidentRecord,
    IncidentRemediationRecord,
    IncidentStageRecord,
)
from agentic_ai.domain.summaries import DiagnosticSummary, VerificationSummary
from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from agentic_ai.persistence.migrator import MigrationRunner
from agentic_ai.persistence.postgres import PostgresIncidentRepository
from agentic_ai.persistence.repository import (
    IncidentConflictError,
    IncidentPersistenceError,
)

NOW = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)


def _record(
    incident_id: str = "INC-1",
    *,
    status: IncidentLifecycleStatus = IncidentLifecycleStatus.RESOLVED,
    outcome: str | None = "resolve",
    severity: Severity = Severity.HIGH,
    correlation_id: str = "CORR-1",
    created_at: datetime = NOW,
    with_children: bool = True,
) -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident_id,
        correlation_id=correlation_id,
        scenario="connection_pool_exhaustion",
        incident_description="Payment service is down",
        severity=severity,
        status=status,
        final_outcome=outcome,
        diagnostic=DiagnosticSummary(
            decision="resolve", trigger_type="verification", primary_reason="recovered"
        ),
        verification=VerificationSummary(status="recovered", attempts=1, reasons=["all clear"]),
        stages=[
            IncidentStageRecord(
                stage_name="investigate",
                status="completed",
                start_time=NOW,
                end_time=NOW + timedelta(seconds=2),
                duration_ms=2000.0,
                metadata={"tools": 3},
            )
        ]
        if with_children
        else [],
        remediations=[
            IncidentRemediationRecord(
                step_order=0,
                action_type="scale_up",
                target_service="payment",
                disposition="allowed",
                execution_status="executed",
                output="[SIMULATED] Scaled up",
            ),
            IncidentRemediationRecord(
                step_order=1,
                action_type="rollback_deployment",
                target_service="payment",
                disposition="requires_approval",
                execution_status="skipped",
            ),
        ]
        if with_children
        else [],
        total_duration_ms=2500.0,
        created_at=created_at,
        updated_at=created_at,
        resolved_at=created_at if status.is_terminal else None,
    )


# --- fake DB-API connection ----------------------------------------------------


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._result: list = []

    def execute(self, sql: str, params=None) -> None:
        self._connection.executed.append((" ".join(sql.split()), params))
        if self._connection.fail_on and self._connection.fail_on in sql:
            raise RuntimeError("driver exploded")
        self._result = list(self._connection.responder(" ".join(sql.split()), params))

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def close(self) -> None:
        pass


class FakeConnection:
    def __init__(self, responder=None, fail_on: str | None = None) -> None:
        self.responder = responder or (lambda sql, params: [])
        self.fail_on = fail_on
        self.executed: list[tuple[str, object]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    def sql_starting(self, prefix: str) -> list[tuple[str, object]]:
        return [e for e in self.executed if e[0].startswith(prefix)]


def _repo(connection: FakeConnection) -> PostgresIncidentRepository:
    return PostgresIncidentRepository(lambda: connection)


# --- save ----------------------------------------------------------------------


def test_save_new_incident_writes_one_transaction() -> None:
    conn = FakeConnection()
    record = _record()

    assert _repo(conn).save_incident(record) is record

    assert len(conn.sql_starting("INSERT INTO incidents")) == 1
    assert len(conn.sql_starting("INSERT INTO incident_stages")) == 1
    assert len(conn.sql_starting("INSERT INTO incident_remediations")) == 2
    assert conn.commits == 1 and conn.rollbacks == 0 and conn.closed


def test_save_locks_the_existing_row_before_deciding() -> None:
    conn = FakeConnection()
    _repo(conn).save_incident(_record())
    first_sql = conn.executed[0][0]
    assert first_sql.startswith("SELECT status") and first_sql.endswith("FOR UPDATE")


def test_save_replaces_children_before_inserting_them() -> None:
    conn = FakeConnection()
    _repo(conn).save_incident(_record())
    sqls = [sql for sql, _ in conn.executed]
    delete_stages = next(
        i for i, s in enumerate(sqls) if s.startswith("DELETE FROM incident_stages")
    )
    insert_stage = next(
        i for i, s in enumerate(sqls) if s.startswith("INSERT INTO incident_stages")
    )
    assert delete_stages < insert_stage


def test_json_columns_are_sent_as_json_text() -> None:
    conn = FakeConnection()
    _repo(conn).save_incident(_record())
    upsert_params = conn.sql_starting("INSERT INTO incidents")[0][1]
    assert json.loads(upsert_params[12]) == ["all clear"]
    stage_params = conn.sql_starting("INSERT INTO incident_stages")[0][1]
    assert json.loads(stage_params[8]) == {"tools": 3}


def test_incident_without_diagnostic_or_verification_stores_nulls() -> None:
    conn = FakeConnection()
    record = _record().model_copy(update={"diagnostic": None, "verification": None})
    _repo(conn).save_incident(record)
    params = conn.sql_starting("INSERT INTO incidents")[0][1]
    assert params[7:10] == (None, None, None)
    assert params[10] is None and params[11] == 0 and json.loads(params[12]) == []


def _existing_row(status: str, outcome: str | None, severity: str):
    def respond(sql: str, params):
        if sql.startswith("SELECT status"):
            return [(status, outcome, severity)]
        return []

    return respond


def test_terminal_record_cannot_revert_to_non_terminal() -> None:
    conn = FakeConnection(_existing_row("resolved", "resolve", "high"))
    reverting = _record(status=IncidentLifecycleStatus.INVESTIGATING, outcome=None)

    with pytest.raises(IncidentConflictError, match="revert"):
        _repo(conn).save_incident(reverting)

    assert conn.rollbacks == 1 and conn.commits == 0 and conn.closed
    assert not conn.sql_starting("INSERT INTO incidents")


def test_terminal_record_cannot_change_outcome_or_severity() -> None:
    conn = FakeConnection(_existing_row("resolved", "resolve", "high"))
    with pytest.raises(IncidentConflictError, match="conflicting"):
        _repo(conn).save_incident(
            _record(outcome="escalate", status=IncidentLifecycleStatus.ESCALATED)
        )

    conn = FakeConnection(_existing_row("resolved", "resolve", "high"))
    with pytest.raises(IncidentConflictError, match="conflicting"):
        _repo(conn).save_incident(_record(severity=Severity.LOW))


def test_non_terminal_record_may_advance_to_terminal() -> None:
    conn = FakeConnection(_existing_row("investigating", None, "high"))
    _repo(conn).save_incident(_record())
    assert conn.commits == 1 and conn.sql_starting("INSERT INTO incidents")


def test_identical_terminal_resave_is_allowed() -> None:
    conn = FakeConnection(_existing_row("resolved", "resolve", "high"))
    _repo(conn).save_incident(_record())
    assert conn.commits == 1


def test_driver_failure_rolls_back_and_is_wrapped() -> None:
    conn = FakeConnection(fail_on="INSERT INTO incident_remediations")

    with pytest.raises(IncidentPersistenceError, match="driver exploded"):
        _repo(conn).save_incident(_record())

    assert conn.rollbacks == 1 and conn.commits == 0 and conn.closed


# --- reads ---------------------------------------------------------------------

_INCIDENT_ROW = (
    "INC-1", "CORR-1", "connection_pool_exhaustion", "Payment service is down",
    "high", "resolved", "resolve",
    "resolve", "verification", "recovered",
    "recovered", 1, '["all clear"]',
    2500.0, NOW, NOW, NOW,
)  # fmt: skip
_STAGE_ROW = (
    "INC-1", "investigate", "completed", NOW, NOW + timedelta(seconds=2),
    2000.0, None, None, '{"tools": 3}',
)  # fmt: skip
_REMEDIATION_ROWS = [
    ("INC-1", 0, "scale_up", "payment", "allowed", "executed", "{}", "[SIMULATED] Scaled up"),
    ("INC-1", 1, "rollback_deployment", "payment", "requires_approval", "skipped", {}, None),
]


def _read_responder(sql: str, params):
    if sql.startswith("SELECT incident_id, correlation_id"):
        return [_INCIDENT_ROW]
    if "FROM incident_stages" in sql:
        return [_STAGE_ROW]
    if "FROM incident_remediations" in sql:
        return _REMEDIATION_ROWS
    return []


def test_get_incident_rebuilds_the_full_record() -> None:
    record = _repo(FakeConnection(_read_responder)).get_incident("INC-1")

    assert record is not None
    assert record.severity == Severity.HIGH
    assert record.status == IncidentLifecycleStatus.RESOLVED
    assert record.diagnostic == DiagnosticSummary(
        decision="resolve", trigger_type="verification", primary_reason="recovered"
    )
    assert record.verification == VerificationSummary(
        status="recovered", attempts=1, reasons=["all clear"]
    )
    assert record.stages[0].metadata == {"tools": 3}
    assert [r.step_order for r in record.remediations] == [0, 1]
    assert record.remediations[1].parameters == {}


def test_get_incident_returns_none_when_absent() -> None:
    assert _repo(FakeConnection()).get_incident("INC-missing") is None


def test_list_queries_children_once_not_per_incident() -> None:
    conn = FakeConnection(_read_responder)
    _repo(conn).list_incidents()
    assert len(conn.sql_starting("SELECT incident_id, stage_name")) == 1
    assert len(conn.sql_starting("SELECT incident_id, step_order")) == 1


def test_list_orders_newest_first_with_a_stable_tiebreak() -> None:
    conn = FakeConnection(_read_responder)
    _repo(conn).list_incidents(limit=5, offset=10)
    sql, params = conn.executed[0]
    assert "ORDER BY created_at DESC, incident_id DESC" in sql
    assert params == (5, 10)


def test_list_filters_by_status_value() -> None:
    conn = FakeConnection(_read_responder)
    _repo(conn).list_incidents(status=IncidentLifecycleStatus.ESCALATED)
    sql, params = conn.executed[0]
    assert "WHERE status = %s" in sql and params == ("escalated", 50, 0)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"offset": -1}])
def test_list_rejects_bad_paging_before_touching_the_database(kwargs) -> None:
    opened: list[FakeConnection] = []

    def connect() -> FakeConnection:
        opened.append(FakeConnection())
        return opened[-1]

    with pytest.raises(ValueError):
        PostgresIncidentRepository(connect).list_incidents(**kwargs)
    assert opened == []


def test_count_returns_an_int() -> None:
    conn = FakeConnection(lambda sql, params: [(7,)])
    assert _repo(conn).count() == 7
    assert _repo(FakeConnection(lambda s, p: [(2,)])).count(IncidentLifecycleStatus.RESOLVED) == 2


def test_from_url_connects_through_psycopg(monkeypatch) -> None:
    import psycopg

    seen: list[str] = []
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda url: seen.append(url) or FakeConnection(lambda sql, params: [(0,)]),
    )

    repo = PostgresIncidentRepository.from_url("postgresql://u:p@localhost/db")
    repo.count()

    assert seen == ["postgresql://u:p@localhost/db"]


# --- contract: same scenario against every implementation ------------------------


def _contract_scenario(repo, ids: dict[str, str]) -> None:
    """The behaviour every IncidentRepository must share."""
    older = _record(
        ids["a"], correlation_id=ids["corr"], created_at=NOW - timedelta(hours=2),
        status=IncidentLifecycleStatus.ESCALATED, outcome="escalate",
    )  # fmt: skip
    newer = _record(ids["b"], correlation_id=ids["corr"], created_at=NOW)
    repo.save_incident(older)
    repo.save_incident(newer)

    # Round trip, including children and JSON columns.
    loaded = repo.get_incident(ids["b"])
    assert loaded == newer
    assert repo.get_incident(ids["missing"]) is None

    # Correlation lookup is newest first.
    assert [r.incident_id for r in repo.get_by_correlation_id(ids["corr"])] == [ids["b"], ids["a"]]

    # Status filter and count agree.
    escalated = [
        r for r in repo.list_incidents(limit=200, status=IncidentLifecycleStatus.ESCALATED)
        if r.incident_id in ids.values()
    ]  # fmt: skip
    assert [r.incident_id for r in escalated] == [ids["a"]]
    assert repo.count(IncidentLifecycleStatus.ESCALATED) >= 1

    # A terminal record is frozen.
    with pytest.raises(IncidentConflictError):
        repo.save_incident(
            _record(ids["b"], status=IncidentLifecycleStatus.INVESTIGATING, outcome=None,
                    correlation_id=ids["corr"])
        )  # fmt: skip
    with pytest.raises(IncidentConflictError):
        repo.save_incident(_record(ids["b"], severity=Severity.LOW, correlation_id=ids["corr"]))
    assert repo.get_incident(ids["b"]) == newer  # the failed saves changed nothing

    # Children are replaced, not accumulated, on a legitimate re-save.
    slimmer = _record(ids["b"], correlation_id=ids["corr"], with_children=False)
    repo.save_incident(slimmer)
    reloaded = repo.get_incident(ids["b"])
    assert reloaded.stages == [] and reloaded.remediations == []


def _ids() -> dict[str, str]:
    token = uuid.uuid4().hex[:10]
    return {
        "a": f"TEST-{token}-A",
        "b": f"TEST-{token}-B",
        "corr": f"TEST-CORR-{token}",
        "missing": f"TEST-{token}-NONE",
    }


def test_contract_holds_for_the_in_memory_repository() -> None:
    _contract_scenario(InMemoryIncidentRepository(), _ids())


def test_contract_holds_for_postgresql() -> None:
    """Runs only against a real server; otherwise skipped, never silently passed."""
    db_url = Settings.from_env().get_database_url_value()
    if not db_url:
        pytest.skip("DATABASE_URL not set; PostgreSQL SQL remains unverified.")

    try:
        import psycopg
    except ImportError:
        pytest.skip("psycopg not installed.")

    try:
        with psycopg.connect(db_url) as conn:
            MigrationRunner().apply_migrations(conn)
    except Exception as exc:
        pytest.skip(f"Could not reach or migrate the configured PostgreSQL: {exc}")

    ids = _ids()
    repo = PostgresIncidentRepository.from_url(db_url)
    try:
        _contract_scenario(repo, ids)
    finally:
        with psycopg.connect(db_url) as conn:
            conn.execute(
                "DELETE FROM incidents WHERE incident_id = ANY(%s)", ([ids["a"], ids["b"]],)
            )


def test_api_round_trip_through_postgresql(monkeypatch) -> None:
    """POST /incidents -> PostgreSQL -> GET /incidents/{id}, over the real SQL path.

    The workflow's LLM steps are stubbed (as in test_api.py); everything from the
    API handler down to the database is real. Skipped without a server.
    """
    from fastapi.testclient import TestClient

    from agentic_ai.api import create_app
    from agentic_ai.domain.remediation import (
        RemediationActionType,
        RemediationPlan,
        RemediationStep,
    )
    from agentic_ai.workflows import incident as workflow_module
    from tests.test_api import _analysis, _assessment, _finding, _result

    db_url = Settings.from_env().get_database_url_value()
    if not db_url:
        pytest.skip("DATABASE_URL not set; PostgreSQL SQL remains unverified.")
    try:
        import psycopg

        with psycopg.connect(db_url) as conn:
            MigrationRunner().apply_migrations(conn)
    except Exception as exc:
        pytest.skip(f"Could not reach or migrate the configured PostgreSQL: {exc}")

    monkeypatch.setattr(
        workflow_module, "assess_severity", lambda description: _assessment(Severity.CRITICAL)
    )
    monkeypatch.setattr(
        workflow_module,
        "investigate",
        lambda description, scenario=None: _result(findings=[_finding()]),
    )
    monkeypatch.setattr(workflow_module, "analyze_root_cause", lambda result: _analysis())
    monkeypatch.setattr(
        workflow_module,
        "plan_remediation",
        lambda description, analysis: RemediationPlan(
            steps=[
                RemediationStep(
                    action_type=RemediationActionType.INCREASE_CONNECTION_POOL,
                    target_service="payment",
                    rationale="Relieve connection pressure",
                )
            ]
        ),
    )

    incident_id = f"TEST-{uuid.uuid4().hex[:10]}-API"
    client = TestClient(create_app(repository=PostgresIncidentRepository.from_url(db_url)))
    try:
        posted = client.post(
            "/incidents",
            json={
                "incident_description": "Database connection pool saturated",
                "scenario": "connection_pool_exhaustion",
                "post_remediation_scenario": "healthy",
                "incident_id": incident_id,
            },
        )
        assert posted.status_code == 200
        assert posted.json()["persisted"] is True

        fetched = client.get(f"/incidents/{incident_id}")
        assert fetched.status_code == 200
        body = fetched.json()
        assert body["status"] == "resolved" and body["final_outcome"] == "resolve"
        assert body["remediations"][0]["execution_status"] == "executed"
        assert body["stages"]
        assert client.get(f"/incidents/{incident_id}-nope").status_code == 404
    finally:
        with psycopg.connect(db_url) as conn:
            conn.execute("DELETE FROM incidents WHERE incident_id = %s", (incident_id,))


# --- migrate command --------------------------------------------------------------


def test_migrate_command_requires_the_migrate_argument(capsys) -> None:
    from agentic_ai.persistence.__main__ import main

    assert main([]) == 2
    assert main(["drop-everything"]) == 2
    assert "usage" in capsys.readouterr().err


def test_migrate_command_needs_a_database_url(monkeypatch, capsys) -> None:
    from agentic_ai.persistence.__main__ import main

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)

    assert main(["migrate"]) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_migrate_command_reports_applied_versions(monkeypatch, capsys) -> None:
    import psycopg

    from agentic_ai.persistence import __main__ as migrate_module

    class _Migration:
        version = "001"

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")

    class _Conn(FakeConnection):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(psycopg, "connect", lambda url: _Conn())
    monkeypatch.setattr(
        migrate_module.MigrationRunner, "apply_migrations", lambda self, conn: [_Migration()]
    )

    assert migrate_module.main(["migrate"]) == 0
    assert "Applied: 001" in capsys.readouterr().out


def test_migrate_command_does_not_echo_connection_details(monkeypatch, capsys) -> None:
    import psycopg

    from agentic_ai.persistence.__main__ import main

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:hunter2@localhost/db")

    def refuse(url):
        raise psycopg.OperationalError(f"could not connect using {url}")

    monkeypatch.setattr(psycopg, "connect", refuse)

    assert main(["migrate"]) == 1
    err = capsys.readouterr().err
    assert "hunter2" not in err and "OperationalError" in err
