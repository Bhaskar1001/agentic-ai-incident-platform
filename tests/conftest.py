import pytest

from agentic_ai.tools.incident_memory import MEMORY_PATH_ENV


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Never let a test read or write the real incident memory file."""
    monkeypatch.setenv(MEMORY_PATH_ENV, str(tmp_path / "memory.jsonl"))


@pytest.fixture(params=["memory", "postgres"])
def incident_store(request):
    """The incident+approval store, once per backend.

    The PostgreSQL leg needs a real server (``DATABASE_URL``); without one it is
    skipped, visibly, never faked. Test incident ids all start with
    ``TEST-APR-`` so cleanup touches nothing else.
    """
    if request.param == "memory":
        from agentic_ai.persistence.in_memory import InMemoryIncidentRepository

        yield InMemoryIncidentRepository()
        return

    from agentic_ai.config import Settings

    database_url = Settings.from_env().get_database_url_value()
    if not database_url:
        pytest.skip("DATABASE_URL not set; the PostgreSQL leg is unverified.")
    try:
        import psycopg

        from agentic_ai.persistence.migrator import MigrationRunner

        with psycopg.connect(database_url) as conn:
            MigrationRunner().apply_migrations(conn)
    except Exception as exc:
        pytest.skip(f"Could not reach or migrate the configured PostgreSQL: {exc}")

    from agentic_ai.persistence.postgres import PostgresIncidentRepository

    yield PostgresIncidentRepository.from_url(database_url)

    with psycopg.connect(database_url) as conn:
        # Children first: approvals and snapshots reference incidents with ON DELETE RESTRICT.
        # (Ledger rows have no foreign key by design, but belong to the same test data.)
        conn.execute("DELETE FROM incident_approvals WHERE incident_id LIKE 'TEST-APR-%'")
        conn.execute("DELETE FROM incident_plan_snapshots WHERE incident_id LIKE 'TEST-APR-%'")
        conn.execute("DELETE FROM incident_step_executions WHERE incident_id LIKE 'TEST-APR-%'")
        conn.execute("DELETE FROM incidents WHERE incident_id LIKE 'TEST-APR-%'")
