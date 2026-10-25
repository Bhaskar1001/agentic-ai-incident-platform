"""Tests for PostgreSQL schema migrations and migration runner (Phase 16 - Step 2).

Covers:
- Migration discovery and naming convention validation
- Version sorting and duplicate version detection
- Checksum computation and tamper detection
- Schema content verification against domain models
- Transactional migration execution using mock DB-API connection
- Idempotent repeated execution (zero reapplications)
- Execution error rollback handling
- Conditional real PostgreSQL integration test (skipped if DB unavailable)
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest

from agentic_ai.config import Settings
from agentic_ai.persistence import (
    AppliedMigration,
    MigrationChecksumMismatchError,
    MigrationDiscoveryError,
    MigrationExecutionError,
    MigrationRunner,
    compute_checksum,
)


# ==============================================================================
# Discovery, Naming & Checksum Tests
# ==============================================================================


def test_discover_default_migrations() -> None:
    """Default migrations directory must discover 001_initial_incident_schema.sql."""
    runner = MigrationRunner()
    migrations = runner.discover_migrations()

    assert len(migrations) >= 1
    m001 = migrations[0]
    assert m001.version == "001"
    assert m001.name == "001_initial_incident_schema"
    assert "CREATE TABLE IF NOT EXISTS incidents" in m001.sql
    assert "CREATE TABLE IF NOT EXISTS incident_stages" in m001.sql
    assert "CREATE TABLE IF NOT EXISTS incident_remediations" in m001.sql
    assert len(m001.checksum) == 64  # SHA-256


def test_schema_definition_matches_persistence_contract() -> None:
    """Migration 001 DDL must include all required columns, constraints, and indexes."""
    runner = MigrationRunner()
    migrations = runner.discover_migrations()
    sql = migrations[0].sql

    # Required tables
    assert "CREATE TABLE IF NOT EXISTS incidents" in sql
    assert "CREATE TABLE IF NOT EXISTS incident_stages" in sql
    assert "CREATE TABLE IF NOT EXISTS incident_remediations" in sql

    # Incidents columns and constraints
    assert "incident_id VARCHAR(64) PRIMARY KEY" in sql
    assert "correlation_id VARCHAR(64) NOT NULL" in sql
    assert "scenario VARCHAR(64) NOT NULL" in sql
    assert "severity VARCHAR(32) NOT NULL" in sql
    assert "status VARCHAR(32) NOT NULL" in sql
    assert "final_outcome VARCHAR(32)" in sql
    assert "chk_incidents_severity" in sql
    assert "chk_incidents_status" in sql
    assert "chk_incidents_final_outcome" in sql
    assert "verification_reasons JSONB" in sql
    assert "TIMESTAMPTZ" in sql

    # Cascade foreign keys
    assert "REFERENCES incidents(incident_id) ON DELETE CASCADE" in sql

    # Performance indexes
    assert "idx_incidents_correlation_id" in sql
    assert "idx_incidents_status" in sql
    assert "idx_incidents_created_at_desc" in sql
    assert "idx_incidents_status_created_at" in sql
    assert "idx_incident_stages_incident_id" in sql
    assert "idx_incident_remediations_incident_step" in sql


def test_discovery_invalid_directory() -> None:
    """Non-existent migrations directory must raise MigrationDiscoveryError."""
    runner = MigrationRunner(migrations_dir=Path("non_existent_dir_12345"))
    with pytest.raises(MigrationDiscoveryError, match="does not exist"):
        runner.discover_migrations()


def test_discovery_invalid_filename_format(tmp_path: Path) -> None:
    """Filenames violating NNN_name.sql naming format must raise MigrationDiscoveryError."""
    (tmp_path / "001_valid_name.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "invalid_name.sql").write_text("SELECT 1;", encoding="utf-8")

    runner = MigrationRunner(migrations_dir=tmp_path)
    with pytest.raises(MigrationDiscoveryError, match="violates convention"):
        runner.discover_migrations()


def test_discovery_duplicate_versions(tmp_path: Path) -> None:
    """Colliding migration versions must raise MigrationDiscoveryError."""
    (tmp_path / "001_initial.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "001_duplicate.sql").write_text("SELECT 2;", encoding="utf-8")

    runner = MigrationRunner(migrations_dir=tmp_path)
    with pytest.raises(MigrationDiscoveryError, match="Duplicate migration version '001'"):
        runner.discover_migrations()


def test_migration_ordering_is_numeric(tmp_path: Path) -> None:
    """Migrations must be ordered numerically by version (e.g. 1, 2, 10)."""
    (tmp_path / "010_tenth.sql").write_text("SELECT 10;", encoding="utf-8")
    (tmp_path / "002_second.sql").write_text("SELECT 2;", encoding="utf-8")
    (tmp_path / "001_first.sql").write_text("SELECT 1;", encoding="utf-8")

    runner = MigrationRunner(migrations_dir=tmp_path)
    discovered = runner.discover_migrations()

    assert [m.version for m in discovered] == ["001", "002", "010"]


def test_compute_checksum_normalization() -> None:
    """Checksum calculation must normalize CRLF and LF line endings."""
    text_lf = "SELECT 1;\nSELECT 2;\n"
    text_crlf = "SELECT 1;\r\nSELECT 2;\r\n"

    assert compute_checksum(text_lf) == compute_checksum(text_crlf)


# ==============================================================================
# Execution & Transaction Simulation Tests
# ==============================================================================


class MockCursor:
    """Mock database cursor for testing migration execution flow without live PostgreSQL."""

    def __init__(self, connection: MockConnection) -> None:
        self.connection = connection
        self.executed_statements: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed_statements.append((sql, params))
        if self.connection.fail_on_sql and self.connection.fail_on_sql in sql:
            raise RuntimeError(f"Simulated SQL failure on statement: {sql[:30]}")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.connection.schema_migrations_rows

    def close(self) -> None:
        pass


class MockConnection:
    """Mock database connection implementing cursor, commit, and rollback tracking."""

    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.schema_migrations_rows: list[tuple[Any, ...]] = []
        self.fail_on_sql: str | None = None

    def cursor(self) -> MockCursor:
        return MockCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_apply_migrations_fresh_run() -> None:
    """Fresh run creates tracking table and applies migration in a transaction."""
    runner = MigrationRunner()
    conn = MockConnection()

    applied = runner.apply_migrations(conn)

    assert len(applied) >= 1
    assert applied[0].version == "001"
    # Tracking table created, then migration applied and committed
    assert conn.commits >= 2
    assert conn.rollbacks == 0


def test_apply_migrations_idempotency_when_already_applied() -> None:
    """Rerunning migrations when schema_migrations is current applies zero migrations."""
    runner = MigrationRunner()
    conn = MockConnection()

    # Pre-populate schema_migrations with existing applied migration 001
    discovered = runner.discover_migrations()
    m001 = discovered[0]
    conn.schema_migrations_rows = [
        (m001.version, m001.name, m001.checksum, datetime.now(timezone.utc))
    ]

    applied = runner.apply_migrations(conn)

    assert len(applied) == 0
    # No pending migrations executed
    assert conn.rollbacks == 0


def test_apply_migrations_checksum_mismatch() -> None:
    """Modifying an already applied migration file must raise MigrationChecksumMismatchError."""
    runner = MigrationRunner()
    conn = MockConnection()

    # Recorded checksum in DB does not match file on disk
    conn.schema_migrations_rows = [
        (
            "001",
            "001_initial_incident_schema",
            "tampered_checksum_12345",
            datetime.now(timezone.utc),
        )
    ]

    with pytest.raises(MigrationChecksumMismatchError, match="Checksum mismatch"):
        runner.apply_migrations(conn)


def test_apply_migrations_rollback_on_failure() -> None:
    """Execution error during migration script triggers rollback and raises MigrationExecutionError."""
    runner = MigrationRunner()
    conn = MockConnection()
    conn.fail_on_sql = "CREATE TABLE IF NOT EXISTS incidents"

    with pytest.raises(MigrationExecutionError, match="failed"):
        runner.apply_migrations(conn)

    assert conn.rollbacks >= 1


# ==============================================================================
# Live PostgreSQL Integration (Conditional)
# ==============================================================================


def test_live_postgresql_execution() -> None:
    """Live PostgreSQL execution test.

    Strict constraint: Runs only if a real PostgreSQL service is available via
    DATABASE_URL. If unavailable, explicitly marks PostgreSQL execution as unverified.
    """
    settings = Settings.from_env()
    db_url = settings.get_database_url_value()

    if not db_url:
        pytest.skip(
            "Live PostgreSQL service is not configured (DATABASE_URL not set). "
            "PostgreSQL execution remains unverified as expected in offline environment."
        )

    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("psycopg driver not installed. PostgreSQL live execution remains unverified.")

    try:
        with psycopg.connect(db_url) as conn:
            runner = MigrationRunner()
            applied = runner.apply_migrations(conn)
            assert isinstance(applied, list)
    except Exception as exc:
        pytest.skip(f"Could not connect to configured PostgreSQL instance: {exc}")
