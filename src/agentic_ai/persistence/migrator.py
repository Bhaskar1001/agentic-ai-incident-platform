"""PostgreSQL schema migration engine and version tracker (Phase 16 - Step 2).

Provides deterministic migration discovery, checksum verification, atomic transactional
execution, and schema version tracking using the `schema_migrations` audit table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

from agentic_ai.persistence.repository import IncidentPersistenceError


class MigrationError(IncidentPersistenceError):
    """Base exception for database schema migration errors."""


class MigrationDiscoveryError(MigrationError):
    """Raised when migration files cannot be discovered or violate naming rules."""


class MigrationChecksumMismatchError(MigrationError):
    """Raised when an applied migration's checksum differs from the file on disk."""


class MigrationExecutionError(MigrationError):
    """Raised when a migration script fails to execute in the database."""


@dataclass(frozen=True)
class Migration:
    """Represents a discovered SQL migration script."""

    version: str
    name: str
    path: Path
    sql: str
    checksum: str


@dataclass(frozen=True)
class AppliedMigration:
    """Represents a migration recorded in the schema_migrations audit table."""

    version: str
    name: str
    checksum: str
    applied_at: datetime


_MIGRATION_FILENAME_PATTERN = re.compile(r"^([0-9]{3,})_([a-zA-Z0-9_-]+)\.sql$")

SCHEMA_MIGRATIONS_TABLE_DDL = """CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def compute_checksum(sql_content: str) -> str:
    """Compute sha256 checksum for normalized SQL migration content."""
    # Normalize line endings to LF before hashing for cross-platform consistency
    normalized = sql_content.strip().replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


class MigrationRunner:
    """Discovers, validates, and executes versioned PostgreSQL migrations."""

    def __init__(self, migrations_dir: Path | None = None) -> None:
        if migrations_dir is None:
            migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir = Path(migrations_dir)

    def discover_migrations(self) -> list[Migration]:
        """Discover and validate all versioned SQL files in the migrations directory.

        Returns
        -------
        list[Migration]
            Discovered migrations ordered ascending by version.

        Raises
        ------
        MigrationDiscoveryError
            If directory does not exist, file naming is invalid, or versions collide.
        """
        if not self.migrations_dir.exists() or not self.migrations_dir.is_dir():
            raise MigrationDiscoveryError(
                f"Migrations directory does not exist: {self.migrations_dir}"
            )

        sql_files = list(self.migrations_dir.glob("*.sql"))
        if not sql_files:
            return []

        migrations: list[Migration] = []
        seen_versions: dict[str, Path] = {}

        for file_path in sql_files:
            match = _MIGRATION_FILENAME_PATTERN.match(file_path.name)
            if not match:
                raise MigrationDiscoveryError(
                    f"Migration filename '{file_path.name}' violates convention. "
                    "Expected format: 'NNN_name.sql' (e.g. '001_initial_incident_schema.sql')."
                )

            version = match.group(1)
            name = file_path.stem

            if version in seen_versions:
                raise MigrationDiscoveryError(
                    f"Duplicate migration version '{version}' detected in files: "
                    f"'{seen_versions[version].name}' and '{file_path.name}'."
                )
            seen_versions[version] = file_path

            sql_content = file_path.read_text(encoding="utf-8")
            checksum = compute_checksum(sql_content)

            migrations.append(
                Migration(
                    version=version,
                    name=name,
                    path=file_path,
                    sql=sql_content,
                    checksum=checksum,
                )
            )

        migrations.sort(key=lambda m: (int(m.version), m.version))
        return migrations

    def ensure_tracking_table(self, connection: Any) -> None:
        """Create schema_migrations table if not already present."""
        cursor = connection.cursor()
        try:
            cursor.execute(SCHEMA_MIGRATIONS_TABLE_DDL)
            connection.commit()
        except Exception as err:
            connection.rollback()
            raise MigrationExecutionError(
                f"Failed to initialize schema_migrations table: {err}"
            ) from err
        finally:
            cursor.close()

    def get_applied_migrations(self, connection: Any) -> list[AppliedMigration]:
        """Fetch all applied migrations from schema_migrations, ordered by version ASC."""
        self.ensure_tracking_table(connection)
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version ASC;"
            )
            rows = cursor.fetchall()
            applied = [
                AppliedMigration(
                    version=str(row[0]),
                    name=str(row[1]),
                    checksum=str(row[2]),
                    applied_at=(
                        row[3] if isinstance(row[3], datetime) else datetime.now(timezone.utc)
                    ),
                )
                for row in rows
            ]
            return applied
        except Exception as err:
            raise MigrationExecutionError(f"Failed to query schema_migrations: {err}") from err
        finally:
            cursor.close()

    def get_pending_migrations(
        self,
        connection: Any | None = None,
        applied_versions: set[str] | None = None,
    ) -> list[Migration]:
        """Identify discovered migrations that have not yet been applied."""
        discovered = self.discover_migrations()
        if applied_versions is None and connection is not None:
            applied = self.get_applied_migrations(connection)
            applied_versions = {m.version for m in applied}
        elif applied_versions is None:
            applied_versions = set()

        return [m for m in discovered if m.version not in applied_versions]

    def apply_migrations(self, connection: Any) -> list[Migration]:
        """Apply all pending migrations in ascending order inside transactions.

        Validates checksums of previously applied migrations before executing
        pending scripts.

        Parameters
        ----------
        connection : Any
            Active database connection supporting cursor(), commit(), and rollback().

        Returns
        -------
        list[Migration]
            List of newly applied Migration instances.

        Raises
        ------
        MigrationChecksumMismatchError
            If an already applied migration's checksum changed on disk.
        MigrationExecutionError
            If SQL execution fails. The active migration is rolled back.
        """
        discovered = self.discover_migrations()
        applied = self.get_applied_migrations(connection)
        applied_by_version = {m.version: m for m in applied}

        # 1. Verify checksum integrity of previously applied migrations
        discovered_by_version = {m.version: m for m in discovered}
        for version, app_mig in applied_by_version.items():
            disc_mig = discovered_by_version.get(version)
            if disc_mig is not None and disc_mig.checksum != app_mig.checksum:
                raise MigrationChecksumMismatchError(
                    f"Checksum mismatch for applied migration '{version}' ({app_mig.name}). "
                    f"Database checksum: {app_mig.checksum}, Local file checksum: {disc_mig.checksum}. "
                    "Modifying applied migration files violates schema immutability."
                )

        # 2. Identify pending migrations in deterministic ascending order
        pending = [m for m in discovered if m.version not in applied_by_version]
        if not pending:
            return []

        applied_now: list[Migration] = []

        # 3. Apply each pending migration in an atomic transaction
        for migration in pending:
            cursor = connection.cursor()
            try:
                # Execute DDL
                cursor.execute(migration.sql)

                # Record migration in schema_migrations
                insert_sql = (
                    "INSERT INTO schema_migrations (version, name, checksum, applied_at) "
                    "VALUES (%s, %s, %s, CURRENT_TIMESTAMP);"
                )
                cursor.execute(
                    insert_sql,
                    (migration.version, migration.name, migration.checksum),
                )
                connection.commit()
                applied_now.append(migration)
            except Exception as err:
                connection.rollback()
                raise MigrationExecutionError(
                    f"Migration '{migration.name}' (version {migration.version}) failed: {err}"
                ) from err
            finally:
                cursor.close()

        return applied_now


__all__ = [
    "AppliedMigration",
    "Migration",
    "MigrationChecksumMismatchError",
    "MigrationDiscoveryError",
    "MigrationError",
    "MigrationExecutionError",
    "MigrationRunner",
    "SCHEMA_MIGRATIONS_TABLE_DDL",
    "compute_checksum",
]
