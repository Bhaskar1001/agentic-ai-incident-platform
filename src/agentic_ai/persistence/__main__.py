"""Apply database migrations: ``python -m agentic_ai.persistence migrate``.

Reads ``DATABASE_URL``. Safe to run repeatedly: already-applied migrations are
skipped, and a changed migration file (checksum mismatch) is refused.
"""

from __future__ import annotations

import sys

from agentic_ai.config import Settings
from agentic_ai.persistence.migrator import MigrationError, MigrationRunner


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["migrate"]:
        print("usage: python -m agentic_ai.persistence migrate", file=sys.stderr)
        return 2

    database_url = Settings.from_env().get_database_url_value()
    if not database_url:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 2

    import psycopg

    try:
        with psycopg.connect(database_url) as connection:
            applied = MigrationRunner().apply_migrations(connection)
    except MigrationError as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        # Report the failure class only: driver messages can echo connection details.
        print(f"Could not reach the database ({type(exc).__name__}).", file=sys.stderr)
        return 1

    if applied:
        print("Applied: " + ", ".join(m.version for m in applied))
    else:
        print("Database is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
