"""Operational incident persistence module (Phase 16).

Exposes storage abstractions, repository protocols, implementations, and migration
tooling for operational incident management.
"""

from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from agentic_ai.persistence.migrator import (
    AppliedMigration,
    Migration,
    MigrationChecksumMismatchError,
    MigrationDiscoveryError,
    MigrationError,
    MigrationExecutionError,
    MigrationRunner,
    compute_checksum,
)
from agentic_ai.persistence.repository import (
    IncidentConflictError,
    IncidentNotFoundError,
    IncidentPersistenceError,
    IncidentRepository,
)

__all__ = [
    "AppliedMigration",
    "InMemoryIncidentRepository",
    "IncidentConflictError",
    "IncidentNotFoundError",
    "IncidentPersistenceError",
    "IncidentRepository",
    "Migration",
    "MigrationChecksumMismatchError",
    "MigrationDiscoveryError",
    "MigrationError",
    "MigrationExecutionError",
    "MigrationRunner",
    "compute_checksum",
]
