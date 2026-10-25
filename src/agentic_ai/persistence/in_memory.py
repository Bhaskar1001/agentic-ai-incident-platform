"""Thread-safe in-memory implementation of IncidentRepository (Phase 16).

Provides a deterministic, zero-dependency repository for unit tests, local
demonstrations, and offline validation.
"""

from __future__ import annotations

import threading
from typing import Sequence

from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRecord
from agentic_ai.persistence.repository import IncidentConflictError, IncidentRepository


class InMemoryIncidentRepository:
    """Thread-safe in-memory store satisfying the IncidentRepository protocol."""

    def __init__(self, initial_records: Sequence[IncidentRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._storage: dict[str, IncidentRecord] = {}
        if initial_records:
            for rec in initial_records:
                self.save_incident(rec)

    def save_incident(self, record: IncidentRecord) -> IncidentRecord:
        """Persist or update an incident record with terminal conflict protection.

        Parameters
        ----------
        record : IncidentRecord
            The incident record to store.

        Returns
        -------
        IncidentRecord
            The stored record.

        Raises
        ------
        IncidentConflictError
            If attempting to alter a terminal record's outcome, severity, or status.
        """
        with self._lock:
            existing = self._storage.get(record.incident_id)
            if existing is not None:
                # Check terminal record immutability rules
                if existing.status.is_terminal:
                    if not record.status.is_terminal:
                        raise IncidentConflictError(
                            f"Cannot revert terminal incident '{record.incident_id}' "
                            f"(status={existing.status.value}) to non-terminal status '{record.status.value}'."
                        )
                    if (
                        existing.final_outcome != record.final_outcome
                        or existing.severity != record.severity
                    ):
                        raise IncidentConflictError(
                            f"Cannot overwrite terminal incident '{record.incident_id}' with conflicting "
                            f"outcome ({existing.final_outcome} -> {record.final_outcome}) "
                            f"or severity ({existing.severity.value} -> {record.severity.value})."
                        )

            self._storage[record.incident_id] = record
            return record

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        """Retrieve an incident record by its unique ID."""
        with self._lock:
            return self._storage.get(incident_id)

    def get_by_correlation_id(self, correlation_id: str) -> list[IncidentRecord]:
        """Retrieve all incidents matching correlation_id, sorted by created_at DESC."""
        with self._lock:
            matching = [
                rec for rec in self._storage.values() if rec.correlation_id == correlation_id
            ]
            matching.sort(key=lambda r: (r.created_at, r.incident_id), reverse=True)
            return matching

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: IncidentLifecycleStatus | None = None,
    ) -> list[IncidentRecord]:
        """Retrieve a paginated slice of records, sorted by created_at DESC."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        with self._lock:
            records = list(self._storage.values())
            if status is not None:
                records = [r for r in records if r.status == status]

            records.sort(key=lambda r: (r.created_at, r.incident_id), reverse=True)
            return records[offset : offset + limit]

    def count(self, status: IncidentLifecycleStatus | None = None) -> int:
        """Return the count of incidents, optionally filtered by status."""
        with self._lock:
            if status is None:
                return len(self._storage)
            return sum(1 for r in self._storage.values() if r.status == status)

    def clear(self) -> None:
        """Reset internal storage (useful for test fixtures)."""
        with self._lock:
            self._storage.clear()


# Ensure implementation strictly satisfies protocol at runtime
assert issubclass(InMemoryIncidentRepository, IncidentRepository) or isinstance(
    InMemoryIncidentRepository(), IncidentRepository
)

__all__ = ["InMemoryIncidentRepository"]
