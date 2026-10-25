"""Persistence contract and exceptions for operational incident storage (Phase 16).

Defines the repository protocol and typed domain exceptions governing operational
incident persistence, independent of underlying storage technology (in-memory,
PostgreSQL, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentic_ai.domain.persistence import IncidentLifecycleStatus, IncidentRecord


class IncidentPersistenceError(Exception):
    """Base exception for all persistence layer errors."""


class IncidentNotFoundError(IncidentPersistenceError):
    """Raised when an incident record is expected but not found."""

    def __init__(self, incident_id: str) -> None:
        super().__init__(f"Incident '{incident_id}' not found.")
        self.incident_id = incident_id


class IncidentConflictError(IncidentPersistenceError):
    """Raised when an operation violates immutability or identity constraints."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@runtime_checkable
class IncidentRepository(Protocol):
    """Abstract persistence protocol for operational incident records.

    Implementations must support safe duplicate submissions, deterministic
    retrieval, lifecycle conflict validation, and thread safety.
    """

    def save_incident(self, record: IncidentRecord) -> IncidentRecord:
        """Persist or update an incident record.

        Parameters
        ----------
        record : IncidentRecord
            The incident record to store.

        Returns
        -------
        IncidentRecord
            The persisted incident record.

        Raises
        ------
        IncidentConflictError
            If attempting to modify a terminal incident with conflicting data.
        """
        ...

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        """Retrieve an incident record by its unique ID.

        Parameters
        ----------
        incident_id : str
            Unique incident identifier (e.g., 'INC-12345').

        Returns
        -------
        IncidentRecord | None
            The matching incident record, or None if not found.
        """
        ...

    def get_by_correlation_id(self, correlation_id: str) -> list[IncidentRecord]:
        """Retrieve all incident records associated with a correlation ID.

        Parameters
        ----------
        correlation_id : str
            Telemetry or trace correlation ID.

        Returns
        -------
        list[IncidentRecord]
            List of matching records ordered descending by created_at.
        """
        ...

    def list_incidents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: IncidentLifecycleStatus | None = None,
    ) -> list[IncidentRecord]:
        """Retrieve a paginated slice of incidents, optionally filtered by status.

        Parameters
        ----------
        limit : int
            Maximum number of records to return (must be >= 1).
        offset : int
            Number of records to skip (must be >= 0).
        status : IncidentLifecycleStatus | None
            Optional lifecycle status filter.

        Returns
        -------
        list[IncidentRecord]
            Slice of matching records ordered descending by created_at.
        """
        ...

    def count(self, status: IncidentLifecycleStatus | None = None) -> int:
        """Return the count of incidents, optionally filtered by status.

        Parameters
        ----------
        status : IncidentLifecycleStatus | None
            Optional lifecycle status filter.

        Returns
        -------
        int
            Number of matching incidents.
        """
        ...


__all__ = [
    "IncidentConflictError",
    "IncidentNotFoundError",
    "IncidentPersistenceError",
    "IncidentRepository",
]
