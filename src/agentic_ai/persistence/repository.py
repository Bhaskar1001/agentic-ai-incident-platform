"""Persistence contract and exceptions for operational incident storage (Phase 16).

Defines the repository protocol and typed domain exceptions governing operational
incident persistence, independent of underlying storage technology (in-memory,
PostgreSQL, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentic_ai.domain.approvals import ApprovalRecord
from agentic_ai.domain.guardrails import ApprovalState
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


class ApprovalNotFoundError(IncidentPersistenceError):
    """No such approval *for that incident*.

    An approval that exists but belongs to a different incident raises this too:
    the caller is told the same thing either way, so one incident's URL cannot be
    used to probe another incident's approvals.
    """

    def __init__(self, incident_id: str, approval_id: str) -> None:
        super().__init__(f"Approval '{approval_id}' not found for incident '{incident_id}'.")
        self.incident_id = incident_id
        self.approval_id = approval_id


class ApprovalConflictError(IncidentPersistenceError):
    """The approval exists but cannot be decided now.

    Already decided, the incident is closed, or the remediation plan it referred
    to was replaced. Never overwritten, never retried.
    """


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


@runtime_checkable
class ApprovalRepository(Protocol):
    """Storage for persisted approvals of high-risk remediation steps.

    Implementations must keep approvals tied to an existing incident, make a
    decision atomic (exactly one decision per approval, ever), and behave
    identically across backends.
    """

    def request_approvals(self, approvals: Sequence[ApprovalRecord]) -> list[ApprovalRecord]:
        """Store pending approvals, idempotently.

        An approval for the same ``(incident, plan, step)`` that already exists is
        kept exactly as it is, never reset to pending, and returned in place of
        the new one. Results are in input order.

        Raises
        ------
        IncidentNotFoundError
            If an approval refers to an incident that is not stored.
        """
        ...

    def get_approval(self, incident_id: str, approval_id: str) -> ApprovalRecord | None:
        """Fetch one approval, or None if it does not exist *for that incident*."""
        ...

    def list_approvals(
        self, incident_id: str, status: ApprovalState | None = None
    ) -> list[ApprovalRecord]:
        """An incident's approvals, oldest request first, optionally by status."""
        ...

    def list_pending_approvals(self, limit: int = 50) -> list[ApprovalRecord]:
        """Pending approvals across all incidents, oldest request first."""
        ...

    def decide_approval(
        self,
        incident_id: str,
        approval_id: str,
        outcome: ApprovalState,
        approver: str,
        reason: str | None = None,
    ) -> ApprovalRecord:
        """Approve or reject a pending approval, atomically.

        Raises
        ------
        IncidentNotFoundError
            If the incident does not exist.
        ApprovalNotFoundError
            If the approval does not exist for that incident.
        ApprovalConflictError
            If it is not pending, the incident is closed, or its plan was replaced.
        ValueError
            If ``outcome`` is not approved/rejected or ``approver`` is blank.
        """
        ...


@runtime_checkable
class IncidentStore(IncidentRepository, ApprovalRepository, Protocol):
    """What the API needs: incident records and their approvals in one store."""


__all__ = [
    "ApprovalConflictError",
    "ApprovalNotFoundError",
    "ApprovalRepository",
    "IncidentConflictError",
    "IncidentNotFoundError",
    "IncidentPersistenceError",
    "IncidentRepository",
    "IncidentStore",
]
