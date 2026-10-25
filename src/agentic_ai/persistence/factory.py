"""Choose the incident repository from configuration."""

from __future__ import annotations

from agentic_ai.config import Settings
from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from agentic_ai.persistence.postgres import PostgresIncidentRepository
from agentic_ai.persistence.repository import IncidentRepository


def build_repository(settings: Settings) -> IncidentRepository:
    """PostgreSQL when ``DATABASE_URL`` is set, otherwise process-local memory.

    The in-memory fallback keeps local runs and tests dependency-free, but its
    contents vanish when the process exits, so it is not an audit trail.
    """
    database_url = settings.get_database_url_value()
    if database_url:
        return PostgresIncidentRepository.from_url(database_url)
    return InMemoryIncidentRepository()
