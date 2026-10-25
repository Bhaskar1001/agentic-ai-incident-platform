"""Persist a finished workflow run, and register the approvals it is waiting on.

Two writes, in this order: the incident record first (approvals reference it by
foreign key), then the pending approvals. They are separate transactions, so a
failure between them is possible, and the result says exactly which part
succeeded instead of claiming more than happened:

* record saved, approvals registered   -> ``persisted=True``
* record saved, approvals NOT registered -> ``persisted=False`` with an error that
  says so. The incident exists but a human has nothing to approve until the same
  incident is submitted again; registration is idempotent, so that is safe.
* record not saved                      -> ``persisted=False``; nothing is registered.

Approvals are registered only while the incident is still open. A run that already
ended resolved, escalated or monitored has no execution left to authorize.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.domain.approvals import pending_requests_for
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.persistence.repository import IncidentConflictError, IncidentStore

logger = logging.getLogger("agentic_ai.persistence")


@dataclass(frozen=True)
class RunPersistence:
    persisted: bool
    pending_approval_ids: list[str] = field(default_factory=list)
    error: str | None = None


def persist_workflow_run(store: IncidentStore, state: Mapping[str, Any]) -> RunPersistence:
    record = IncidentRecord.from_workflow_state(state)

    try:
        store.save_incident(record)
    except IncidentConflictError as exc:
        logger.error("Incident record conflicts with the stored one: %s", exc)
        return RunPersistence(persisted=False, error=f"conflict: {exc}")
    except Exception as exc:
        logger.error("Failed saving incident record: %s", type(exc).__name__)
        return RunPersistence(persisted=False, error=f"record not saved ({type(exc).__name__})")

    guarded = state.get("guarded_plan")
    if guarded is None or record.status.is_terminal:
        return RunPersistence(persisted=True)

    requests = pending_requests_for(record.incident_id, guarded)
    if not requests:
        return RunPersistence(persisted=True)

    try:
        stored = store.request_approvals(requests)
    except Exception as exc:
        logger.error(
            "Incident '%s' saved but its approvals were not registered: %s",
            record.incident_id,
            type(exc).__name__,
        )
        return RunPersistence(
            persisted=False,
            error=(
                f"record saved but approvals not registered ({type(exc).__name__}); "
                "resubmitting the incident retries registration"
            ),
        )

    return RunPersistence(
        persisted=True,
        pending_approval_ids=[a.approval_id for a in stored if a.is_pending],
    )


__all__ = ["RunPersistence", "persist_workflow_run"]
