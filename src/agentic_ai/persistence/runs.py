"""Persist a finished workflow run, and register the approvals it is waiting on.

Up to three writes, in this order: the incident record first (everything else
references it by foreign key); then, if a step is waiting on a human, the immutable
plan snapshot that resume will later run; then the pending approvals. They are
separate transactions, so a failure between them is possible, and the result says
exactly which part succeeded instead of claiming more than happened:

* everything written                     -> ``persisted=True``
* record saved, snapshot or approvals NOT written -> ``persisted=False`` with an
  error that says so. The incident exists but a human has nothing to approve until
  the same incident is submitted again; every write is idempotent, so that is safe.
* record not saved                      -> ``persisted=False``; nothing else is written.

Approvals are registered only while the incident is still open. A run that already
ended resolved, escalated or monitored has no execution left to authorize.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.domain.approvals import pending_requests_for
from agentic_ai.domain.guardrails import investigated_services
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.domain.plan_snapshots import build_snapshot
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

    # A plan that is waiting on a human must be resumable exactly as presented, so store it
    # (with the inputs the guardrails used) before any approval refers to it. The snapshot is
    # immutable and never overwrites: presenting the same plan again changes nothing.
    analysis = state.get("root_cause_analysis")
    if analysis is not None:
        try:
            store.save_plan_snapshot(
                build_snapshot(
                    record.incident_id,
                    guarded,
                    analysis,
                    sorted(investigated_services(state.get("investigation"))),
                )
            )
        except Exception as exc:
            logger.error(
                "Incident '%s' saved but its plan snapshot was not stored: %s",
                record.incident_id,
                type(exc).__name__,
            )
            return RunPersistence(
                persisted=False,
                error=(
                    f"record saved but plan snapshot not stored ({type(exc).__name__}); "
                    "approvals not registered; resubmitting the incident retries"
                ),
            )

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
