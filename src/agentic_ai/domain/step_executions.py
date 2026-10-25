"""The execution ledger: a durable record that a step was attempted, and how it ended.

The problem this solves
-----------------------
Remediation is a side effect. If the system cannot tell whether a step already ran, a
retry, a resubmission or a crash-and-resume can run it twice. So every execution path
goes through one durable record per ``(incident, plan, step)``:

1. ``STARTED`` is written *before* the action runs (intent, committed on its own).
2. The action runs.
3. The record is completed as ``SUCCEEDED`` or ``FAILED``.

At most one record can ever exist per step (a uniqueness rule in storage), which is what
prevents duplicate execution: the second caller loses the claim and runs nothing.

What this does and does not guarantee
-------------------------------------
It guarantees the system never *starts* a step it has already started. It does **not**
guarantee exactly-once execution against an external system: if the process dies after
the action ran but before the record was completed, the record stays ``STARTED`` and the
real outcome is unknown. That state is surfaced as unknown and is never replayed
automatically, because replaying a non-idempotent action could do harm. A step that
ended ``FAILED`` is likewise not retried automatically.

Free of LLM, database and framework imports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepExecutionError(Exception):
    """An execution-record transition the lifecycle does not permit."""


class StepExecutionStatus(str, Enum):
    STARTED = "started"
    """Intent recorded; the action may or may not have run. Never finished = unknown."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExecutionSource(str, Enum):
    WORKFLOW = "workflow"
    """Executed by a run of the incident workflow (POST /incidents)."""
    RESUME = "resume"
    """Executed by resuming a stored plan (POST /incidents/{id}/resume)."""


def new_execution_id() -> str:
    return f"EXE-{uuid4().hex[:12]}"


class StepExecutionRecord(BaseModel):
    execution_id: str
    incident_id: str
    plan_fingerprint: str
    step_index: int = Field(ge=0)
    step_fingerprint: str
    action_type: str
    target_service: str
    status: StepExecutionStatus = StepExecutionStatus.STARTED
    source: ExecutionSource
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    output: str | None = None
    error: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _lifecycle_is_coherent(self) -> StepExecutionRecord:
        started = self.status == StepExecutionStatus.STARTED
        if started and self.finished_at is not None:
            raise ValueError("A started execution cannot have a finish time.")
        if not started and self.finished_at is None:
            raise ValueError("A finished execution must record when it finished.")
        return self

    @property
    def is_unknown(self) -> bool:
        """Started and never finished: the real outcome cannot be known from here."""
        return self.status == StepExecutionStatus.STARTED

    def finish(
        self,
        status: StepExecutionStatus,
        output: str | None,
        error: str | None,
        *,
        at: datetime | None = None,
    ) -> StepExecutionRecord:
        """The one transition: STARTED -> SUCCEEDED or FAILED. Nothing else is allowed."""
        if status == StepExecutionStatus.STARTED:
            raise StepExecutionError("An execution can only be finished as succeeded or failed.")
        if self.status != StepExecutionStatus.STARTED:
            raise StepExecutionError(
                f"Execution '{self.execution_id}' is already {self.status.value}; "
                "a finished execution is never rewritten."
            )
        return self.model_copy(
            update={
                "status": status,
                "finished_at": at or datetime.now(timezone.utc),
                "output": output,
                "error": error,
            }
        )


__all__ = [
    "ExecutionSource",
    "StepExecutionError",
    "StepExecutionRecord",
    "StepExecutionStatus",
    "new_execution_id",
]
