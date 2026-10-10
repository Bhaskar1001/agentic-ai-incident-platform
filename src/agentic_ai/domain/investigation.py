"""Domain models produced by an incident investigation.

These models are deliberately free of LLM and framework imports: they describe
*what an investigation produced*, not how it was produced.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TerminationReason(str, Enum):
    """Why an investigation stopped.

    Deliberately separate from confidence: a truncated investigation and a
    low-confidence conclusion are different facts, and conflating them lets an
    incomplete result masquerade as a considered one.
    """

    COMPLETED = "completed"
    """The agent decided it had gathered sufficient evidence."""

    MAX_STEPS_REACHED = "max_steps_reached"
    """Deterministic code stopped the agent at its step budget."""

    TIMEOUT = "timeout"
    """Reserved. Not reachable until wall-clock enforcement is implemented."""

    TOOL_ERROR = "tool_error"
    """An unrecoverable tool failure. A single failed tool call must NOT
    produce this - that is a recoverable observation the agent reacts to."""


class TimelineEventType(str, Enum):
    """The kind of process event a timeline entry records."""

    INVESTIGATION_STARTED = "investigation_started"
    TOOL_SELECTED = "tool_selected"
    TOOL_SUCCEEDED = "tool_succeeded"
    TOOL_FAILED = "tool_failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVESTIGATION_CONCLUDED = "investigation_concluded"
    STEP_LIMIT_REACHED = "step_limit_reached"


class TimelineEntry(BaseModel):
    """An audit record of something that happened *during* the investigation.

    Distinct from Evidence: this describes the process, not the incident.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: TimelineEventType
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """An observation gathered from a tool.

    Invariant: evidence must always be traceable back to an unmodified tool
    result. ``raw_output`` is authoritative and is never rewritten by the LLM;
    ``observation`` is a rendering of it for readability.
    """

    id: UUID = Field(default_factory=uuid4)
    source_tool: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    raw_output: str
    observation: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Finding(BaseModel):
    """An inference drawn from evidence.

    A finding can be wrong while the evidence supporting it is entirely
    correct - which is precisely why the two are separate models.
    """

    id: UUID = Field(default_factory=uuid4)
    """Lets a later synthesis step (RCA) cite exactly which findings a
    conclusion rests on, rather than restating or re-deriving them. Added
    when RCA actually needed it (Phase 7) rather than speculatively in
    Phase 5, per supporting_evidence_ids having been deliberately deferred
    there until something concrete required it."""

    statement: str
    model_assessed_confidence: float = Field(ge=0.0, le=1.0)
    # Deferred: supporting_evidence_ids linking each finding to specific
    # Evidence. Needed for human review and evaluation; still premature.


class InvestigationResult(BaseModel):
    """The structured outcome of one investigation."""

    summary: str
    evidence: list[Evidence] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    model_assessed_confidence: float = Field(ge=0.0, le=1.0)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_action: str
    termination_reason: TerminationReason
    timeline: list[TimelineEntry] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True only if the agent concluded on its own terms."""
        return self.termination_reason is TerminationReason.COMPLETED
