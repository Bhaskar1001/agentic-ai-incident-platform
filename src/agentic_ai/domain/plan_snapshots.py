"""Immutable snapshots of the remediation plan an approval was given for.

Why a snapshot exists
---------------------
An approval says "this exact step of this exact plan is allowed". To act on it later
without asking the LLM planner for a new plan, the plan itself has to be stored, not
re-derived. A snapshot is that stored plan together with the *inputs the guardrails
used* (the root cause confidence, whether a knowledge-base precedent agreed, and which
services were actually examined). With those, the deterministic guardrail policy can be
re-run on exactly the data the approver was looking at, with no model involved.

What identifies it
------------------
``plan_fingerprint`` is the same ordered ``(action_type, target_service)`` hash that
approvals are bound to (``domain.approvals.plan_fingerprint``), so approvals keep
matching. ``checksum`` additionally covers the guardrail inputs. Both are recomputed on
load (``verify``). The model's rationale prose and the root cause *text* are kept for
display only; like in Phase 17 they are excluded from identity, because they are not
execution parameters and would change on every re-plan.

Honest limit: a checksum is not a signature. Someone who can write the database can
change content and checksum together and the check cannot see it. What still holds is
that changing the *steps* changes the recomputed plan fingerprint, so a tampered plan
cannot inherit approvals recorded for the original, and the incident's own stored plan
must match it before anything runs. Guarding against a database-level attacker would
need a keyed signature, which this project does not have.

Free of LLM, database and framework imports.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_ai.domain.approvals import plan_fingerprint, plan_fingerprint_of
from agentic_ai.domain.guardrails import GuardedPlan
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationPlan, RemediationStep

SNAPSHOT_VERSION = 1


class SnapshotIntegrityError(Exception):
    """A stored snapshot is not what was originally stored, or cannot be understood."""


def new_snapshot_id() -> str:
    return f"SNP-{uuid4().hex[:12]}"


def compute_checksum(
    version: int,
    steps: Iterable[tuple[str, str]],
    confidence: float,
    consistency: str,
    investigated_services: Iterable[str],
) -> str:
    """Hash of everything execution-relevant, in a canonical form.

    Deliberately excludes rationale and root-cause prose (display only).
    """
    canonical = json.dumps(
        {
            "v": version,
            "steps": [[a, t] for a, t in steps],
            "confidence": confidence,
            "kb": consistency,
            "services": sorted(set(investigated_services)),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class SnapshotStep(BaseModel):
    action_type: str
    target_service: str
    rationale: str = ""
    """Display only; not part of the snapshot's identity."""

    model_config = ConfigDict(frozen=True)


class PlanSnapshot(BaseModel):
    """The plan as presented for approval, plus the inputs the guardrails used."""

    snapshot_id: str
    incident_id: str
    plan_fingerprint: str
    version: int = SNAPSHOT_VERSION
    steps: list[SnapshotStep]
    root_cause: str = ""
    """Display only; not part of the snapshot's identity."""
    root_cause_confidence: float = Field(ge=0.0, le=1.0)
    knowledge_base_consistency: KnowledgeBaseConsistency
    investigated_services: list[str]
    checksum: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    # -- integrity ---------------------------------------------------------------

    def _expected_checksum(self) -> str:
        return compute_checksum(
            self.version,
            ((s.action_type, s.target_service) for s in self.steps),
            self.root_cause_confidence,
            self.knowledge_base_consistency.value,
            self.investigated_services,
        )

    def verify(self) -> None:
        """Raise ``SnapshotIntegrityError`` unless this is exactly what was stored.

        Checked on every load. A snapshot that fails must not be executed and must not
        be treated as covered by any approval.
        """
        if self.version != SNAPSHOT_VERSION:
            raise SnapshotIntegrityError(
                f"Unsupported plan snapshot version {self.version} (this code reads "
                f"version {SNAPSHOT_VERSION})."
            )
        recomputed_fingerprint = plan_fingerprint(
            (s.action_type, s.target_service) for s in self.steps
        )
        if recomputed_fingerprint != self.plan_fingerprint:
            raise SnapshotIntegrityError(
                "Plan snapshot steps do not match their recorded plan fingerprint."
            )
        if self._expected_checksum() != self.checksum:
            raise SnapshotIntegrityError("Plan snapshot content does not match its checksum.")

    # -- rebuilding what the guardrails need -----------------------------------------

    def to_plan(self) -> RemediationPlan:
        """The stored plan as the domain type. Never involves the LLM."""
        try:
            return RemediationPlan(
                steps=[
                    RemediationStep(
                        action_type=s.action_type,
                        target_service=s.target_service,
                        rationale=s.rationale,
                    )
                    for s in self.steps
                ]
            )
        except ValidationError as exc:
            raise SnapshotIntegrityError(
                "Plan snapshot contains a step this code cannot represent."
            ) from exc

    def to_analysis(self) -> RootCauseAnalysis:
        """The guardrail inputs as the type ``guard_plan`` takes.

        Only the confidence and precedent consistency influence the policy; the text is
        carried for display.
        """
        return RootCauseAnalysis(
            root_cause=self.root_cause or "(not recorded)",
            root_cause_confidence=self.root_cause_confidence,
            knowledge_base_consistency=self.knowledge_base_consistency,
        )


def build_snapshot(
    incident_id: str,
    guarded: GuardedPlan,
    analysis: RootCauseAnalysis,
    investigated_services: Sequence[str],
) -> PlanSnapshot:
    """Snapshot the plan in ``guarded`` along with the inputs that guarded it."""
    steps = [
        SnapshotStep(
            action_type=getattr(g.step.action_type, "value", str(g.step.action_type)),
            target_service=g.step.target_service,
            rationale=g.step.rationale,
        )
        for g in guarded.steps
    ]
    services = sorted(set(investigated_services))
    return PlanSnapshot(
        snapshot_id=new_snapshot_id(),
        incident_id=incident_id,
        plan_fingerprint=plan_fingerprint_of(guarded),
        steps=steps,
        root_cause=analysis.root_cause,
        root_cause_confidence=analysis.root_cause_confidence,
        knowledge_base_consistency=analysis.knowledge_base_consistency,
        investigated_services=services,
        checksum=compute_checksum(
            SNAPSHOT_VERSION,
            ((s.action_type, s.target_service) for s in steps),
            analysis.root_cause_confidence,
            analysis.knowledge_base_consistency.value,
            services,
        ),
    )


__all__ = [
    "SNAPSHOT_VERSION",
    "PlanSnapshot",
    "SnapshotIntegrityError",
    "SnapshotStep",
    "build_snapshot",
    "compute_checksum",
    "new_snapshot_id",
]
