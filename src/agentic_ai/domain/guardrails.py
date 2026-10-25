"""Guardrails: deterministic policy deciding what a remediation step may do.

Pure code, no LLM and no framework, for the same reason escalation is: a
safety decision must read facts, not be persuaded by narrative. The model
proposes a step; this module decides whether it may run, needs a human, or is
blocked.

Tier policy (Phase 10 model)::

    LOW      -> allowed
    MEDIUM   -> allowed only if it passes validation, else needs approval
    HIGH     -> needs human approval
    CRITICAL -> blocked; cannot be approved, a human must act outside the system

One asymmetry runs through all of it: weak signals can **raise** a step's
risk tier, never lower it. A low-confidence or precedent-contradicted root
cause makes the system more cautious; a confident-sounding one earns no
discount. The starting tier is the code-assigned one from
``RISK_BY_ACTION``; nothing the model says about its own proposal changes it
downward.

Nothing here executes anything. ``may_execute`` is what a Phase 11 executor
must check before acting.
"""

from collections.abc import Collection
from enum import Enum

from pydantic import BaseModel, Field

from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationPlan, RemediationStep, RiskTier

LOW_CONFIDENCE_THRESHOLD = 0.6
"""Below this RCA confidence every step is treated as one tier riskier."""

_TIER_ORDER = (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL)


class GuardrailError(Exception):
    """An approval operation the policy does not permit."""


class Disposition(str, Enum):
    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    BLOCKED = "blocked"


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def _raise_tier(tier: RiskTier) -> RiskTier:
    index = _TIER_ORDER.index(tier)
    return _TIER_ORDER[min(index + 1, len(_TIER_ORDER) - 1)]


class GuardedStep(BaseModel, frozen=True):
    """A proposed step plus the policy's verdict on it."""

    step: RemediationStep
    effective_tier: RiskTier
    disposition: Disposition
    reasons: list[str] = Field(default_factory=list)
    approval: ApprovalState = ApprovalState.NOT_REQUIRED
    decided_by: str | None = None

    @property
    def may_execute(self) -> bool:
        if self.disposition == Disposition.ALLOWED:
            return True
        return (
            self.disposition == Disposition.REQUIRES_APPROVAL
            and self.approval == ApprovalState.APPROVED
        )


class GuardedPlan(BaseModel, frozen=True):
    steps: list[GuardedStep] = Field(default_factory=list)

    @property
    def pending_approvals(self) -> list[int]:
        return [i for i, s in enumerate(self.steps) if s.approval == ApprovalState.PENDING]

    def approve(self, index: int, approver: str) -> "GuardedPlan":
        return self._decide(index, approver, ApprovalState.APPROVED)

    def reject(self, index: int, approver: str) -> "GuardedPlan":
        return self._decide(index, approver, ApprovalState.REJECTED)

    def _decide(self, index: int, approver: str, outcome: ApprovalState) -> "GuardedPlan":
        if not approver.strip():
            raise GuardrailError("An approval decision must name who made it.")
        if not 0 <= index < len(self.steps):
            raise GuardrailError(f"No step at index {index}.")

        guarded = self.steps[index]
        if guarded.disposition == Disposition.BLOCKED:
            raise GuardrailError("A blocked step cannot be approved or rejected.")
        if guarded.approval != ApprovalState.PENDING:
            raise GuardrailError(
                f"Step {index} is not awaiting a decision (state: {guarded.approval.value})."
            )

        decided = guarded.model_copy(update={"approval": outcome, "decided_by": approver.strip()})
        steps = list(self.steps)
        steps[index] = decided
        return self.model_copy(update={"steps": steps})


def _validation_failures(
    step: RemediationStep, investigated_services: Collection[str]
) -> list[str]:
    """Checks a MEDIUM step must pass to proceed without a human.

    Kept to facts the system can check itself: the step must target a
    service the investigation actually looked at. A fix aimed at something
    nobody examined is a guess.
    """
    if step.target_service not in investigated_services:
        return [f"{step.target_service!r} was not examined during the investigation"]
    return []


def evaluate_step(
    step: RemediationStep,
    analysis: RootCauseAnalysis,
    investigated_services: Collection[str],
) -> GuardedStep:
    tier = step.risk_tier
    reasons: list[str] = []

    if analysis.root_cause_confidence < LOW_CONFIDENCE_THRESHOLD:
        tier = _raise_tier(tier)
        reasons.append(
            f"root cause confidence {analysis.root_cause_confidence:.2f} is below "
            f"{LOW_CONFIDENCE_THRESHOLD}: risk raised"
        )
    if analysis.knowledge_base_consistency == KnowledgeBaseConsistency.CONTRADICTED:
        tier = _raise_tier(tier)
        reasons.append("a knowledge base precedent contradicts the root cause: risk raised")

    if tier == RiskTier.CRITICAL:
        return GuardedStep(
            step=step,
            effective_tier=tier,
            disposition=Disposition.BLOCKED,
            reasons=[*reasons, "critical risk: a human must act outside the system"],
        )

    if tier == RiskTier.HIGH:
        return GuardedStep(
            step=step,
            effective_tier=tier,
            disposition=Disposition.REQUIRES_APPROVAL,
            reasons=[*reasons, "high risk: human approval required"],
            approval=ApprovalState.PENDING,
        )

    if tier == RiskTier.MEDIUM:
        failures = _validation_failures(step, investigated_services)
        if failures:
            return GuardedStep(
                step=step,
                effective_tier=tier,
                disposition=Disposition.REQUIRES_APPROVAL,
                reasons=[*reasons, *(f"validation failed: {f}" for f in failures)],
                approval=ApprovalState.PENDING,
            )
        return GuardedStep(
            step=step,
            effective_tier=tier,
            disposition=Disposition.ALLOWED,
            reasons=[*reasons, "medium risk: validation passed"],
        )

    return GuardedStep(
        step=step,
        effective_tier=tier,
        disposition=Disposition.ALLOWED,
        reasons=reasons or ["low risk"],
    )


def guard_plan(
    plan: RemediationPlan,
    analysis: RootCauseAnalysis,
    investigated_services: Collection[str],
) -> GuardedPlan:
    return GuardedPlan(
        steps=[evaluate_step(s, analysis, investigated_services) for s in plan.steps]
    )
