"""Domain model for remediation planning and execution results.

A remediation plan is a *proposal*: nothing in the planning phase executes
anything. Two decisions are made structurally rather than left to the model:

- The model chooses from a closed menu of action types. An action outside the
  menu is invalid structured output, not a creative suggestion.
- The risk tier of a step is derived in code from its action type. The model
  never authors it, for the same reason it never authors ``evidence`` or the
  escalation decision: a safety-relevant label should not be something a
  confident-sounding narrative can talk its way into.

Phase 11 adds execution-result models: ``ExecutionStatus``,
``StepExecutionResult``, and ``RemediationExecution``. These record what
happened when each guarded step was acted on (or deliberately not acted on).
They do not implement the executor itself.

Free of LLM and framework imports.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class RemediationActionType(str, Enum):
    SCALE_UP = "scale_up"
    CLEAR_CACHE = "clear_cache"
    INCREASE_CONNECTION_POOL = "increase_connection_pool"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    FAILOVER = "failover"
    ROTATE_CERTIFICATE = "rotate_certificate"
    ESCALATE_TO_TEAM = "escalate_to_team"
    """Nothing safe to automate; hand the specific problem to a human team."""


class RiskTier(str, Enum):
    """Mirrors the Phase 10 guardrail model:
    LOW automatic, MEDIUM validated, HIGH human approval, CRITICAL blocked."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_ORDER = (RiskTier.LOW, RiskTier.MEDIUM, RiskTier.HIGH, RiskTier.CRITICAL)

RISK_BY_ACTION: dict[RemediationActionType, RiskTier] = {
    RemediationActionType.SCALE_UP: RiskTier.LOW,
    RemediationActionType.CLEAR_CACHE: RiskTier.LOW,
    RemediationActionType.ESCALATE_TO_TEAM: RiskTier.LOW,
    RemediationActionType.INCREASE_CONNECTION_POOL: RiskTier.MEDIUM,
    RemediationActionType.RESTART_SERVICE: RiskTier.MEDIUM,
    RemediationActionType.ROLLBACK_DEPLOYMENT: RiskTier.HIGH,
    RemediationActionType.FAILOVER: RiskTier.HIGH,
    RemediationActionType.ROTATE_CERTIFICATE: RiskTier.HIGH,
}


class RemediationStep(BaseModel):
    action_type: RemediationActionType
    target_service: str
    rationale: str
    """Why this action follows from the root cause."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_tier(self) -> RiskTier:
        return RISK_BY_ACTION[self.action_type]


class RemediationPlan(BaseModel):
    steps: list[RemediationStep] = Field(default_factory=list)
    """Ordered. Empty when no step is justified by the root cause."""

    @property
    def highest_risk(self) -> RiskTier | None:
        if not self.steps:
            return None
        return max((s.risk_tier for s in self.steps), key=_RISK_ORDER.index)

    @property
    def requires_human_approval(self) -> bool:
        risk = self.highest_risk
        return risk is not None and _RISK_ORDER.index(risk) >= _RISK_ORDER.index(RiskTier.HIGH)


class ExecutionStatus(str, Enum):
    """What happened when a guarded step was acted on."""

    EXECUTED = "executed"
    """The mock executor ran the action successfully."""

    SKIPPED = "skipped"
    """The step was not attempted (e.g. not approved, or not required)."""

    BLOCKED = "blocked"
    """The guardrail policy forbade execution outright."""

    FAILED = "failed"
    """The executor attempted the action and it did not succeed."""

    UNKNOWN = "unknown"
    """An attempt was recorded as started but its outcome was never recorded (or the
    executor raised), so whether the action took effect cannot be known. The step is not
    replayed automatically: replaying a non-idempotent action could do harm."""


class StepExecutionResult(BaseModel, frozen=True):
    """The outcome of acting (or declining to act) on one guarded step.

    Frozen for the same reason ``GuardedStep`` is: once the system has
    recorded what it did, rewriting that record is a bug, not a feature.
    """

    step: RemediationStep
    status: ExecutionStatus
    output: str
    """Deterministic description of what the executor did or why it did not."""
    error: str | None = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RemediationExecution(BaseModel, frozen=True):
    """The complete record of executing (or declining) a guarded plan."""

    step_results: list[StepExecutionResult] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "ExecutionStatus",
    "RISK_BY_ACTION",
    "RemediationActionType",
    "RemediationExecution",
    "RemediationPlan",
    "RemediationStep",
    "RiskTier",
    "StepExecutionResult",
]
