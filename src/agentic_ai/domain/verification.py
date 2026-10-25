"""Domain model and deterministic evaluation for post-remediation verification.

Defines the contract for verifying service health after a remediation action:
- Pure code, no LLM and no framework imports.
- Explicit deterministic recovery criteria based on observable facts.
- Distinguishes execution success (action was performed) from verified recovery
  (service returned to safe thresholds).
- Missing, malformed, or unreadable evidence is never treated as healthy;
  it produces an INCONCLUSIVE outcome and escalates.
- Bounded to at most MAX_VERIFICATION_ATTEMPTS (2) to prevent unbounded loops.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from agentic_ai.domain.escalation import evaluate
from agentic_ai.domain.investigation import Evidence

MAX_VERIFICATION_ATTEMPTS = 2
"""Upper bound on post-remediation verification observations."""

REQUIRED_VERIFICATION_TOOLS = ("get_mock_metrics", "get_mock_pod_status")
"""Observability sources required to confirm service recovery."""


class VerificationStatus(str, Enum):
    """The outcome of verifying post-remediation service health."""

    RECOVERED = "recovered"
    """All health criteria met: metrics normal, pods ready, no open circuit breakers."""

    UNRESOLVED = "unresolved"
    """Observable anomalies persist after remediation."""

    INCONCLUSIVE = "inconclusive"
    """Evidence is missing, unreadable, or insufficient to evaluate recovery."""

    SKIPPED = "skipped"
    """No remediation action was executed, so verification was not performed."""


class VerificationResult(BaseModel, frozen=True):
    """The outcome of post-remediation health verification."""

    status: VerificationStatus
    attempts: int = Field(default=1, ge=0, le=MAX_VERIFICATION_ATTEMPTS)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_recovered(self) -> bool:
        return self.status == VerificationStatus.RECOVERED


def evaluate_verification(
    evidence: list[Evidence],
    attempts: int = 1,
) -> VerificationResult:
    """Deterministically evaluate service recovery from post-remediation evidence.

    Invariants:
    - Empty evidence is INCONCLUSIVE.
    - Missing required observability tools is INCONCLUSIVE.
    - Unreadable or error-shaped tool outputs are INCONCLUSIVE.
    - Any health threshold violation (escalation rule) results in UNRESOLVED.
    - RECOVERED is only returned when all required tools are present, readable,
      and strictly within normal operational thresholds.
    """
    if not 0 <= attempts <= MAX_VERIFICATION_ATTEMPTS:
        raise ValueError(
            f"Verification attempts ({attempts}) must be between 0 and "
            f"{MAX_VERIFICATION_ATTEMPTS}."
        )

    if not evidence:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            attempts=attempts,
            reasons=["no evidence was gathered"],
            evidence=[],
        )

    present_tools = {item.source_tool for item in evidence}
    missing_required = set(REQUIRED_VERIFICATION_TOOLS) - present_tools
    if missing_required:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            attempts=attempts,
            reasons=[
                f"missing required verification tool(s): "
                f"{', '.join(sorted(missing_required))}"
            ],
            evidence=list(evidence),
        )

    decision = evaluate(evidence)

    unreadable_reasons = [
        r for r in decision.reasons if r.startswith("could not read output from:")
    ]
    if unreadable_reasons:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            attempts=attempts,
            reasons=list(decision.reasons),
            evidence=list(evidence),
        )

    if decision.escalate:
        return VerificationResult(
            status=VerificationStatus.UNRESOLVED,
            attempts=attempts,
            reasons=list(decision.reasons),
            evidence=list(evidence),
        )

    return VerificationResult(
        status=VerificationStatus.RECOVERED,
        attempts=attempts,
        reasons=["all recovery criteria met"],
        evidence=list(evidence),
    )


__all__ = [
    "MAX_VERIFICATION_ATTEMPTS",
    "REQUIRED_VERIFICATION_TOOLS",
    "VerificationResult",
    "VerificationStatus",
    "evaluate_verification",
]

