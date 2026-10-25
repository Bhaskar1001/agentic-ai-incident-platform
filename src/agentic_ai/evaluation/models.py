"""Domain models for evaluating the Incident Resolution Platform.

Defines schemas for evaluation cases, expected outcomes, controlled fixtures,
actual execution outputs, and aggregate metric results.

All evaluation models are validated and immutable (frozen=True).
Controlled fixtures hold test inputs and dependency substitutions only;
they never author or inject final verdicts or routing outcomes.
Expected outcomes and actual execution results remain cleanly separated.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from agentic_ai.domain.guardrails import Disposition
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import InvestigationResult
from agentic_ai.domain.rca import RootCauseAnalysis
from agentic_ai.domain.remediation import (
    RemediationPlan,
    RemediationStep,
)
from agentic_ai.domain.verification import VerificationStatus


class EvalCategory(str, Enum):
    """Classification of evaluation scenarios."""

    NORMAL = "normal"
    """Benign or healthy operation; should never escalate to urgent or remediation."""

    DEGRADED = "degraded"
    """Active, verified incident requiring urgent investigation and remediation."""

    AMBIGUOUS = "ambiguous"
    """Conflicting signals, understated descriptions, or unreadable telemetry."""

    UNSAFE = "unsafe"
    """Proposals containing high-risk, unapproved, or uninvestigated actions."""


class GateADecision(str, Enum):
    """Intermediate routing decision after incident investigation."""

    URGENT = "urgent"
    """Requires root cause analysis, remediation planning, and verification."""

    NORMAL = "normal"
    """Benign; routes directly to continuous monitoring without remediation."""


class GateBOutcome(str, Enum):
    """Final workflow resolution outcome."""

    RESOLVE = "resolve"
    """Incident successfully addressed with independently verified recovery."""

    ESCALATE = "escalate"
    """Incident requires human intervention (unresolved, inconclusive, or skipped)."""

    MONITOR = "monitor"
    """Continuous passive monitoring (from Gate A normal path)."""


class ControlledFixture(BaseModel, frozen=True):
    """Controlled inputs and dependency substitutions for reproducible evaluation.

    Contains only test inputs or dependency stubs. Never injects final routing
    or verification results as actual results.
    """

    injected_investigation: InvestigationResult | None = None
    """Optional pre-computed investigation result to bypass or stub the ReAct loop."""

    injected_rca: RootCauseAnalysis | None = None
    """Optional pre-computed RCA for testing downstream guardrails and planning."""

    injected_plan: RemediationPlan | None = None
    """Optional pre-computed remediation plan for testing guardrail policies."""

    injected_severity: Severity | None = None
    """Optional pre-computed severity assessment to stub LLM severity evaluation."""

    telemetry_override: dict[str, str] | None = None
    """Optional mapping of tool name to raw response string (e.g. malformed JSON)."""

    mock_service: str | None = None
    """Optional target service override (defaults to 'payment')."""


class ExpectedOutcome(BaseModel, frozen=True):
    """Expected decisions, outcomes, and safety invariants for an evaluation case."""

    gate_a: GateADecision
    """Expected intermediate routing decision after investigation."""

    gate_b: GateBOutcome
    """Expected final outcome of the workflow."""

    verification_status: VerificationStatus | None = None
    """Expected verification status. None indicates verification was bypassed (e.g. monitor)."""

    expected_severity_band: list[Severity] | None = None
    """Acceptable severity levels for LLM-assessed severity."""

    expected_dispositions: dict[str, Disposition] | None = None
    """Expected guardrail dispositions keyed by action type string."""

    require_no_unauthorized_execution: bool = True
    """Safety invariant: no blocked or unapproved step may execute."""

    require_no_repeat_remediation: bool = True
    """Safety invariant: remediation execution must never run more than once."""


class EvalCase(BaseModel, frozen=True):
    """A single evaluation benchmark case."""

    case_id: str
    """Unique identifier for the evaluation case (e.g. 'CASE-NORM-01')."""

    name: str
    """Short human-readable title."""

    category: EvalCategory
    """Operational scenario category."""

    incident_description: str
    """The incident report or alert text fed into the workflow."""

    initial_scenario: str
    """Baseline scenario name for observability mock tools."""

    post_remediation_scenario: str | None = None
    """Optional scenario name for post-remediation verification observations."""

    fixture: ControlledFixture = Field(default_factory=ControlledFixture)
    """Controlled inputs and dependency stubs (cleanly separated from expectations)."""

    expected: ExpectedOutcome
    """Ground-truth expected outcome and safety constraints."""


class ActualExecution(BaseModel, frozen=True):
    """The actual observed decisions and execution trace from a workflow run."""

    gate_a: GateADecision | None = None
    gate_b: GateBOutcome | None = None
    severity: Severity | None = None
    verification_status: VerificationStatus | None = None
    verification_ran: bool = False
    remediation_executed_count: int = 0
    executed_steps: list[RemediationStep] = Field(default_factory=list)
    unauthorized_executions: list[RemediationStep] = Field(default_factory=list)
    step_dispositions: dict[str, Disposition] = Field(default_factory=dict)


class CaseEvaluationResult(BaseModel, frozen=True):
    """Evaluation result for a single case, comparing expected vs. actual."""

    case_id: str
    passed: bool
    expected: ExpectedOutcome
    actual: ActualExecution | None = None
    violations: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InvariantViolation(BaseModel, frozen=True):
    """A specific zero-tolerance safety invariant violation."""

    case_id: str
    invariant_name: str
    step: str | None = None
    details: str

    def __str__(self) -> str:
        step_part = f" (step: {self.step})" if self.step else ""
        return f"[{self.case_id}] {self.invariant_name}{step_part}: {self.details}"


class EvaluationMetrics(BaseModel, frozen=True):
    """Summary metrics across an evaluation run.

    Rates are set to None when their denominator is zero, avoiding misleading
    scores (e.g. 100% when no steps executed).
    """

    total_cases: int = 0
    passed_cases: int = 0

    escalation_decision_accuracy: float | None = None
    """Agreement between actual and expected Gate A decisions. None if 0 cases."""

    under_escalation_rate: float | None = None
    """Gate A under-escalations / total urgent expected cases. None if 0 urgent cases."""

    false_escalation_rate: float | None = None
    """Gate A false escalations / total normal expected cases. None if 0 normal cases."""

    false_recovery_rate: float | None = None
    """False recoveries / total expected unresolved or inconclusive cases."""

    execution_authorization_rate: float | None = None
    """Authorized executed steps / total executed steps. None if 0 executed steps."""

    blocked_step_containment_rate: float | None = None
    """Contained blocked steps / total proposed blocked steps. None if 0 blocked steps."""

    invariant_pass_rate: float | None = None
    """Proportion of evaluated invariants that passed. None if 0 invariants evaluated."""

    unauthorized_execution_count: int = 0
    """Strict zero-tolerance count of unauthorized executions."""

    repeat_remediation_violation_count: int = 0
    """Strict zero-tolerance count of repeated remediation execution runs."""

    invalid_resolution_count: int = 0
    """Strict zero-tolerance count of cases resolved without verified recovery."""

    all_invariants_passed: bool = True
    """True only if all zero-tolerance invariant counts are 0."""


class EvaluationReport(BaseModel, frozen=True):
    """Aggregate evaluation report covering an evaluation run."""

    passed: bool
    """Overall status: True only if passed_cases == total_cases and all_invariants_passed."""

    metrics: EvaluationMetrics
    case_results: list[CaseEvaluationResult]
    violations: list[InvariantViolation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary_text(self) -> str:
        """Render a readable console summary table."""
        lines: list[str] = [
            "=" * 78,
            f"EVALUATION REPORT: {'PASSED' if self.passed else 'FAILED'}",
            f"Generated at: {self.generated_at.isoformat()}",
            "-" * 78,
            (
                f"Total Cases: {self.metrics.total_cases} | "
                f"Passed: {self.metrics.passed_cases} | "
                f"Failed: {self.metrics.total_cases - self.metrics.passed_cases}"
            ),
            f"Escalation Decision Accuracy: {self.metrics.escalation_decision_accuracy}",
            f"Under-escalation Rate (Gate A): {self.metrics.under_escalation_rate}",
            f"False-escalation Rate (Gate A): {self.metrics.false_escalation_rate}",
            f"False-recovery Rate (Gate B):   {self.metrics.false_recovery_rate}",
            f"Execution Authorization Rate:  {self.metrics.execution_authorization_rate}",
            f"Blocked-step Containment Rate:  {self.metrics.blocked_step_containment_rate}",
            f"Invariant Pass Rate:            {self.metrics.invariant_pass_rate}",
            "-" * 78,
            "ZERO-TOLERANCE INVARIANTS:",
            f"  Unauthorized Executions:    {self.metrics.unauthorized_execution_count}",
            f"  Repeat Remediation Runs:    {self.metrics.repeat_remediation_violation_count}",
            f"  Invalid Resolutions:        {self.metrics.invalid_resolution_count}",
            f"  All Invariants Passed:      {self.metrics.all_invariants_passed}",
            "-" * 78,
            "PER-CASE OUTCOMES:",
        ]
        for res in self.case_results:
            status_tag = "PASS" if res.passed else "FAIL"
            act_gate_a = res.actual.gate_a.value if res.actual and res.actual.gate_a else "N/A"
            act_gate_b = res.actual.gate_b.value if res.actual and res.actual.gate_b else "N/A"
            exp_gate_a = res.expected.gate_a.value
            exp_gate_b = res.expected.gate_b.value
            lines.append(
                f"  [{status_tag}] {res.case_id}: "
                f"Gate A ({act_gate_a} vs exp {exp_gate_a}), "
                f"Gate B ({act_gate_b} vs exp {exp_gate_b})"
            )
            if res.violations:
                for v in res.violations:
                    lines.append(f"         * VIOLATION: {v}")
        if self.violations:
            lines.append("-" * 78)
            lines.append("INVARIANT VIOLATION DETAILS:")
            for v in self.violations:
                lines.append(f"  ! {v}")
        lines.append("=" * 78)
        return "\n".join(lines)


__all__ = [
    "ActualExecution",
    "CaseEvaluationResult",
    "ControlledFixture",
    "EvalCase",
    "EvalCategory",
    "EvaluationMetrics",
    "EvaluationReport",
    "ExpectedOutcome",
    "GateADecision",
    "GateBOutcome",
    "InvariantViolation",
]
