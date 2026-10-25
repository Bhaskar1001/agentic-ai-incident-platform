"""Evaluation framework for the AI Engineering & Incident Resolution Platform.

Provides golden evaluation datasets, evaluation models, deterministic
metric definitions, and an evaluation runner for assessing severity,
investigation, guardrail enforcement, remediation execution, and verification.
"""

from agentic_ai.evaluation.dataset import (
    CASE_MAP,
    GOLDEN_DATASET,
    get_case_by_id,
    get_cases_by_category,
)
from agentic_ai.evaluation.metrics import (
    INVARIANT_BLOCKED_OR_UNAPPROVED_EXECUTION,
    INVARIANT_INVALID_RESOLUTION,
    INVARIANT_REPEAT_REMEDIATION,
    INVARIANT_UNAUTHORIZED_EXECUTION,
    calculate_blocked_step_containment_rate,
    calculate_escalation_decision_accuracy,
    calculate_execution_authorization_rate,
    calculate_false_escalation_rate,
    calculate_false_recovery_rate,
    calculate_invariant_pass_rate,
    calculate_under_escalation_rate,
    check_all_invariants,
    check_blocked_or_unapproved_execution,
    check_case_invariants,
    check_no_invalid_resolution,
    check_no_repeat_remediation,
    check_unauthorized_execution,
    compute_evaluation_metrics,
)
from agentic_ai.evaluation.models import (
    ActualExecution,
    CaseEvaluationResult,
    ControlledFixture,
    EvalCase,
    EvalCategory,
    EvaluationMetrics,
    EvaluationReport,
    ExpectedOutcome,
    GateADecision,
    GateBOutcome,
    InvariantViolation,
)
from agentic_ai.evaluation.runner import (
    run_case,
    run_evaluation,
)

__all__ = [
    "CASE_MAP",
    "GOLDEN_DATASET",
    "INVARIANT_BLOCKED_OR_UNAPPROVED_EXECUTION",
    "INVARIANT_INVALID_RESOLUTION",
    "INVARIANT_REPEAT_REMEDIATION",
    "INVARIANT_UNAUTHORIZED_EXECUTION",
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
    "calculate_blocked_step_containment_rate",
    "calculate_escalation_decision_accuracy",
    "calculate_execution_authorization_rate",
    "calculate_false_escalation_rate",
    "calculate_false_recovery_rate",
    "calculate_invariant_pass_rate",
    "calculate_under_escalation_rate",
    "check_all_invariants",
    "check_blocked_or_unapproved_execution",
    "check_case_invariants",
    "check_no_invalid_resolution",
    "check_no_repeat_remediation",
    "check_unauthorized_execution",
    "compute_evaluation_metrics",
    "get_case_by_id",
    "get_cases_by_category",
    "run_case",
    "run_evaluation",
]
