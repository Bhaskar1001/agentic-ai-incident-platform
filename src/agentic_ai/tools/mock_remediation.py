"""Deterministic mock remediation execution tools.

Executes remediation steps in simulation mode only. Outputs clearly state that
these are simulated actions. No real network calls, infrastructure modifications,
ticket creation, or state changes are performed.

Follows the project's core boundary:
- The LLM proposes actions.
- Deterministic code in domain/guardrails.py decides whether an action may run.
- This module executes only steps permitted by the guardrail policy and approval gate.
"""

from collections.abc import Callable

from agentic_ai.domain.guardrails import (
    ApprovalState,
    Disposition,
    GuardedPlan,
)
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationActionType,
    RemediationExecution,
    RemediationStep,
    StepExecutionResult,
)
from agentic_ai.tools.mock_tools import AVAILABLE_SERVICES

_SIMULATED_ACTION_OUTPUTS: dict[RemediationActionType, str] = {
    RemediationActionType.SCALE_UP: (
        "[SIMULATED] Scaled up service '{service}'. Increased replica count."
    ),
    RemediationActionType.CLEAR_CACHE: (
        "[SIMULATED] Cleared application cache for service '{service}'."
    ),
    RemediationActionType.INCREASE_CONNECTION_POOL: (
        "[SIMULATED] Increased database connection pool size for service '{service}'."
    ),
    RemediationActionType.RESTART_SERVICE: (
        "[SIMULATED] Initiated graceful rolling restart for service '{service}'."
    ),
    RemediationActionType.ROLLBACK_DEPLOYMENT: (
        "[SIMULATED] Rolled back deployment for service '{service}' to previous revision."
    ),
    RemediationActionType.FAILOVER: (
        "[SIMULATED] Executed failover for service '{service}' to standby instance."
    ),
    RemediationActionType.ROTATE_CERTIFICATE: (
        "[SIMULATED] Rotated TLS/SSL certificates for service '{service}'."
    ),
    RemediationActionType.ESCALATE_TO_TEAM: (
        "[SIMULATED] Escalated incident for service '{service}' to on-call engineering team."
    ),
}


def execute_mock_remediation_step(step: RemediationStep) -> StepExecutionResult:
    """Simulate execution of a single remediation step.

    Deterministic mock implementation that validates the target service and action type.
    Never modifies real infrastructure.
    """
    if step.target_service not in AVAILABLE_SERVICES:
        return StepExecutionResult(
            step=step,
            status=ExecutionStatus.FAILED,
            output=(
                f"[SIMULATED] Failed to execute {step.action_type.value} on service "
                f"'{step.target_service}': unknown service. "
                f"Available services: {', '.join(AVAILABLE_SERVICES)}."
            ),
            error=f"Service '{step.target_service}' is not a recognized service.",
        )

    template = _SIMULATED_ACTION_OUTPUTS.get(step.action_type)
    if template is None:
        return StepExecutionResult(
            step=step,
            status=ExecutionStatus.FAILED,
            output=f"[SIMULATED] Unsupported action type: '{step.action_type}'.",
            error=f"Unsupported action type: {step.action_type}",
        )

    return StepExecutionResult(
        step=step,
        status=ExecutionStatus.EXECUTED,
        output=template.format(service=step.target_service),
        error=None,
    )


def execute_guarded_plan(
    plan: GuardedPlan,
    *,
    step_executor: Callable[[RemediationStep], StepExecutionResult] = execute_mock_remediation_step,
) -> RemediationExecution:
    """Execute a guarded plan respecting deterministic guardrail and approval boundaries.

    Safety invariants enforced:
    - BLOCKED steps are never passed to the executor; recorded with ExecutionStatus.BLOCKED.
    - Steps with REQUIRES_APPROVAL that are PENDING or REJECTED are never passed
      to the executor; recorded with ExecutionStatus.SKIPPED.
    - Steps with REQUIRES_APPROVAL must explicitly be APPROVED (not inferred from may_execute alone).
    - Steps with ALLOWED disposition execute only if may_execute is True and approval is NOT_REQUIRED.
    """
    results: list[StepExecutionResult] = []

    for guarded in plan.steps:
        if guarded.disposition == Disposition.BLOCKED:
            reasons_text = "; ".join(guarded.reasons) if guarded.reasons else "Policy blocked step."
            results.append(
                StepExecutionResult(
                    step=guarded.step,
                    status=ExecutionStatus.BLOCKED,
                    output=f"[SIMULATED] Step execution blocked: {reasons_text}",
                    error=reasons_text,
                )
            )
        elif guarded.disposition == Disposition.REQUIRES_APPROVAL:
            if guarded.approval == ApprovalState.APPROVED:
                if guarded.may_execute:
                    results.append(step_executor(guarded.step))
                else:
                    results.append(
                        StepExecutionResult(
                            step=guarded.step,
                            status=ExecutionStatus.SKIPPED,
                            output="[SIMULATED] Step execution skipped: approved but not permitted to execute.",
                            error="Inconsistent approval/execution state.",
                        )
                    )
            elif guarded.approval == ApprovalState.REJECTED:
                decided_by = f" by {guarded.decided_by}" if guarded.decided_by else ""
                results.append(
                    StepExecutionResult(
                        step=guarded.step,
                        status=ExecutionStatus.SKIPPED,
                        output=f"[SIMULATED] Step execution skipped: rejected{decided_by}.",
                        error=f"Approval rejected{decided_by}.",
                    )
                )
            else:
                # PENDING or any unapproved state
                results.append(
                    StepExecutionResult(
                        step=guarded.step,
                        status=ExecutionStatus.SKIPPED,
                        output="[SIMULATED] Step execution skipped: awaiting human approval.",
                    )
                )
        elif guarded.disposition == Disposition.ALLOWED:
            if guarded.may_execute and guarded.approval == ApprovalState.NOT_REQUIRED:
                results.append(step_executor(guarded.step))
            else:
                results.append(
                    StepExecutionResult(
                        step=guarded.step,
                        status=ExecutionStatus.SKIPPED,
                        output="[SIMULATED] Step execution skipped: not permitted to execute.",
                        error="Step not permitted to execute.",
                    )
                )
        else:
            results.append(
                StepExecutionResult(
                    step=guarded.step,
                    status=ExecutionStatus.BLOCKED,
                    output=f"[SIMULATED] Step execution blocked: unknown disposition '{guarded.disposition}'.",
                    error=f"Unknown disposition: {guarded.disposition}",
                )
            )

    return RemediationExecution(step_results=results)


__all__ = [
    "execute_guarded_plan",
    "execute_mock_remediation_step",
]
