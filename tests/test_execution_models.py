"""Tests for Phase 11 execution-result domain models.

Covers construction, frozen-model behaviour, timezone-aware timestamps,
JSON round-trip serialisation, defaults, and the error field's optionality.
No LLM, no framework, purely deterministic.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationActionType,
    RemediationExecution,
    RemediationStep,
    StepExecutionResult,
)


def _step(
    action: str = "scale_up", service: str = "payment"
) -> RemediationStep:
    return RemediationStep(
        action_type=action, target_service=service, rationale="test"
    )


def _result(
    status: ExecutionStatus = ExecutionStatus.EXECUTED,
    output: str = "Scaled up.",
    error: str | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step=_step(), status=status, output=output, error=error
    )


# --- ExecutionStatus -------------------------------------------------------


def test_execution_status_values() -> None:
    assert ExecutionStatus.EXECUTED == "executed"
    assert ExecutionStatus.SKIPPED == "skipped"
    assert ExecutionStatus.BLOCKED == "blocked"
    assert ExecutionStatus.FAILED == "failed"


def test_execution_status_is_exhaustive() -> None:
    assert set(ExecutionStatus) == {
        ExecutionStatus.EXECUTED,
        ExecutionStatus.SKIPPED,
        ExecutionStatus.BLOCKED,
        ExecutionStatus.FAILED,
    }


# --- StepExecutionResult construction -------------------------------------


def test_step_execution_result_construction() -> None:
    ts = datetime.now(timezone.utc)
    result = StepExecutionResult(
        step=_step(),
        status=ExecutionStatus.EXECUTED,
        output="Scaled payment from 2 to 4 replicas.",
        executed_at=ts,
    )
    assert result.step.action_type == RemediationActionType.SCALE_UP
    assert result.status is ExecutionStatus.EXECUTED
    assert result.output == "Scaled payment from 2 to 4 replicas."
    assert result.error is None
    assert result.executed_at == ts


def test_step_execution_result_with_error() -> None:
    result = _result(
        status=ExecutionStatus.FAILED,
        output="Attempted restart.",
        error="Pod failed readiness check after restart.",
    )
    assert result.status is ExecutionStatus.FAILED
    assert result.error == "Pod failed readiness check after restart."


def test_step_execution_result_error_defaults_to_none() -> None:
    result = _result()
    assert result.error is None


# --- StepExecutionResult timestamps ----------------------------------------


def test_step_execution_result_timestamp_is_timezone_aware() -> None:
    result = _result()
    assert result.executed_at.tzinfo is not None


def test_step_execution_result_default_timestamp_is_utc() -> None:
    before = datetime.now(timezone.utc)
    result = _result()
    after = datetime.now(timezone.utc)
    assert before <= result.executed_at <= after


# --- StepExecutionResult is frozen -----------------------------------------


def test_step_execution_result_is_frozen() -> None:
    result = _result()
    with pytest.raises(ValidationError):
        result.status = ExecutionStatus.SKIPPED


def test_step_execution_result_output_is_frozen() -> None:
    result = _result()
    with pytest.raises(ValidationError):
        result.output = "tampered"


# --- RemediationExecution construction -------------------------------------


def test_remediation_execution_construction() -> None:
    ts = datetime.now(timezone.utc)
    r1 = _result(status=ExecutionStatus.EXECUTED, output="done")
    r2 = _result(status=ExecutionStatus.BLOCKED, output="blocked by policy")

    execution = RemediationExecution(
        step_results=[r1, r2], completed_at=ts
    )

    assert len(execution.step_results) == 2
    assert execution.step_results[0].status is ExecutionStatus.EXECUTED
    assert execution.step_results[1].status is ExecutionStatus.BLOCKED
    assert execution.completed_at == ts


def test_remediation_execution_defaults_to_empty_results() -> None:
    execution = RemediationExecution()
    assert execution.step_results == []


def test_remediation_execution_timestamp_is_timezone_aware() -> None:
    execution = RemediationExecution()
    assert execution.completed_at.tzinfo is not None


# --- RemediationExecution is frozen ----------------------------------------


def test_remediation_execution_is_frozen() -> None:
    execution = RemediationExecution(step_results=[_result()])
    with pytest.raises(ValidationError):
        execution.completed_at = datetime.now(timezone.utc)


# --- JSON round-trip -------------------------------------------------------


def test_step_execution_result_serialises_and_deserialises() -> None:
    original = _result(
        status=ExecutionStatus.FAILED,
        output="restart failed",
        error="timeout",
    )
    json_str = original.model_dump_json()
    restored = StepExecutionResult.model_validate_json(json_str)

    assert restored.status is ExecutionStatus.FAILED
    assert restored.output == "restart failed"
    assert restored.error == "timeout"
    assert restored.step.action_type == original.step.action_type
    assert restored.executed_at == original.executed_at


def test_remediation_execution_serialises_and_deserialises() -> None:
    r1 = _result(status=ExecutionStatus.EXECUTED, output="ok")
    r2 = _result(status=ExecutionStatus.SKIPPED, output="not needed")
    original = RemediationExecution(step_results=[r1, r2])

    json_str = original.model_dump_json()
    restored = RemediationExecution.model_validate_json(json_str)

    assert len(restored.step_results) == 2
    assert restored.step_results[0].status is ExecutionStatus.EXECUTED
    assert restored.step_results[1].status is ExecutionStatus.SKIPPED
    assert restored.completed_at == original.completed_at


# --- Validation ------------------------------------------------------------


def test_invalid_execution_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StepExecutionResult(
            step=_step(),
            status="magic",
            output="nope",
        )

