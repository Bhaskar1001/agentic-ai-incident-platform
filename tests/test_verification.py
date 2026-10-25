"""Tests for Phase 12 Step 1: post-remediation verification models and evaluator.

Covers:
- VerificationStatus values and exhaustiveness.
- VerificationResult construction, frozen immutability, timezone-aware timestamps,
  and JSON round-trip serialization.
- Bounded attempts (at most MAX_VERIFICATION_ATTEMPTS).
- Deterministic recovery evaluation:
  - Successful recovery (RECOVERED).
  - Failed recovery (UNRESOLVED).
  - Missing evidence (INCONCLUSIVE).
  - Missing required tools (INCONCLUSIVE).
  - Malformed or unreadable output (INCONCLUSIVE).
  - Contradictory evidence (UNRESOLVED).
  - Skipped verification (SKIPPED).
- verify_service_health tool integration with mock observability fixtures.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_ai.domain.investigation import Evidence
from agentic_ai.domain.verification import (
    MAX_VERIFICATION_ATTEMPTS,
    REQUIRED_VERIFICATION_TOOLS,
    VerificationResult,
    VerificationStatus,
    evaluate_verification,
)
from agentic_ai.tools.mock_tools import (
    Scenario,
    get_mock_metrics,
    get_mock_pod_status,
)
from agentic_ai.tools.mock_verification import verify_service_health


def _evidence_for(scenario: Scenario, service: str = "payment") -> list[Evidence]:
    return [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": service},
            raw_output=get_mock_metrics(service, scenario),
            observation="metrics check",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": service},
            raw_output=get_mock_pod_status(service, scenario),
            observation="pod check",
        ),
    ]


# --- VerificationStatus ---------------------------------------------------


def test_verification_status_values() -> None:
    assert VerificationStatus.RECOVERED == "recovered"
    assert VerificationStatus.UNRESOLVED == "unresolved"
    assert VerificationStatus.INCONCLUSIVE == "inconclusive"
    assert VerificationStatus.SKIPPED == "skipped"


def test_verification_status_is_exhaustive() -> None:
    assert set(VerificationStatus) == {
        VerificationStatus.RECOVERED,
        VerificationStatus.UNRESOLVED,
        VerificationStatus.INCONCLUSIVE,
        VerificationStatus.SKIPPED,
    }


# --- VerificationResult Model ---------------------------------------------


def test_verification_result_construction() -> None:
    ts = datetime.now(timezone.utc)
    res = VerificationResult(
        status=VerificationStatus.RECOVERED,
        attempts=1,
        reasons=["all recovery criteria met"],
        evidence=[],
        verified_at=ts,
    )
    assert res.status == VerificationStatus.RECOVERED
    assert res.is_recovered is True
    assert res.attempts == 1
    assert res.reasons == ["all recovery criteria met"]
    assert res.verified_at == ts


def test_verification_result_is_frozen() -> None:
    res = VerificationResult(status=VerificationStatus.RECOVERED, attempts=1)
    with pytest.raises(ValidationError):
        res.status = VerificationStatus.UNRESOLVED


def test_verification_result_timestamp_is_utc_aware() -> None:
    res = VerificationResult(status=VerificationStatus.RECOVERED, attempts=1)
    assert res.verified_at.tzinfo is not None


def test_verification_result_attempts_bound() -> None:
    # 0 is allowed for SKIPPED
    res_zero = VerificationResult(status=VerificationStatus.SKIPPED, attempts=0)
    assert res_zero.attempts == 0

    # Upper bound enforced by Pydantic
    with pytest.raises(ValidationError):
        VerificationResult(
            status=VerificationStatus.RECOVERED,
            attempts=MAX_VERIFICATION_ATTEMPTS + 1,
        )


def test_verification_result_serialisation_round_trip() -> None:
    evidence = _evidence_for(Scenario.HEALTHY)
    original = VerificationResult(
        status=VerificationStatus.RECOVERED,
        attempts=2,
        reasons=["all recovery criteria met"],
        evidence=evidence,
    )
    dumped = original.model_dump_json()
    restored = VerificationResult.model_validate_json(dumped)

    assert restored.status == original.status
    assert restored.attempts == original.attempts
    assert restored.reasons == original.reasons
    assert len(restored.evidence) == len(original.evidence)
    assert restored.verified_at == original.verified_at


# --- Deterministic evaluate_verification ----------------------------------


def test_healthy_evidence_evaluates_to_recovered() -> None:
    evidence = _evidence_for(Scenario.HEALTHY)
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.RECOVERED
    assert result.is_recovered is True
    assert result.attempts == 1
    assert "all recovery criteria met" in result.reasons


def test_active_incident_evaluates_to_unresolved() -> None:
    evidence = _evidence_for(Scenario.CONNECTION_POOL_EXHAUSTION)
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.UNRESOLVED
    assert result.is_recovered is False
    assert any("connection pool" in r for r in result.reasons)
    assert any("pods not ready" in r for r in result.reasons)


def test_upstream_failure_evaluates_to_unresolved() -> None:
    evidence = _evidence_for(Scenario.UPSTREAM_DEPENDENCY_FAILURE)
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.UNRESOLVED
    assert result.is_recovered is False
    assert any("error rate" in r for r in result.reasons)
    assert any("circuit breaker open" in r for r in result.reasons)


def test_empty_evidence_is_inconclusive() -> None:
    result = evaluate_verification([], attempts=1)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.is_recovered is False
    assert "no evidence was gathered" in result.reasons


def test_missing_required_tool_is_inconclusive() -> None:
    # Only pod status, missing metrics
    evidence = [
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_pod_status("payment", Scenario.HEALTHY),
            observation="pod check",
        )
    ]
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.is_recovered is False
    assert any("missing required verification tool" in r for r in result.reasons)
    assert "get_mock_metrics" in result.reasons[0]


def test_unreadable_or_malformed_evidence_is_inconclusive() -> None:
    evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": "payment"},
            raw_output="not valid json",
            observation="corrupt metrics",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_pod_status("payment", Scenario.HEALTHY),
            observation="pod check",
        ),
    ]
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.is_recovered is False
    assert any("could not read output" in r for r in result.reasons)


def test_error_shaped_tool_response_is_inconclusive() -> None:
    # Service not found returns a JSON with status: error
    evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": "unknown_svc"},
            raw_output='{"status": "error", "error_type": "service_not_found"}',
            observation="unknown service",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": "unknown_svc"},
            raw_output=get_mock_pod_status("payment", Scenario.HEALTHY),
            observation="pod check",
        ),
    ]
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert any("could not read output" in r for r in result.reasons)


def test_contradictory_evidence_does_not_declare_recovery() -> None:
    # Healthy pods (from healthy scenario) combined with high error rate metrics
    evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_metrics("payment", Scenario.UPSTREAM_DEPENDENCY_FAILURE),
            observation="metrics check",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": "payment"},
            raw_output=get_mock_pod_status("payment", Scenario.HEALTHY),
            observation="pod check",
        ),
    ]
    result = evaluate_verification(evidence, attempts=1)

    assert result.status == VerificationStatus.UNRESOLVED
    assert result.is_recovered is False


def test_attempts_exceeding_max_raises_value_error() -> None:
    evidence = _evidence_for(Scenario.HEALTHY)
    with pytest.raises(ValueError, match="must be between 0 and 2"):
        evaluate_verification(evidence, attempts=MAX_VERIFICATION_ATTEMPTS + 1)


# --- verify_service_health Tool Tests -------------------------------------


def test_verify_service_health_healthy_scenario() -> None:
    result = verify_service_health("payment", scenario=Scenario.HEALTHY, attempts=1)

    assert result.status == VerificationStatus.RECOVERED
    assert result.is_recovered is True
    assert len(result.evidence) == 2


def test_verify_service_health_unresolved_scenario() -> None:
    result = verify_service_health(
        "payment", scenario=Scenario.CONNECTION_POOL_EXHAUSTION, attempts=2
    )

    assert result.status == VerificationStatus.UNRESOLVED
    assert result.is_recovered is False
    assert result.attempts == 2


def test_verify_service_health_unknown_service_is_inconclusive() -> None:
    result = verify_service_health("unknown_service", scenario=Scenario.HEALTHY)

    assert result.status == VerificationStatus.INCONCLUSIVE
    assert result.is_recovered is False


def test_verify_service_health_bounds_attempts() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        verify_service_health("payment", attempts=3)
