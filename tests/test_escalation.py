"""Deterministic tests for the escalation policy.

No LLM involvement anywhere in this file. This is the safety boundary: if it is
wrong, a real incident can be silently downgraded to "monitor", so its behaviour
is pinned exactly - including at the threshold values themselves, where an
accidental > / >= swap would otherwise go unnoticed.
"""

import json

import pytest

from agentic_ai.domain.escalation import (
    CONNECTION_UTILISATION_ESCALATION_PERCENT,
    ERROR_RATE_ESCALATION_PERCENT,
    LATENCY_P95_ESCALATION_MS,
    UPSTREAM_ERROR_RATE_ESCALATION_PERCENT,
    EscalationDecision,
    evaluate,
)
from agentic_ai.domain.investigation import Evidence
from agentic_ai.tools.mock_tools import (
    Scenario,
    get_mock_logs,
    get_mock_metrics,
    get_mock_pod_status,
)


def _evidence(payload, tool: str = "get_mock_metrics") -> list[Evidence]:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return [Evidence(source_tool=tool, raw_output=raw, observation="")]


def _metrics(**metrics) -> list[Evidence]:
    return _evidence({"status": "success", "service": "payment", "metrics": metrics})


def _pods(pods: list[dict]) -> list[Evidence]:
    return _evidence(
        {"status": "success", "service": "payment", "pods": pods},
        tool="get_mock_pod_status",
    )


# --- thresholds: exact boundaries ---------------------------------------
#
# Each test restates the constant it depends on, so changing a threshold has
# to be a deliberate edit here rather than a silent behaviour drift.


@pytest.mark.parametrize(
    ("error_rate", "escalates"),
    [(4.99, False), (5.0, False), (5.01, True), (100.0, True)],
)
def test_error_rate_rule_is_strictly_greater_than(error_rate, escalates) -> None:
    assert ERROR_RATE_ESCALATION_PERCENT == 5.0

    assert evaluate(_metrics(error_rate_percent=error_rate)).escalate is escalates


@pytest.mark.parametrize(
    ("utilisation", "escalates"),
    [(89.99, False), (90.0, True), (90.01, True), (100, True)],
)
def test_pool_utilisation_rule_is_greater_or_equal(utilisation, escalates) -> None:
    assert CONNECTION_UTILISATION_ESCALATION_PERCENT == 90.0
    evidence = _metrics(database_connections={"utilization_percent": utilisation})

    assert evaluate(evidence).escalate is escalates


@pytest.mark.parametrize(
    ("p95", "escalates"),
    [(1999, False), (2000, False), (2001, True), (10_200, True)],
)
def test_p95_latency_rule_is_strictly_greater_than(p95, escalates) -> None:
    assert LATENCY_P95_ESCALATION_MS == 2_000.0

    assert evaluate(_metrics(latency_ms={"p95": p95})).escalate is escalates


@pytest.mark.parametrize(
    ("dependency_error_rate", "escalates"),
    [(9.99, False), (10.0, False), (10.01, True), (97.0, True)],
)
def test_upstream_error_rule_is_strictly_greater_than(
    dependency_error_rate, escalates
) -> None:
    assert UPSTREAM_ERROR_RATE_ESCALATION_PERCENT == 10.0
    evidence = _metrics(
        upstream_dependencies={
            "card-authorization-api": {
                "error_rate_percent": dependency_error_rate,
                "circuit_breaker": "closed",
            }
        }
    )

    assert evaluate(evidence).escalate is escalates


# --- non-threshold rules -------------------------------------------------


def test_open_circuit_breaker_escalates_regardless_of_error_rate() -> None:
    evidence = _metrics(
        upstream_dependencies={
            "ledger-api": {"error_rate_percent": 0.0, "circuit_breaker": "open"}
        }
    )

    decision = evaluate(evidence)

    assert decision.escalate
    assert "circuit breaker open for ledger-api" in decision.reasons


def test_closed_circuit_breaker_does_not_escalate() -> None:
    evidence = _metrics(
        upstream_dependencies={
            "ledger-api": {"error_rate_percent": 0.1, "circuit_breaker": "closed"}
        }
    )

    assert not evaluate(evidence).escalate


def test_a_single_unready_pod_escalates() -> None:
    decision = evaluate(_pods([{"ready": True}, {"ready": False}, {"ready": True}]))

    assert decision.escalate
    assert "1 of 3 pods not ready" in decision.reasons


def test_all_ready_pods_do_not_escalate() -> None:
    assert not evaluate(_pods([{"ready": True}, {"ready": True}])).escalate


def test_restart_count_alone_does_not_escalate() -> None:
    """An old restart is routine noise, not a reason to page someone."""
    assert not evaluate(_pods([{"ready": True, "restart_count": 3}])).escalate


# --- missing and malformed evidence --------------------------------------
#
# Absence of readable evidence must never be read as evidence of absence.


def test_no_evidence_escalates() -> None:
    decision = evaluate([])

    assert decision.escalate
    assert decision.reasons == ("no evidence was gathered",)


@pytest.mark.parametrize(
    "raw",
    ["", "not json", "[1, 2, 3]", '"a bare string"', "null"],
)
def test_unreadable_output_escalates(raw) -> None:
    decision = evaluate(_evidence(raw))

    assert decision.escalate
    assert any("could not read output" in reason for reason in decision.reasons)


def test_error_shaped_tool_output_escalates() -> None:
    """A tool that ran but reported failure is missing evidence, not good news."""
    decision = evaluate(
        _evidence({"status": "error", "error_type": "service_not_found"})
    )

    assert decision.escalate


def test_one_unreadable_source_escalates_even_when_others_are_healthy() -> None:
    healthy = Evidence(
        source_tool="get_mock_metrics",
        raw_output=get_mock_metrics("payment", Scenario.HEALTHY),
        observation="",
    )
    broken = Evidence(source_tool="get_mock_logs", raw_output="{{{", observation="")

    assert evaluate([healthy, broken]).escalate


def test_success_payload_without_metrics_or_pods_does_not_escalate() -> None:
    """Readable but topic-free output is not itself alarming."""
    evidence = _evidence({"status": "success", "service": "payment"})

    assert not evaluate(evidence).escalate


@pytest.mark.parametrize(
    "metrics",
    [
        {"error_rate_percent": "12.5"},
        {"error_rate_percent": None},
        {"database_connections": "not-a-dict"},
        {"latency_ms": ["not", "a", "dict"]},
        {"upstream_dependencies": "not-a-dict"},
        {"upstream_dependencies": {"api": "not-a-dict"}},
    ],
)
def test_wrongly_typed_metric_fields_are_ignored_not_crashed(metrics) -> None:
    """Type guards must hold: a malformed field must not raise."""
    assert not evaluate(_metrics(**metrics)).escalate


# --- multiple simultaneous reasons ---------------------------------------


def test_all_tripped_rules_are_reported_together() -> None:
    """The decision must explain every reason, not stop at the first."""
    evidence = _metrics(
        error_rate_percent=40.0,
        database_connections={"utilization_percent": 95},
        latency_ms={"p95": 9000},
        upstream_dependencies={
            "card-authorization-api": {
                "error_rate_percent": 97.0,
                "circuit_breaker": "open",
            }
        },
    )

    decision = evaluate(evidence)

    assert decision.escalate
    assert len(decision.reasons) == 5


# --- the real fixtures ---------------------------------------------------


@pytest.mark.parametrize(
    ("scenario", "escalates"),
    [
        (Scenario.CONNECTION_POOL_EXHAUSTION, True),
        (Scenario.UPSTREAM_DEPENDENCY_FAILURE, True),
        (Scenario.HEALTHY, False),
    ],
)
def test_policy_discriminates_the_real_scenarios(scenario, escalates) -> None:
    evidence = [
        Evidence(source_tool=name, raw_output=tool("payment", scenario), observation="")
        for name, tool in (
            ("get_mock_logs", get_mock_logs),
            ("get_mock_metrics", get_mock_metrics),
            ("get_mock_pod_status", get_mock_pod_status),
        )
    ]

    assert evaluate(evidence).escalate is escalates


def test_decision_is_falsy_when_not_escalating() -> None:
    assert not EscalationDecision(False)
    assert EscalationDecision(True, ("reason",))


def test_repeated_evidence_does_not_repeat_a_reason() -> None:
    """The agent sometimes calls the same tool twice despite being told not to.

    Observed live: get_mock_pod_status was called twice and the decision
    reported "2 of 3 pods not ready" twice, which reads as two problems.
    """
    pods = get_mock_pod_status("payment", Scenario.CONNECTION_POOL_EXHAUSTION)
    duplicated = [
        Evidence(source_tool="get_mock_pod_status", raw_output=pods, observation="")
        for _ in range(3)
    ]

    decision = evaluate(duplicated)

    assert decision.escalate
    assert decision.reasons == ("2 of 3 pods not ready",)


def test_distinct_reasons_from_different_tools_are_all_kept() -> None:
    """Deduplication must not collapse genuinely different observations."""
    scenario = Scenario.CONNECTION_POOL_EXHAUSTION
    evidence = [
        Evidence(source_tool=name, raw_output=tool("payment", scenario), observation="")
        for name, tool in (
            ("get_mock_metrics", get_mock_metrics),
            ("get_mock_pod_status", get_mock_pod_status),
        )
    ]

    decision = evaluate(evidence)

    assert len(decision.reasons) == len(set(decision.reasons))
    assert "2 of 3 pods not ready" in decision.reasons
    assert any("error rate" in reason for reason in decision.reasons)
