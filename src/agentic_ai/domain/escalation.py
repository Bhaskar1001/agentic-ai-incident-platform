"""Escalation policy: does this evidence warrant waking a human?

Deliberately deterministic and deliberately separate from the agent. The model
finds and explains problems; whether the numbers justify paging someone is a
policy decision, and policy reads the raw observations rather than the model's
characterisation of them.

That distinction is not academic. A live run produced the finding "the payment
service is operating with a low error rate of 34.0%" - the model read the
number correctly and then attached the wrong judgement to it. Code comparing
34.0 > 5.0 cannot make that mistake.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from agentic_ai.domain.investigation import Evidence

ERROR_RATE_ESCALATION_PERCENT = 5.0
"""Sustained error rate above this needs a human."""

CONNECTION_UTILISATION_ESCALATION_PERCENT = 90.0
"""A connection pool this saturated is about to start refusing work."""

LATENCY_P95_ESCALATION_MS = 2_000.0
"""p95 above this means a substantial share of users are visibly affected."""

UPSTREAM_ERROR_RATE_ESCALATION_PERCENT = 10.0
"""A dependency failing this often will take the service down with it."""


@dataclass(frozen=True)
class EscalationDecision:
    """Whether to escalate, and the specific observations that decided it."""

    escalate: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return self.escalate


def _parse(evidence: Evidence) -> dict[str, Any] | None:
    """Read a tool's raw output, or None if it is not usable JSON.

    Returning None means "this observation could not be read", which is very
    different from "this observation showed nothing wrong". ``evaluate``
    escalates on it rather than letting it pass silently.
    """
    try:
        payload = json.loads(evidence.raw_output)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict) or payload.get("status") != "success":
        return None

    return payload


def _check_metrics(payload: dict[str, Any]) -> list[str]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return []

    reasons: list[str] = []

    error_rate = metrics.get("error_rate_percent")
    if (
        isinstance(error_rate, (int, float))
        and error_rate > ERROR_RATE_ESCALATION_PERCENT
    ):
        reasons.append(
            f"error rate {error_rate}% exceeds "
            f"{ERROR_RATE_ESCALATION_PERCENT}%"
        )

    connections = metrics.get("database_connections")
    if isinstance(connections, dict):
        utilisation = connections.get("utilization_percent")
        if (
            isinstance(utilisation, (int, float))
            and utilisation >= CONNECTION_UTILISATION_ESCALATION_PERCENT
        ):
            reasons.append(
                f"connection pool {utilisation}% utilised, at or above "
                f"{CONNECTION_UTILISATION_ESCALATION_PERCENT}%"
            )

    latency = metrics.get("latency_ms")
    if isinstance(latency, dict):
        p95 = latency.get("p95")
        if isinstance(p95, (int, float)) and p95 > LATENCY_P95_ESCALATION_MS:
            reasons.append(
                f"p95 latency {p95}ms exceeds {LATENCY_P95_ESCALATION_MS}ms"
            )

    dependencies = metrics.get("upstream_dependencies")
    if isinstance(dependencies, dict):
        for name, dependency in dependencies.items():
            if not isinstance(dependency, dict):
                continue

            if dependency.get("circuit_breaker") == "open":
                reasons.append(f"circuit breaker open for {name}")

            dep_errors = dependency.get("error_rate_percent")
            if (
                isinstance(dep_errors, (int, float))
                and dep_errors > UPSTREAM_ERROR_RATE_ESCALATION_PERCENT
            ):
                reasons.append(f"{name} failing {dep_errors}% of requests")

    return reasons


def _check_pods(payload: dict[str, Any]) -> list[str]:
    pods = payload.get("pods")
    if not isinstance(pods, list):
        return []

    unready = [p for p in pods if isinstance(p, dict) and not p.get("ready", True)]
    if not unready:
        return []

    return [f"{len(unready)} of {len(pods)} pods not ready"]


def evaluate(evidence: list[Evidence]) -> EscalationDecision:
    """Decide escalation from raw tool output.

    Absence of readable evidence is never treated as evidence of absence. An
    incident where no data could be gathered, or where every tool returned
    something unreadable, is precisely one a human should look at - so both
    escalate rather than falling through to "nothing tripped a threshold".
    """
    if not evidence:
        return EscalationDecision(True, ("no evidence was gathered",))

    reasons: list[str] = []
    unreadable: list[str] = []

    for item in evidence:
        payload = _parse(item)
        if payload is None:
            unreadable.append(item.source_tool)
            continue
        reasons.extend(_check_metrics(payload))
        reasons.extend(_check_pods(payload))

    if unreadable:
        reasons.append(
            "could not read output from: " + ", ".join(sorted(set(unreadable)))
        )

    # The agent sometimes calls the same tool twice despite being asked not
    # to, which would otherwise report an identical observation more than
    # once. Reasons are a set of distinct facts, not a log of checks run -
    # a reader seeing the same line twice would suspect two problems.
    return EscalationDecision(bool(reasons), tuple(dict.fromkeys(reasons)))
