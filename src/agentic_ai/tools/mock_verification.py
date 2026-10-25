"""Mock verification tool for post-remediation service health.

Collects observations from mock observability tools (get_mock_metrics,
get_mock_pod_status) and evaluates recovery criteria deterministically.
No real network calls or infrastructure checks.
"""

from agentic_ai.domain.investigation import Evidence
from agentic_ai.domain.verification import (
    MAX_VERIFICATION_ATTEMPTS,
    VerificationResult,
    evaluate_verification,
)
from agentic_ai.tools.mock_tools import (
    DEFAULT_SCENARIO,
    Scenario,
    get_mock_metrics,
    get_mock_pod_status,
)


def verify_service_health(
    service: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
    attempts: int = 1,
) -> VerificationResult:
    """Observe mock service health and evaluate recovery criteria.

    Bounded to at most MAX_VERIFICATION_ATTEMPTS.
    """
    if attempts > MAX_VERIFICATION_ATTEMPTS:
        raise ValueError(
            f"Verification attempts ({attempts}) cannot exceed "
            f"MAX_VERIFICATION_ATTEMPTS ({MAX_VERIFICATION_ATTEMPTS})."
        )

    try:
        metrics_raw = get_mock_metrics(service, scenario=scenario)
    except Exception as exc:
        metrics_raw = f'{{"status": "error", "error": "{exc}"}}'

    try:
        pods_raw = get_mock_pod_status(service, scenario=scenario)
    except Exception as exc:
        pods_raw = f'{{"status": "error", "error": "{exc}"}}'

    evidence = [
        Evidence(
            source_tool="get_mock_metrics",
            tool_arguments={"service": service},
            raw_output=metrics_raw,
            observation="Post-remediation metrics check",
        ),
        Evidence(
            source_tool="get_mock_pod_status",
            tool_arguments={"service": service},
            raw_output=pods_raw,
            observation="Post-remediation pod status check",
        ),
    ]

    return evaluate_verification(evidence, attempts=attempts)


__all__ = [
    "verify_service_health",
]

