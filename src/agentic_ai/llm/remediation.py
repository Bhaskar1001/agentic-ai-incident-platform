"""Remediation planning: a single structured-output call over an RCA.

Like RCA, this is a one-shot capability, not an agent: no tools, no loop.
It proposes steps from a closed menu of action types; risk tiers are derived
in code from the action type (see domain/remediation.py), never authored by
the model. Nothing here executes anything.
"""

from pydantic import ValidationError

from agentic_ai.domain.rca import RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationActionType, RemediationPlan
from agentic_ai.llm.client import LLMClient, get_client
from agentic_ai.tools.mock_tools import AVAILABLE_SERVICES


class RemediationPlanningError(Exception):
    """Base exception for remediation planning failures."""


class InvalidRemediationPlanError(RemediationPlanningError):
    """The model repeatedly failed to produce a valid plan."""


_INSTRUCTION = (
    "You are proposing remediation steps for an incident whose root cause "
    "has already been analysed. You have no tools and nothing you propose "
    "is executed automatically - a human or a guarded executor decides.\n\n"
    "INCIDENT: {description}\n\n"
    "ROOT CAUSE (confidence {confidence}): {root_cause}\n\n"
    "Respond as JSON with a `steps` list, in the order they should be "
    "taken. Each step has:\n"
    "- action_type: exactly one of: {actions}\n"
    "- target_service: exactly one of: {services}\n"
    "- rationale: one sentence on why this action follows from the root "
    "cause.\n"
    "Propose only steps the root cause justifies; prefer the least "
    "disruptive action that addresses it. If the root cause is unclear, "
    "propose a single escalate_to_team step. Do not include a risk level - "
    "it is assigned separately."
)


def plan_remediation(
    incident_description: str,
    analysis: RootCauseAnalysis,
    *,
    client: LLMClient | None = None,
) -> RemediationPlan:
    """Propose a remediation plan for an analysed incident."""
    client = client or get_client()
    schema = RemediationPlan.model_json_schema()
    prompt = _INSTRUCTION.format(
        description=incident_description,
        confidence=analysis.root_cause_confidence,
        root_cause=analysis.root_cause,
        actions=", ".join(a.value for a in RemediationActionType),
        services=", ".join(AVAILABLE_SERVICES),
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.complete(
                messages=[{"role": "user", "content": prompt}],
                json_schema=schema,
            )
            plan = RemediationPlan.model_validate_json(response.content)

            unknown = {
                s.target_service
                for s in plan.steps
                if s.target_service not in AVAILABLE_SERVICES
            }
            if unknown:
                raise ValueError(f"Unknown target service(s): {sorted(unknown)}")

            return plan

        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == 1:
                raise InvalidRemediationPlanError(
                    "LLM produced an invalid remediation plan after retry."
                ) from last_error
