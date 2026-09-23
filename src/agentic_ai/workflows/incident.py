"""Incident response workflow.

Orchestrates the pieces into one flow::

    assess severity -> investigate -> route on severity
                                        |- HIGH/CRITICAL -> escalate
                                        `- LOW/MEDIUM    -> monitor

Every incident is investigated before routing, so an escalation always carries
evidence with it. Deciding what to *do* about an incident is a separate concern
from finding out what is wrong with it.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_ai.agents.investigation import InvestigationError, investigate
from agentic_ai.domain.escalation import evaluate
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import InvestigationResult
from agentic_ai.llm.severity import SeverityAssessment, assess_severity
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario

ESCALATION_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})


class IncidentWorkflowState(TypedDict):
    """State threaded through the incident workflow.

    Holds the *incident description* rather than an ``Incident`` entity:
    orchestration state and domain entities have different lifecycles, and the
    entity's lifecycle invariants should not be run through graph state
    merging. A persisted version would carry an ``incident_id`` instead.
    """

    incident_description: str
    scenario: str
    severity_assessment: SeverityAssessment | None
    investigation: InvestigationResult | None
    investigation_error: str | None
    next_step: str | None


def assess_severity_node(
    state: IncidentWorkflowState,
) -> dict[str, SeverityAssessment]:
    """Classify how serious the incident is, from its description alone."""
    return {"severity_assessment": assess_severity(state["incident_description"])}


def investigate_node(state: IncidentWorkflowState) -> dict[str, object]:
    """Run the investigation agent to gather evidence and form findings.

    A failed investigation must not sink the workflow: the incident still
    needs routing, and an unexplained incident is precisely one a human should
    see. The error is recorded and the flow continues.
    """
    try:
        result = investigate(
            state["incident_description"],
            scenario=state["scenario"],
        )
    except InvestigationError as exc:
        return {"investigation": None, "investigation_error": str(exc)}

    return {"investigation": result, "investigation_error": None}


def route_by_severity(
    state: IncidentWorkflowState,
) -> Literal["urgent", "normal"]:
    """Decide what happens now that the incident has been investigated.

    Deterministic on purpose, and based on the *evidence* rather than the
    model's reading of it. Severity is assessed from the incident description
    before any data exists, so it cannot be the only input: a report phrased
    mildly can turn out to be serious once the numbers are in.

    A live run made the case for this plainly. The model looked at a 34% error
    rate and a dependency failing 97% of requests, then wrote "the payment
    service is operating with a low error rate of 34.0%". Code comparing
    34.0 > 5.0 cannot make that mistake.
    """
    assessment = state["severity_assessment"]

    if assessment is None:
        raise ValueError("Severity assessment is missing.")

    if assessment.severity in ESCALATION_SEVERITIES:
        return "urgent"

    # An investigation that failed or could not complete gets human eyes
    # regardless of how mild the incident first appeared.
    investigation = state["investigation"]
    if investigation is None or not investigation.is_complete:
        return "urgent"

    # The description looked mild and the investigation finished - so let the
    # gathered numbers have the final say.
    if evaluate(investigation.evidence):
        return "urgent"

    return "normal"


def escalate_node(state: IncidentWorkflowState) -> dict[str, str]:
    """Hand off to a human, with whatever evidence was gathered."""
    return {"next_step": "escalate"}


def monitor_node(state: IncidentWorkflowState) -> dict[str, str]:
    """Low-severity incidents with a complete investigation: keep watching."""
    return {"next_step": "monitor"}


def build_incident_workflow():
    builder = StateGraph(IncidentWorkflowState)

    builder.add_node("assess_severity", assess_severity_node)
    builder.add_node("investigate", investigate_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("monitor", monitor_node)

    builder.add_edge(START, "assess_severity")
    builder.add_edge("assess_severity", "investigate")

    builder.add_conditional_edges(
        "investigate",
        route_by_severity,
        path_map={
            "urgent": "escalate",
            "normal": "monitor",
        },
    )

    builder.add_edge("escalate", END)
    builder.add_edge("monitor", END)

    return builder.compile()


incident_workflow = build_incident_workflow()


def handle_incident(
    incident_description: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> IncidentWorkflowState:
    """Run one incident through the full workflow."""
    return incident_workflow.invoke(
        {
            "incident_description": incident_description,
            "scenario": Scenario(scenario).value,
            "severity_assessment": None,
            "investigation": None,
            "investigation_error": None,
            "next_step": None,
        }
    )
