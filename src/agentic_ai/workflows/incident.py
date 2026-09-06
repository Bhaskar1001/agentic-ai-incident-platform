from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agentic_ai.domain.incident import Severity
from agentic_ai.llm.severity import SeverityAssessment, assess_severity


class IncidentWorkflowState(TypedDict):
    incident_description: str
    severity_assessment: SeverityAssessment | None
    next_step: str | None


def assess_severity_node(
    state: IncidentWorkflowState,
) -> dict[str, SeverityAssessment]:
    assessment = assess_severity(state["incident_description"])

    return {
        "severity_assessment": assessment,
    }


def route_by_severity(
    state: IncidentWorkflowState,
) -> Literal["urgent", "normal"]:
    assessment = state["severity_assessment"]

    if assessment is None:
        raise ValueError("Severity assessment is missing.")

    if assessment.severity in {Severity.HIGH, Severity.CRITICAL}:
        return "urgent"

    return "normal"


def escalate_node(
    state: IncidentWorkflowState,
) -> dict[str, str]:
    return {
        "next_step": "escalate",
    }


def investigate_node(
    state: IncidentWorkflowState,
) -> dict[str, str]:
    return {
        "next_step": "investigate",
    }


def build_incident_workflow():
    builder = StateGraph(IncidentWorkflowState)

    builder.add_node("assess_severity", assess_severity_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("investigate", investigate_node)

    builder.add_edge(START, "assess_severity")

    builder.add_conditional_edges(
        "assess_severity",
        route_by_severity,
        path_map={
            "urgent": "escalate",
            "normal": "investigate",
        },
    )

    builder.add_edge("escalate", END)
    builder.add_edge("investigate", END)

    return builder.compile()


incident_workflow = build_incident_workflow()