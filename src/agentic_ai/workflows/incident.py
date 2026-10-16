"""Incident response workflow.

Orchestrates the pieces into one flow::

    assess severity -> investigate -> route on evidence
                                        |- urgent -> root cause analysis
                                        |            -> remediation plan -> remember -> escalate
                                        `- normal -> monitor

Every incident is investigated before routing, so an escalation always carries
evidence with it. Root cause analysis runs only on the escalation path: an
incident routed to monitor was not an incident, and there is no root cause to
explain. Routing itself never consults RCA's output - escalation is decided
from raw evidence before RCA exists, the same principle that keeps the
escalation policy reading tool output rather than an LLM's narrative about it.
"""

from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agentic_ai.agents.investigation import InvestigationError, investigate
from agentic_ai.domain.escalation import evaluate
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import InvestigationResult
from agentic_ai.domain.knowledge import KnowledgeEntry
from agentic_ai.domain.rca import RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationPlan
from agentic_ai.llm.rca import RootCauseAnalysisError, analyze_root_cause
from agentic_ai.llm.remediation import RemediationPlanningError, plan_remediation
from agentic_ai.llm.severity import SeverityAssessment, assess_severity
from agentic_ai.tools.incident_memory import remember
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
    root_cause_analysis: RootCauseAnalysis | None
    root_cause_analysis_error: str | None
    remediation_plan: RemediationPlan | None
    remediation_plan_error: str | None
    memory_recorded: bool
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


def analyze_root_cause_node(state: IncidentWorkflowState) -> dict[str, object]:
    """Synthesize a root cause analysis for an escalating incident.

    Only reached on the escalation path - a "normal" incident was not an
    incident, and there is nothing here to explain. An investigation that
    never completed has no findings worth synthesizing either, so this is
    skipped rather than asked to produce an analysis from nothing.

    A failure here must not block escalation: a human is already being
    paged, and an unexplained incident with no RCA is still better handed to
    a human than silently dropped.
    """
    investigation = state["investigation"]
    if investigation is None or not investigation.findings:
        return {"root_cause_analysis": None, "root_cause_analysis_error": None}

    try:
        analysis = analyze_root_cause(investigation)
    except RootCauseAnalysisError as exc:
        return {"root_cause_analysis": None, "root_cause_analysis_error": str(exc)}

    return {"root_cause_analysis": analysis, "root_cause_analysis_error": None}


def plan_remediation_node(state: IncidentWorkflowState) -> dict[str, object]:
    """Propose (never execute) remediation steps from the root cause.

    Skipped without an RCA: a plan with no stated cause would be a guess. A
    failure here does not block escalation, for the same reason an RCA failure
    does not.
    """
    analysis = state["root_cause_analysis"]
    if analysis is None:
        return {"remediation_plan": None, "remediation_plan_error": None}

    try:
        plan = plan_remediation(state["incident_description"], analysis)
    except RemediationPlanningError as exc:
        return {"remediation_plan": None, "remediation_plan_error": str(exc)}

    return {"remediation_plan": plan, "remediation_plan_error": None}


def remember_incident_node(state: IncidentWorkflowState) -> dict[str, bool]:
    """Record the analysed incident in long-term memory, flagged as learned.

    Needs an RCA - without a root cause there is nothing worth remembering.
    A storage failure must not block escalation, and the entry is stored as
    unverified so later retrieval cannot mistake it for curated precedent.
    """
    analysis = state["root_cause_analysis"]
    if analysis is None:
        return {"memory_recorded": False}

    plan = state["remediation_plan"]
    if plan and plan.steps:
        resolution = "; ".join(
            f"{step.action_type.value} on {step.target_service}"
            for step in plan.steps
        )
    else:
        resolution = "No remediation proposed."

    description = state["incident_description"]
    entry = KnowledgeEntry(
        id=f"MEM-{uuid4().hex[:8]}",
        title=description[:80],
        symptom_description=description,
        root_cause=analysis.root_cause,
        resolution=f"Proposed (unexecuted): {resolution}",
        tags=["learned"],
        source="learned",
    )
    try:
        return {"memory_recorded": remember(entry)}
    except OSError:
        return {"memory_recorded": False}


def escalate_node(state: IncidentWorkflowState) -> dict[str, str]:
    """Hand off to a human, with whatever evidence and analysis were gathered."""
    return {"next_step": "escalate"}


def monitor_node(state: IncidentWorkflowState) -> dict[str, str]:
    """Low-severity incidents with a complete investigation: keep watching."""
    return {"next_step": "monitor"}


def build_incident_workflow():
    builder = StateGraph(IncidentWorkflowState)

    builder.add_node("assess_severity", assess_severity_node)
    builder.add_node("investigate", investigate_node)
    builder.add_node("analyze_root_cause", analyze_root_cause_node)
    builder.add_node("plan_remediation", plan_remediation_node)
    builder.add_node("remember_incident", remember_incident_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("monitor", monitor_node)

    builder.add_edge(START, "assess_severity")
    builder.add_edge("assess_severity", "investigate")

    builder.add_conditional_edges(
        "investigate",
        route_by_severity,
        path_map={
            "urgent": "analyze_root_cause",
            "normal": "monitor",
        },
    )

    builder.add_edge("analyze_root_cause", "plan_remediation")
    builder.add_edge("plan_remediation", "remember_incident")
    builder.add_edge("remember_incident", "escalate")
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
            "root_cause_analysis": None,
            "root_cause_analysis_error": None,
            "remediation_plan": None,
            "remediation_plan_error": None,
            "memory_recorded": False,
            "next_step": None,
        }
    )
