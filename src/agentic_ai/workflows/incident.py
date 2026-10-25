"""Incident response workflow.

Orchestrates the pieces into one flow::

    assess severity -> investigate -> route on evidence
                                        |- urgent -> root cause analysis
                                        |            -> remediation plan -> guardrails -> execute remediation
                                        |            -> verify remediation -> remember -> route after verification:
                                        |                                                    |- recovered -> resolve
                                        |                                                    `- unresolved/inconclusive/skipped -> escalate
                                        `- normal -> monitor

Every incident is investigated before routing, so an escalation always carries
evidence with it. Root cause analysis runs only on the escalation path: an
incident routed to monitor was not an incident, and there is no root cause to
explain. Routing itself never consults RCA's output - escalation is decided
from raw evidence before RCA exists, the same principle that keeps the
escalation policy reading tool output rather than an LLM's narrative about it.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agentic_ai.agents.investigation import InvestigationError, investigate
from agentic_ai.domain.approvals import apply_persisted_decisions
from agentic_ai.domain.escalation import evaluate
from agentic_ai.domain.guardrails import Disposition, GuardedPlan, guard_plan
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import InvestigationResult
from agentic_ai.domain.knowledge import KnowledgeEntry
from agentic_ai.domain.rca import RootCauseAnalysis
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationExecution,
    RemediationPlan,
)
from agentic_ai.domain.verification import VerificationResult, VerificationStatus
from agentic_ai.llm.rca import RootCauseAnalysisError, analyze_root_cause
from agentic_ai.llm.remediation import RemediationPlanningError, plan_remediation
from agentic_ai.llm.severity import SeverityAssessment, assess_severity
from agentic_ai.observability.events import (
    DiagnosticReason,
    IncidentTrace,
)
from agentic_ai.observability.tracer import OperationalTracer
from agentic_ai.persistence.repository import ApprovalRepository
from agentic_ai.tools.incident_memory import remember
from agentic_ai.tools.mock_remediation import execute_guarded_plan
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.tools.mock_verification import verify_service_health

ESCALATION_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})

logger = logging.getLogger("agentic_ai.workflow")


class IncidentWorkflowState(TypedDict):
    """State threaded through the incident workflow.

    Holds the *incident description* rather than an ``Incident`` entity:
    orchestration state and domain entities have different lifecycles, and the
    entity's lifecycle invariants should not be run through graph state
    merging. A persisted version would carry an ``incident_id`` instead.
    """

    incident_description: str
    scenario: str
    post_remediation_scenario: str | None
    severity_assessment: SeverityAssessment | None
    investigation: InvestigationResult | None
    investigation_error: str | None
    root_cause_analysis: RootCauseAnalysis | None
    root_cause_analysis_error: str | None
    remediation_plan: RemediationPlan | None
    remediation_plan_error: str | None
    guarded_plan: GuardedPlan | None
    remediation_execution: RemediationExecution | None
    verification: VerificationResult | None
    approval_repository: ApprovalRepository | None
    approval_lookup_error: str | None
    memory_recorded: bool
    next_step: str | None
    incident_id: str | None
    correlation_id: str | None
    tracer: OperationalTracer | None
    trace: IncidentTrace | None


@contextmanager
def _stage_span(
    tracer: OperationalTracer | None,
    stage_name: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Safe stage span context manager. Bypasses cleanly if tracer is omitted."""
    if tracer is None:
        yield None
        return
    try:
        span_cm = tracer.stage_span(stage_name, metadata=metadata)
    except Exception:
        yield None
        return

    with span_cm as handle:
        yield handle


def _compute_diagnostic(
    outcome: Literal["resolve", "escalate", "monitor"],
    state: IncidentWorkflowState,
) -> DiagnosticReason:
    """Compute structured diagnostic rationale explaining the final workflow outcome."""
    if outcome == "resolve":
        return DiagnosticReason(
            decision="resolve",
            primary_reason="Remediation verified: service health recovered.",
            trigger_type="verification_recovered",
            details={"verification_status": "recovered"},
        )

    if outcome == "monitor":
        assessment = state.get("severity_assessment")
        sev = assessment.severity.value if assessment else "low"
        return DiagnosticReason(
            decision="monitor",
            primary_reason="Incident assessed as low severity and evidence shows healthy baseline.",
            trigger_type="severity_baseline",
            details={"severity": sev},
        )

    # outcome == "escalate"
    verification = state.get("verification")
    execution = state.get("remediation_execution")
    guarded = state.get("guarded_plan")
    investigation = state.get("investigation")
    assessment = state.get("severity_assessment")

    # 1. Did verification run post-execution and fail?
    if verification is not None:
        if verification.status == VerificationStatus.UNRESOLVED:
            reasons_str = (
                "; ".join(verification.reasons) if verification.reasons else "anomalies persist"
            )
            return DiagnosticReason(
                decision="escalate",
                primary_reason=f"Post-remediation verification unresolved: {reasons_str}.",
                trigger_type="verification_unresolved",
                details={
                    "verification_status": verification.status.value,
                    "reasons": verification.reasons,
                },
            )
        if verification.status == VerificationStatus.INCONCLUSIVE:
            reasons_str = (
                "; ".join(verification.reasons) if verification.reasons else "insufficient evidence"
            )
            return DiagnosticReason(
                decision="escalate",
                primary_reason=f"Post-remediation verification inconclusive: {reasons_str}.",
                trigger_type="verification_unresolved",
                details={
                    "verification_status": verification.status.value,
                    "reasons": verification.reasons,
                },
            )

    # 2. Was execution blocked by guardrails or execution failure?
    if execution is not None and execution.step_results:
        has_blocked = any(r.status == ExecutionStatus.BLOCKED for r in execution.step_results)
        has_failed = any(r.status == ExecutionStatus.FAILED for r in execution.step_results)
        if has_blocked:
            return DiagnosticReason(
                decision="escalate",
                primary_reason="Remediation actions blocked by safety guardrails.",
                trigger_type="guardrail_rejection",
                details={"execution_status": "blocked"},
            )
        if has_failed:
            return DiagnosticReason(
                decision="escalate",
                primary_reason="Remediation action failed during execution.",
                trigger_type="execution_failure",
                details={"execution_status": "failed"},
            )

    # 3. Guardrail plan rejected or blocked steps
    if guarded is not None and guarded.steps:
        blocked_or_pending = [
            s
            for s in guarded.steps
            if s.disposition in (Disposition.BLOCKED, Disposition.REQUIRES_APPROVAL)
        ]
        if blocked_or_pending:
            reasons = [r for s in blocked_or_pending for r in s.reasons]
            return DiagnosticReason(
                decision="escalate",
                primary_reason="Remediation blocked or requires approval by safety guardrails.",
                trigger_type="guardrail_rejection",
                details={
                    "dispositions": [s.disposition.value for s in blocked_or_pending],
                    "reasons": reasons,
                },
            )

    # 4. Investigation incomplete or failed
    if state.get("investigation_error") or investigation is None or not investigation.is_complete:
        err = state.get("investigation_error") or (
            investigation.termination_reason.value if investigation else "missing investigation"
        )
        return DiagnosticReason(
            decision="escalate",
            primary_reason=f"Investigation could not complete: {err}.",
            trigger_type="investigation_incomplete",
            details={"investigation_error": str(err)},
        )

    # 5. Evidence metrics triggered escalation
    if investigation and evaluate(investigation.evidence):
        if assessment and assessment.severity not in ESCALATION_SEVERITIES:
            return DiagnosticReason(
                decision="escalate",
                primary_reason="Telemetry metrics exceeded safe operational thresholds.",
                trigger_type="evidence_metrics",
                details={"severity_assessed": assessment.severity.value},
            )

    # 6. Severity baseline
    sev_str = assessment.severity.value if assessment else "unknown"
    return DiagnosticReason(
        decision="escalate",
        primary_reason=f"Incident classified as {sev_str.upper()} severity requiring human escalation.",
        trigger_type="severity_baseline",
        details={"severity": sev_str},
    )


def assess_severity_node(
    state: IncidentWorkflowState,
) -> dict[str, SeverityAssessment]:
    """Classify how serious the incident is, from its description alone."""
    tracer = state.get("tracer")
    with _stage_span(tracer, "assess_severity"):
        return {"severity_assessment": assess_severity(state["incident_description"])}


def investigate_node(state: IncidentWorkflowState) -> dict[str, object]:
    """Run the investigation agent to gather evidence and form findings.

    A failed investigation must not sink the workflow: the incident still
    needs routing, and an unexplained incident is precisely one a human should
    see. The error is recorded and the flow continues.
    """
    tracer = state.get("tracer")
    with _stage_span(tracer, "investigate"):
        try:
            result = investigate(
                state["incident_description"],
                scenario=state["scenario"],
            )
        except InvestigationError as exc:
            return {"investigation": None, "investigation_error": str(exc)}

        if tracer is not None and result is not None and result.evidence:
            try:
                for item in result.evidence:
                    target_svc = (
                        item.tool_arguments.get("service")
                        if isinstance(item.tool_arguments, dict)
                        else None
                    )
                    args = item.tool_arguments if isinstance(item.tool_arguments, dict) else {}
                    with tracer.tool_span(
                        tool_name=item.source_tool,
                        target_service=target_svc,
                        arguments=args,
                    ):
                        pass
            except Exception:
                pass

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
    tracer = state.get("tracer")
    with _stage_span(tracer, "analyze_root_cause"):
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
    tracer = state.get("tracer")
    with _stage_span(tracer, "plan_remediation"):
        analysis = state["root_cause_analysis"]
        if analysis is None:
            return {"remediation_plan": None, "remediation_plan_error": None}

        try:
            plan = plan_remediation(state["incident_description"], analysis)
        except RemediationPlanningError as exc:
            return {"remediation_plan": None, "remediation_plan_error": str(exc)}

        return {"remediation_plan": plan, "remediation_plan_error": None}


def apply_guardrails_node(state: IncidentWorkflowState) -> dict[str, GuardedPlan | None]:
    """Judge every proposed step against the deterministic guardrail policy.

    Pure code, so no failure handling: it reads the plan, the RCA and the
    services the investigation actually examined, and nothing here executes.
    """
    tracer = state.get("tracer")
    with _stage_span(tracer, "apply_guardrails"):
        plan = state["remediation_plan"]
        analysis = state["root_cause_analysis"]
        if plan is None or analysis is None:
            return {"guarded_plan": None}

        investigation = state["investigation"]
        investigated = (
            {
                item.tool_arguments["service"]
                for item in investigation.evidence
                if isinstance(item.tool_arguments, dict) and "service" in item.tool_arguments
            }
            if investigation
            else set()
        )
        return {"guarded_plan": guard_plan(plan, analysis, investigated)}


def _apply_stored_approvals(
    state: IncidentWorkflowState, guarded: GuardedPlan
) -> tuple[GuardedPlan, str | None]:
    """Apply persisted approve/reject decisions to ``guarded``; fail closed on error.

    With no approval repository (CLI, evaluation, tests) or no incident id the plan
    is returned untouched, which is the pre-approval behaviour: steps awaiting a
    human stay unexecuted. If the lookup itself fails, the same happens and the
    failure is reported, never swallowed into "approved": an unreachable approval
    store must not be able to widen what executes.
    """
    repository = state.get("approval_repository")
    incident_id = state.get("incident_id")
    if repository is None or incident_id is None:
        return guarded, None

    try:
        approvals = repository.list_approvals(incident_id)
    except Exception as exc:
        logger.warning(
            "Approval lookup failed (%s); steps awaiting approval stay unexecuted.",
            type(exc).__name__,
        )
        return guarded, (
            f"approval lookup failed ({type(exc).__name__}); "
            "steps awaiting approval were left unexecuted"
        )

    return apply_persisted_decisions(guarded, approvals), None


def execute_remediation_node(
    state: IncidentWorkflowState,
) -> dict[str, RemediationExecution | None]:
    """Execute permitted steps in the guarded plan in simulation mode.

    Passes the guarded plan directly to execute_guarded_plan without
    reconstructing or weakening the guardrail decision.
    """
    tracer = state.get("tracer")
    with _stage_span(tracer, "execute_remediation"):
        guarded = state["guarded_plan"]
        if guarded is None:
            return {"remediation_execution": None}

        # The enforcement point. Stored human decisions are folded back onto the
        # freshly guarded plan *here*, immediately before the executor sees it, so an
        # approval only ever counts if it matches an exact step of this exact plan.
        guarded, lookup_error = _apply_stored_approvals(state, guarded)

        execution = execute_guarded_plan(guarded)

        if tracer is not None and execution and execution.step_results:
            try:
                for step_res in execution.step_results:
                    action_type = (
                        step_res.step.action_type.value
                        if hasattr(step_res.step.action_type, "value")
                        else str(step_res.step.action_type)
                    )
                    svc = step_res.step.target_service
                    step_params: dict[str, Any] = {
                        "service": svc,
                        "rationale": step_res.step.rationale,
                    }
                    extra_params = getattr(step_res.step, "parameters", None)
                    if isinstance(extra_params, dict):
                        step_params.update(extra_params)

                    if step_res.status == ExecutionStatus.EXECUTED:
                        with tracer.tool_span(
                            tool_name=action_type,
                            target_service=svc,
                            arguments=step_params,
                        ):
                            pass
                    elif step_res.status == ExecutionStatus.BLOCKED:
                        tracer.record_blocked_tool(
                            tool_name=action_type,
                            target_service=svc,
                            arguments=step_params,
                            reason=step_res.output or "Blocked by safety guardrail",
                        )
                    elif step_res.status == ExecutionStatus.FAILED:
                        with tracer.tool_span(
                            tool_name=action_type,
                            target_service=svc,
                            arguments=step_params,
                        ) as th:
                            th.status = "failure"
                            th.error_message = step_res.output
            except Exception:
                pass

        return {
            "remediation_execution": execution,
            # Downstream (persistence, tracing) must see the plan as it was executed.
            "guarded_plan": guarded,
            "approval_lookup_error": lookup_error,
        }


def verify_remediation_node(
    state: IncidentWorkflowState,
) -> dict[str, VerificationResult]:
    """Verify service health post-remediation using deterministic criteria.

    Invariants:
    - If no remediation action was executed, status is SKIPPED (never RECOVERED).
    - If remediation was executed, queries observability tools for the target service.
    - Evaluates recovery deterministically using raw tool evidence.
    - Bounded to 1 attempt per workflow invocation (never triggers automatic repeat remediation).
    """
    tracer = state.get("tracer")
    with _stage_span(tracer, "verify_remediation"):
        execution = state.get("remediation_execution")
        if execution is None or not execution.step_results:
            return {
                "verification": VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    attempts=0,
                    reasons=["no remediation action was executed"],
                    evidence=[],
                )
            }

        executed_steps = [
            res.step for res in execution.step_results if res.status == ExecutionStatus.EXECUTED
        ]

        if not executed_steps:
            return {
                "verification": VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    attempts=0,
                    reasons=["no remediation action was executed (all steps skipped or blocked)"],
                    evidence=[],
                )
            }

        service = executed_steps[0].target_service
        scenario = state.get("post_remediation_scenario") or state["scenario"]

        if tracer is not None:
            try:
                tool_cm = tracer.tool_span(
                    tool_name="verify_service_health",
                    target_service=service,
                    arguments={"service": service, "scenario": scenario, "attempts": 1},
                )
            except Exception:
                tool_cm = None

            if tool_cm is not None:
                with tool_cm:
                    result = verify_service_health(service=service, scenario=scenario, attempts=1)
            else:
                result = verify_service_health(service=service, scenario=scenario, attempts=1)
        else:
            result = verify_service_health(service=service, scenario=scenario, attempts=1)

        return {"verification": result}


def _remember_incident_impl(state: IncidentWorkflowState) -> dict[str, bool]:
    """Record the analysed incident in long-term memory, flagged as learned.

    Needs an RCA - without a root cause there is nothing worth remembering.
    A storage failure must not block escalation, and the entry is stored as
    unverified so later retrieval cannot mistake it for curated precedent.
    """
    analysis = state["root_cause_analysis"]
    if analysis is None:
        return {"memory_recorded": False}

    execution = state.get("remediation_execution")
    plan = state.get("remediation_plan")
    verification = state.get("verification")

    if execution and execution.step_results:
        parts: list[str] = []
        for res in execution.step_results:
            action_desc = f"{res.step.action_type.value} on {res.step.target_service}"
            if res.status == ExecutionStatus.EXECUTED:
                parts.append(f"Executed: {action_desc}")
            elif res.status == ExecutionStatus.BLOCKED:
                parts.append(f"Blocked: {action_desc}")
            elif res.status == ExecutionStatus.SKIPPED:
                parts.append(f"Skipped: {action_desc}")
            elif res.status == ExecutionStatus.FAILED:
                parts.append(f"Failed: {action_desc}")
            else:
                parts.append(f"{res.status.value}: {action_desc}")
        base_resolution = "; ".join(parts)
    elif plan and plan.steps:
        proposed = "; ".join(
            f"{step.action_type.value} on {step.target_service}" for step in plan.steps
        )
        base_resolution = f"Proposed (unexecuted): {proposed}"
    else:
        base_resolution = "No remediation proposed."

    if verification is not None:
        if verification.status == VerificationStatus.RECOVERED:
            recovery_desc = "Recovery: verified recovered"
        elif verification.status == VerificationStatus.UNRESOLVED:
            reasons_str = (
                "; ".join(verification.reasons) if verification.reasons else "anomalies persist"
            )
            recovery_desc = f"Recovery: unresolved ({reasons_str})"
        elif verification.status == VerificationStatus.INCONCLUSIVE:
            reasons_str = (
                "; ".join(verification.reasons) if verification.reasons else "inconclusive evidence"
            )
            recovery_desc = f"Recovery: inconclusive ({reasons_str})"
        elif verification.status == VerificationStatus.SKIPPED:
            recovery_desc = "Recovery: skipped (no remediation executed)"
        else:
            recovery_desc = f"Recovery: {verification.status.value}"
        resolution = f"{base_resolution} | {recovery_desc}"
    else:
        resolution = base_resolution

    description = state["incident_description"]
    entry = KnowledgeEntry(
        id=f"MEM-{uuid4().hex[:8]}",
        title=description[:80],
        symptom_description=description,
        root_cause=analysis.root_cause,
        resolution=resolution,
        tags=["learned"],
        source="learned",
    )
    try:
        return {"memory_recorded": remember(entry)}
    except OSError:
        return {"memory_recorded": False}


def remember_incident_node(state: IncidentWorkflowState) -> dict[str, bool]:
    """Record the analysed incident in long-term memory, flagged as learned."""
    tracer = state.get("tracer")
    with _stage_span(tracer, "remember_incident"):
        return _remember_incident_impl(state)


def resolve_node(state: IncidentWorkflowState) -> dict[str, Any]:
    """Mark the incident resolved after verified recovery."""
    tracer = state.get("tracer")
    diag = _compute_diagnostic("resolve", state)
    trace = None
    if tracer is not None:
        try:
            with _stage_span(tracer, "resolve"):
                pass
            tracer.set_final_outcome("resolve", diagnostic=diag)
            trace = tracer.get_trace()
        except Exception:
            pass
    return {"next_step": "resolve", "trace": trace}


def escalate_node(state: IncidentWorkflowState) -> dict[str, Any]:
    """Hand off to a human, with whatever evidence and analysis were gathered."""
    tracer = state.get("tracer")
    diag = _compute_diagnostic("escalate", state)
    trace = None
    if tracer is not None:
        try:
            with _stage_span(tracer, "escalate"):
                pass
            tracer.set_final_outcome("escalate", diagnostic=diag)
            trace = tracer.get_trace()
        except Exception:
            pass
    return {"next_step": "escalate", "trace": trace}


def monitor_node(state: IncidentWorkflowState) -> dict[str, Any]:
    """Low-severity incidents with a complete investigation: keep watching."""
    tracer = state.get("tracer")
    diag = _compute_diagnostic("monitor", state)
    trace = None
    if tracer is not None:
        try:
            with _stage_span(tracer, "monitor"):
                pass
            tracer.set_final_outcome("monitor", diagnostic=diag)
            trace = tracer.get_trace()
        except Exception:
            pass
    return {"next_step": "monitor", "trace": trace}


def route_after_verification(
    state: IncidentWorkflowState,
) -> Literal["resolve", "escalate"]:
    """Route based on deterministic verification outcome.

    - RECOVERED -> resolve
    - UNRESOLVED, INCONCLUSIVE, or SKIPPED -> escalate
    """
    verification = state.get("verification")
    if verification is not None and verification.status == VerificationStatus.RECOVERED:
        return "resolve"
    return "escalate"


def build_incident_workflow():
    builder = StateGraph(IncidentWorkflowState)

    builder.add_node("assess_severity", assess_severity_node)
    builder.add_node("investigate", investigate_node)
    builder.add_node("analyze_root_cause", analyze_root_cause_node)
    builder.add_node("plan_remediation", plan_remediation_node)
    builder.add_node("apply_guardrails", apply_guardrails_node)
    builder.add_node("execute_remediation", execute_remediation_node)
    builder.add_node("verify_remediation", verify_remediation_node)
    builder.add_node("remember_incident", remember_incident_node)
    builder.add_node("resolve", resolve_node)
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
    builder.add_edge("plan_remediation", "apply_guardrails")
    builder.add_edge("apply_guardrails", "execute_remediation")
    builder.add_edge("execute_remediation", "verify_remediation")
    builder.add_edge("verify_remediation", "remember_incident")

    builder.add_conditional_edges(
        "remember_incident",
        route_after_verification,
        path_map={
            "resolve": "resolve",
            "escalate": "escalate",
        },
    )

    builder.add_edge("resolve", END)
    builder.add_edge("escalate", END)
    builder.add_edge("monitor", END)

    return builder.compile()


incident_workflow = build_incident_workflow()


def handle_incident(
    incident_description: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
    post_remediation_scenario: Scenario | str | None = None,
    incident_id: str | None = None,
    correlation_id: str | None = None,
    tracer: OperationalTracer | None = None,
    approval_repository: ApprovalRepository | None = None,
) -> IncidentWorkflowState:
    """Run one incident through the full workflow with operational tracing."""
    effective_incident_id = incident_id or (
        tracer.incident_id if tracer else f"INC-{uuid4().hex[:8]}"
    )
    effective_correlation_id = correlation_id or (
        tracer.correlation_id if tracer else f"CORR-{uuid4().hex[:8]}"
    )
    effective_tracer = tracer or OperationalTracer(
        incident_id=effective_incident_id,
        correlation_id=effective_correlation_id,
    )

    result = incident_workflow.invoke(
        {
            "incident_description": incident_description,
            "scenario": Scenario(scenario).value,
            "post_remediation_scenario": (
                Scenario(post_remediation_scenario).value
                if post_remediation_scenario is not None
                else None
            ),
            "severity_assessment": None,
            "investigation": None,
            "investigation_error": None,
            "root_cause_analysis": None,
            "root_cause_analysis_error": None,
            "remediation_plan": None,
            "remediation_plan_error": None,
            "guarded_plan": None,
            "remediation_execution": None,
            "verification": None,
            "approval_repository": approval_repository,
            "approval_lookup_error": None,
            "memory_recorded": False,
            "next_step": None,
            "incident_id": effective_incident_id,
            "correlation_id": effective_correlation_id,
            "tracer": effective_tracer,
            "trace": None,
        }
    )

    if result.get("trace") is None and effective_tracer is not None:
        try:
            if effective_tracer._final_outcome is None:
                outcome = result.get("next_step")
                if outcome in ("resolve", "escalate", "monitor"):
                    diag = _compute_diagnostic(outcome, result)
                    effective_tracer.set_final_outcome(outcome, diagnostic=diag)
            result["trace"] = effective_tracer.get_trace()
        except Exception:
            pass

    if result.get("incident_id") is None:
        result["incident_id"] = effective_incident_id
    if result.get("correlation_id") is None:
        result["correlation_id"] = effective_correlation_id
    if result.get("tracer") is None:
        result["tracer"] = effective_tracer
    return result
