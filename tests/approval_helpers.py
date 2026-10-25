"""Builders for approval tests.

Records are produced by the real planner types, the real guardrail policy and the
real mock executor, then mapped with the real ``from_workflow_state``, so the
fingerprints in a stored approval are exactly the ones the workflow will compute
and the incident's stored remediations really do describe the same plan.

``install_pipeline_stubs`` additionally gives a whole pipeline with a real planner,
guardrails, executor and verification. Only the LLM-backed steps are controlled:
severity, investigation and RCA are stubbed, and the *planner's LLM client* is
replaced by a fake that returns real JSON, so the actual planner prompt, schema
validation and retry logic still run.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from agentic_ai.domain.approvals import ApprovalRecord, pending_requests_for
from agentic_ai.domain.guardrails import GuardedPlan, guard_plan
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import RemediationPlan, RemediationStep
from agentic_ai.llm.client import ChatResponse
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.tools.mock_remediation import execute_guarded_plan
from agentic_ai.tools.mock_tools import Scenario, get_mock_metrics, get_mock_pod_status
from agentic_ai.workflows import incident as workflow_module

ID_PREFIX = "TEST-APR-"

# Risk tiers, from domain/remediation.py: scale_up is LOW; restart_service and
# increase_connection_pool are MEDIUM; rollback_deployment and failover are HIGH.
LOW = ("scale_up", "payment")
HIGH_ROLLBACK = ("rollback_deployment", "payment")
HIGH_FAILOVER = ("failover", "payment")


def new_incident_id() -> str:
    return f"{ID_PREFIX}{uuid.uuid4().hex[:10]}"


def analysis(confidence: float = 0.9) -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Database connection pool exhaustion.",
        root_cause_confidence=confidence,
        knowledge_base_consistency=KnowledgeBaseConsistency.SUPPORTED,
    )


def plan_of(*actions: tuple[str, str]) -> RemediationPlan:
    return RemediationPlan(
        steps=[
            RemediationStep(action_type=a, target_service=t, rationale=f"because of {a}")
            for a, t in actions
        ]
    )


def guarded(*actions: tuple[str, str], confidence: float = 0.9) -> GuardedPlan:
    return guard_plan(plan_of(*actions), analysis(confidence), {"payment"})


def state_for(incident_id: str, plan: GuardedPlan, outcome: str = "escalate") -> dict:
    return {
        "incident_id": incident_id,
        "incident_description": "Payment service is degraded",
        "guarded_plan": plan,
        "remediation_execution": execute_guarded_plan(plan),
        "next_step": outcome,
    }


def saved_incident(
    store, *actions: tuple[str, str], incident_id: str | None = None, confidence: float = 0.9
) -> tuple[IncidentRecord, GuardedPlan, list[ApprovalRecord]]:
    """Save an escalated run's record and register its pending approvals.

    Returns the stored record, the guarded plan, and the stored approvals.
    """
    incident_id = incident_id or new_incident_id()
    plan = guarded(*actions, confidence=confidence)
    record = IncidentRecord.from_workflow_state(state_for(incident_id, plan))
    store.save_incident(record)
    approvals = store.request_approvals(pending_requests_for(incident_id, plan))
    return record, plan, approvals


# --- a whole pipeline -----------------------------------------------------------------


@dataclass
class PipelineConfig:
    """What the stubbed model 'decides' on the next run. Mutate between runs."""

    actions: list[tuple[str, str]] = field(default_factory=lambda: [HIGH_ROLLBACK])
    confidence: float = 0.9


class _PlannerClient:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def complete(self, messages, *, tools=None, json_schema=None) -> ChatResponse:
        steps = [
            {"action_type": a, "target_service": t, "rationale": f"because of {a}"}
            for a, t in self._config.actions
        ]
        return ChatResponse(content=json.dumps({"steps": steps}))


def _investigation() -> InvestigationResult:
    evidence = [
        Evidence(
            source_tool=name,
            tool_arguments={"service": "payment"},
            raw_output=tool("payment", Scenario.CONNECTION_POOL_EXHAUSTION),
            observation=f"{name} returned data",
        )
        for name, tool in (
            ("get_mock_metrics", get_mock_metrics),
            ("get_mock_pod_status", get_mock_pod_status),
        )
    ]
    return InvestigationResult(
        summary="Connection pool exhausted.",
        model_assessed_confidence=0.8,
        recommended_next_action="Remediate.",
        termination_reason=TerminationReason.COMPLETED,
        evidence=evidence,
        findings=[
            Finding(statement="Connection pool utilisation is 100%.", model_assessed_confidence=0.9)
        ],
    )


def install_pipeline_stubs(monkeypatch) -> PipelineConfig:
    config = PipelineConfig()
    monkeypatch.setattr(
        workflow_module,
        "assess_severity",
        lambda description: SeverityAssessment(severity=Severity.CRITICAL, reasoning="test"),
    )
    monkeypatch.setattr(
        workflow_module, "investigate", lambda description, scenario=None: _investigation()
    )
    monkeypatch.setattr(
        workflow_module, "analyze_root_cause", lambda result: analysis(config.confidence)
    )
    monkeypatch.setattr("agentic_ai.llm.remediation.get_client", lambda: _PlannerClient(config))
    return config
