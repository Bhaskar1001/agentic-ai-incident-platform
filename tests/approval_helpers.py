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
import time
import uuid
from dataclasses import dataclass, field

from agentic_ai.domain.approvals import ApprovalRecord, pending_requests_for
from agentic_ai.domain.guardrails import ApprovalState, GuardedPlan, guard_plan
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import (
    ExecutionStatus,
    RemediationPlan,
    RemediationStep,
    StepExecutionResult,
)
from agentic_ai.llm.client import ChatResponse
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.persistence.runs import persist_workflow_run
from agentic_ai.tools.mock_remediation import execute_guarded_plan, execute_mock_remediation_step
from agentic_ai.tools.mock_tools import Scenario, get_mock_metrics, get_mock_pod_status
from agentic_ai.workflows import incident as workflow_module
from agentic_ai.workflows.incident import handle_incident

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


# --- Phase 18: submitting through the ledger, resuming, and breaking things on purpose ------

HEALTHY = Scenario.HEALTHY.value
STILL_BROKEN = Scenario.CONNECTION_POOL_EXHAUSTION.value


def submit(store, incident_id: str, *, post: str = HEALTHY, lookup=None):
    """One ``POST /incidents``: the workflow (reading approvals, claiming steps in the
    ledger), then persistence. Returns ``(state, RunPersistence)``."""
    state = handle_incident(
        "Payment service is degraded",
        scenario=Scenario.CONNECTION_POOL_EXHAUSTION,
        post_remediation_scenario=post,
        incident_id=incident_id,
        approval_repository=lookup if lookup is not None else store,
        execution_ledger=store,
    )
    return state, persist_workflow_run(store, state)


def decide(store, incident_id: str, index: int, outcome: ApprovalState, who: str = "alice"):
    approval = store.list_approvals(incident_id)[index]
    return store.decide_approval(incident_id, approval.approval_id, outcome, who)


def statuses(state) -> list[ExecutionStatus]:
    return [r.status for r in state["remediation_execution"].step_results]


def forbid_llm(monkeypatch) -> None:
    """Make any attempt to plan, assess or analyse fail the test.

    Used around resume, which must never call the planner (or any model). Replaces the
    stubs ``install_pipeline_stubs`` put in place, and the planner's client factory.
    """

    def _called(*args, **kwargs):
        raise AssertionError("an LLM-backed step was called during resume")

    monkeypatch.setattr(workflow_module, "assess_severity", _called)
    monkeypatch.setattr(workflow_module, "investigate", _called)
    monkeypatch.setattr(workflow_module, "analyze_root_cause", _called)
    monkeypatch.setattr("agentic_ai.llm.remediation.get_client", _called)


class CountingExecutor:
    """A step executor that counts what it was asked to run, and can misbehave on cue.

    ``fail_steps`` makes those action types return FAILED; ``raise_on`` makes them raise
    ``raise_with`` (use a BaseException subclass to simulate the process dying).
    """

    def __init__(self, *, fail_actions=(), raise_on=(), raise_with=RuntimeError, delay=0.0):
        self.calls: list[tuple[str, str]] = []
        self._delay = delay
        self._fail_actions = set(fail_actions)
        self._raise_on = set(raise_on)
        self._raise_with = raise_with

    def __call__(self, step: RemediationStep) -> StepExecutionResult:
        action = getattr(step.action_type, "value", str(step.action_type))
        self.calls.append((action, step.target_service))
        if self._delay:
            time.sleep(self._delay)  # long enough for concurrent callers to overlap
        if action in self._raise_on:
            raise self._raise_with(f"simulated failure of {action}")
        if action in self._fail_actions:
            return StepExecutionResult(
                step=step,
                status=ExecutionStatus.FAILED,
                output=f"[SIMULATED] {action} failed",
                error="simulated failure",
            )
        return execute_mock_remediation_step(step)


class WrappedStore:
    """The real store with selected operations replaced; everything else delegates."""

    def __init__(self, store, **overrides):
        self._store = store
        self._overrides = overrides

    def __getattr__(self, name):
        return self._overrides.get(name) or getattr(self._store, name)


def boom(*args, **kwargs):
    raise RuntimeError("database down")


def seed_realistic_approvals(conn, n: int = 2000) -> None:
    """Fill the approvals table the way it looks in use: many decided, few pending.

    A query-plan assertion on a near-empty table is meaningless: PostgreSQL rightly prefers a
    bitmap scan and a sort over a handful of rows, and which it picks depends on table
    statistics. With a realistic distribution (here 2% pending) and fresh statistics the
    choice is stable. Every id carries the ``TEST-APR-`` prefix, so the fixtures' teardown
    removes these rows.
    """
    conn.execute(
        "INSERT INTO incidents (incident_id, correlation_id, scenario, incident_description, "
        "severity, status) SELECT 'TEST-APR-SEED-' || g, 'C', 's', 'd', 'high', "
        "'awaiting_approval' FROM generate_series(1, %s) g",
        (n,),
    )
    conn.execute(
        "INSERT INTO incident_approvals (approval_id, incident_id, plan_fingerprint, step_index, "
        "step_fingerprint, action_type, target_service, status, decided_at, approver) "
        "SELECT 'APR-SEED-' || g, 'TEST-APR-SEED-' || g, repeat('a', 64), 0, repeat('b', 64), "
        # (%% is a literal modulo: this statement also carries a %s parameter.)
        "'failover', 'payment', CASE WHEN g %% 50 = 0 THEN 'pending' ELSE 'approved' END, "
        "CASE WHEN g %% 50 = 0 THEN NULL ELSE now() END, "
        "CASE WHEN g %% 50 = 0 THEN NULL ELSE 'x' END FROM generate_series(1, %s) g",
        (n,),
    )
    conn.execute("ANALYZE incident_approvals")


def explain(conn, sql: str) -> list[str]:
    """The query plan as lines, with sequential scans discouraged so an index is shown if usable."""
    conn.execute("SET enable_seqscan = off")
    return [row[0] for row in conn.execute("EXPLAIN " + sql).fetchall()]
