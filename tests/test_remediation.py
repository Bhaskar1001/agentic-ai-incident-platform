import json

import pytest

from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.domain.remediation import (
    RISK_BY_ACTION,
    RemediationActionType,
    RemediationPlan,
    RemediationStep,
    RiskTier,
)
from agentic_ai.llm.client import ChatResponse
from agentic_ai.llm.remediation import InvalidRemediationPlanError, plan_remediation


class _FakeClient:
    def __init__(self, contents):
        self._contents = [contents] if isinstance(contents, str) else list(contents)
        self.calls = 0
        self.last_kwargs = None

    def complete(self, messages, *, tools=None, json_schema=None):
        self.last_kwargs = {"tools": tools, "json_schema": json_schema}
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return ChatResponse(content=content)


def _analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Connection pool exhaustion.",
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )


def _step(action: str, service: str = "payment") -> dict:
    return {"action_type": action, "target_service": service, "rationale": "because"}


def _plan_json(*steps: dict) -> str:
    return json.dumps({"steps": list(steps)})


def _model_step(action: str) -> RemediationStep:
    return RemediationStep(
        action_type=action, target_service="payment", rationale="x"
    )


def test_every_action_type_has_a_risk_tier() -> None:
    assert set(RISK_BY_ACTION) == set(RemediationActionType)


def test_risk_tier_is_derived_from_action_not_authored_by_the_model() -> None:
    # The model tries to label a rollback as low risk; the field is ignored.
    raw = _step("rollback_deployment") | {"risk_tier": "low"}
    plan = RemediationPlan.model_validate_json(_plan_json(raw))
    assert plan.steps[0].risk_tier == RiskTier.HIGH


def test_risk_tier_is_not_in_the_schema_offered_to_the_model() -> None:
    assert "risk_tier" not in json.dumps(RemediationPlan.model_json_schema())


def test_highest_risk_and_approval_requirement() -> None:
    low = RemediationPlan(steps=[_model_step("scale_up")])
    high = RemediationPlan(steps=[_model_step("scale_up"), _model_step("failover")])
    assert low.highest_risk == RiskTier.LOW and not low.requires_human_approval
    assert high.highest_risk == RiskTier.HIGH and high.requires_human_approval
    assert RemediationPlan().highest_risk is None


def test_valid_plan_is_returned_and_no_tools_are_offered() -> None:
    client = _FakeClient(_plan_json(_step("increase_connection_pool")))
    plan = plan_remediation("Payment down", _analysis(), client=client)
    assert plan.steps[0].action_type == RemediationActionType.INCREASE_CONNECTION_POOL
    assert client.last_kwargs["tools"] is None


def test_action_outside_the_menu_is_retried_then_raises() -> None:
    client = _FakeClient(_plan_json(_step("delete_database")))
    with pytest.raises(InvalidRemediationPlanError):
        plan_remediation("Payment down", _analysis(), client=client)
    assert client.calls == 2


def test_unknown_service_is_retried() -> None:
    client = _FakeClient(
        [_plan_json(_step("scale_up", "nonexistent")), _plan_json(_step("scale_up"))]
    )
    plan = plan_remediation("Payment down", _analysis(), client=client)
    assert plan.steps[0].target_service == "payment"
    assert client.calls == 2
