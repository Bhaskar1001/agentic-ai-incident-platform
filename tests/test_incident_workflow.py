from agentic_ai.domain.incident import Severity
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.workflows import incident as workflow_module


def test_critical_incident_is_escalated(monkeypatch):
    def fake_assess_severity(description: str) -> SeverityAssessment:
        return SeverityAssessment(
            severity=Severity.CRITICAL,
            reasoning="The production service is completely unavailable.",
        )

    monkeypatch.setattr(
        workflow_module,
        "assess_severity",
        fake_assess_severity,
    )

    graph = workflow_module.build_incident_workflow()

    result = graph.invoke(
        {
            "incident_description": (
                "Production database is completely unavailable and "
                "all customer requests are failing."
            ),
            "severity_assessment": None,
            "next_step": None,
        }
    )

    assert result["severity_assessment"].severity == Severity.CRITICAL
    assert result["next_step"] == "escalate"


def test_low_severity_incident_is_investigated(monkeypatch):
    def fake_assess_severity(description: str) -> SeverityAssessment:
        return SeverityAssessment(
            severity=Severity.LOW,
            reasoning="The issue affects a non-critical development environment.",
        )

    monkeypatch.setattr(
        workflow_module,
        "assess_severity",
        fake_assess_severity,
    )

    graph = workflow_module.build_incident_workflow()

    result = graph.invoke(
        {
            "incident_description": (
                "A development environment has a minor logging delay."
            ),
            "severity_assessment": None,
            "next_step": None,
        }
    )

    assert result["severity_assessment"].severity == Severity.LOW
    assert result["next_step"] == "investigate"


def test_route_by_severity_raises_when_assessment_is_none():
    import pytest

    with pytest.raises(ValueError, match="Severity assessment is missing"):
        workflow_module.route_by_severity(
            {
                "incident_description": "irrelevant",
                "severity_assessment": None,
                "next_step": None,
            }
        )