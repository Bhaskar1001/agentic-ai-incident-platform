import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from agentic_ai.domain.incident import Severity
from agentic_ai.llm import severity as severity_module
from agentic_ai.llm.severity import (
    InvalidAssessmentError,
    SeverityAssessment,
    assess_severity,
)


def test_valid_severity_assessment():
    assessment = SeverityAssessment(
        severity=Severity.CRITICAL,
        reasoning="All customers are affected by a production outage.",
    )

    assert assessment.severity == Severity.CRITICAL
    assert assessment.reasoning == (
        "All customers are affected by a production outage."
    )


def test_invalid_severity_is_rejected():
    with pytest.raises(ValidationError):
        SeverityAssessment(
            severity="super-critical",
            reasoning="The incident is severe.",
        )


def test_assess_severity_live():
    assessment = assess_severity(
        "Database is completely down and all customers are affected."
    )

    assert isinstance(assessment, SeverityAssessment)
    assert assessment.severity == Severity.CRITICAL
    assert assessment.reasoning


def test_invalid_llm_output_retries_and_raises_custom_error(monkeypatch):
    calls = 0

    def fake_chat(**kwargs):
        nonlocal calls

        calls += 1

        return SimpleNamespace(
            message=SimpleNamespace(
                content='{"severity": "super-critical", "reasoning": "Invalid"}'
            )
        )

    monkeypatch.setattr(severity_module.ollama, "chat", fake_chat)

    with pytest.raises(InvalidAssessmentError) as exc_info:
        assess_severity("Database is completely down.")

    assert calls == 2
    assert isinstance(exc_info.value.__cause__, ValidationError)