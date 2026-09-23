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
    """End-to-end against the real model.

    Asserts a *range* rather than an exact value. The model returns CRITICAL
    for this input most of the time and HIGH occasionally; pinning the exact
    label made the suite fail at random, which trains you to ignore failures.
    What matters for routing is that a total outage lands in the severe band.
    """
    assessment = assess_severity(
        "Database is completely down and all customers are affected."
    )

    assert isinstance(assessment, SeverityAssessment)
    assert assessment.severity in {Severity.HIGH, Severity.CRITICAL}
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