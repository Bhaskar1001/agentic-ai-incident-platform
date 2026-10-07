import pytest
from pydantic import ValidationError

from agentic_ai.domain.incident import Severity
from agentic_ai.llm.client import ChatResponse
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
    """End-to-end against the real configured LLM provider.

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


class _FakeClient:
    """A minimal LLMClient stand-in for deterministic, LLM-free tests."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def complete(self, messages, *, tools=None, json_schema=None):
        self.calls += 1
        return ChatResponse(content=self.content)


def test_invalid_llm_output_retries_and_raises_custom_error():
    fake = _FakeClient('{"severity": "super-critical", "reasoning": "Invalid"}')

    with pytest.raises(InvalidAssessmentError) as exc_info:
        assess_severity("Database is completely down.", client=fake)

    assert fake.calls == 2
    assert isinstance(exc_info.value.__cause__, ValidationError)
