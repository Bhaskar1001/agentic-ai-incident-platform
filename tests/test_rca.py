import json

import pytest

from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.rca import (
    AlternativeHypothesis,
    KnowledgeBaseConsistency,
    RootCauseAnalysis,
)
from agentic_ai.llm.client import ChatResponse
from agentic_ai.llm.rca import (
    InvalidRootCauseAnalysisError,
    _findings_section,
    _knowledge_base_section,
    analyze_root_cause,
)


class _FakeClient:
    """A minimal LLMClient stand-in, same pattern as test_severity.py."""

    def __init__(self, contents):
        # Accept either a single content string or a list, consumed in order
        # across retries.
        self._contents = [contents] if isinstance(contents, str) else list(contents)
        self.calls = 0

    def complete(self, messages, *, tools=None, json_schema=None):
        content = self._contents[min(self.calls, len(self._contents) - 1)]
        self.calls += 1
        return ChatResponse(content=content)


def _investigation(
    findings: list[Finding] | None = None,
    evidence: list[Evidence] | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        summary="The payment service shows database connection pool exhaustion.",
        evidence=evidence or [],
        findings=findings
        if findings is not None
        else [
            Finding(
                statement="Connection pool utilisation is 100%.",
                model_assessed_confidence=0.95,
            ),
            Finding(
                statement="Error rate is 12.5%, above normal thresholds.",
                model_assessed_confidence=0.9,
            ),
        ],
        model_assessed_confidence=0.9,
        recommended_next_action="Increase pool size.",
        termination_reason=TerminationReason.COMPLETED,
    )


def _kb_evidence(results: list[dict]) -> Evidence:
    return Evidence(
        source_tool="search_knowledge_base",
        raw_output=json.dumps({"status": "success", "query": "x", "results": results}),
        observation="search_knowledge_base returned results",
    )


# --- domain model ----------------------------------------------------------


def test_root_cause_confidence_must_be_within_range() -> None:
    with pytest.raises(Exception):
        RootCauseAnalysis(
            root_cause="x",
            root_cause_confidence=1.5,
            knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
        )


def test_alternative_hypothesis_requires_both_fields() -> None:
    with pytest.raises(Exception):
        AlternativeHypothesis(statement="x")


# --- prompt section builders -------------------------------------------


def test_findings_section_includes_ids_for_citation() -> None:
    investigation = _investigation()

    section = _findings_section(investigation)

    for finding in investigation.findings:
        assert str(finding.id) in section
        assert finding.statement in section


def test_findings_section_handles_no_findings() -> None:
    investigation = _investigation(findings=[])

    assert "No findings" in _findings_section(investigation)


def test_knowledge_base_section_reports_no_search_performed() -> None:
    investigation = _investigation(evidence=[])

    section = _knowledge_base_section(investigation)

    assert "No knowledge base search was performed" in section


def test_knowledge_base_section_renders_actual_results() -> None:
    investigation = _investigation(
        evidence=[
            _kb_evidence(
                [
                    {
                        "id": "RB-001",
                        "title": "Database connection pool exhaustion under load",
                        "root_cause": "Pool sized for average load, not peak.",
                        "similarity": 0.72,
                    }
                ]
            )
        ]
    )

    section = _knowledge_base_section(investigation)

    assert "RB-001" in section
    assert "Pool sized for average load, not peak." in section


def test_knowledge_base_section_ignores_evidence_from_other_tools() -> None:
    investigation = _investigation(
        evidence=[
            Evidence(
                source_tool="get_mock_metrics",
                raw_output='{"status": "success"}',
                observation="metrics returned",
            )
        ]
    )

    assert "No knowledge base search was performed" in _knowledge_base_section(
        investigation
    )


# --- analyze_root_cause ---------------------------------------------------


def test_valid_analysis_is_returned() -> None:
    investigation = _investigation()
    finding_id = str(investigation.findings[0].id)

    valid_response = json.dumps(
        {
            "root_cause": "Database connection pool exhaustion under load.",
            "root_cause_confidence": 0.9,
            "supporting_finding_ids": [finding_id],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )

    fake = _FakeClient(valid_response)
    analysis = analyze_root_cause(investigation, client=fake)

    assert isinstance(analysis, RootCauseAnalysis)
    assert str(analysis.supporting_finding_ids[0]) == finding_id
    assert fake.calls == 1


def test_citation_of_a_nonexistent_finding_id_is_rejected_then_retried() -> None:
    investigation = _investigation()
    real_id = str(investigation.findings[0].id)

    bad_response = json.dumps(
        {
            "root_cause": "x",
            "root_cause_confidence": 0.9,
            "supporting_finding_ids": ["00000000-0000-0000-0000-000000000000"],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )
    good_response = json.dumps(
        {
            "root_cause": "x",
            "root_cause_confidence": 0.9,
            "supporting_finding_ids": [real_id],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )

    fake = _FakeClient([bad_response, good_response])
    analysis = analyze_root_cause(investigation, client=fake)

    assert str(analysis.supporting_finding_ids[0]) == real_id
    assert fake.calls == 2


def test_persistent_invalid_citation_raises_after_retry() -> None:
    investigation = _investigation()

    bad_response = json.dumps(
        {
            "root_cause": "x",
            "root_cause_confidence": 0.9,
            "supporting_finding_ids": ["00000000-0000-0000-0000-000000000000"],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )

    fake = _FakeClient(bad_response)

    with pytest.raises(InvalidRootCauseAnalysisError):
        analyze_root_cause(investigation, client=fake)

    assert fake.calls == 2


def test_malformed_json_is_rejected_then_retried() -> None:
    investigation = _investigation()
    finding_id = str(investigation.findings[0].id)

    good_response = json.dumps(
        {
            "root_cause": "x",
            "root_cause_confidence": 0.9,
            "supporting_finding_ids": [finding_id],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )

    fake = _FakeClient(["not json at all", good_response])
    analysis = analyze_root_cause(investigation, client=fake)

    assert str(analysis.supporting_finding_ids[0]) == finding_id


def test_empty_citation_list_is_always_valid() -> None:
    """Not citing any finding is a legitimate (if weak) analysis, not an error."""
    investigation = _investigation()

    response = json.dumps(
        {
            "root_cause": "Unclear from available evidence.",
            "root_cause_confidence": 0.2,
            "supporting_finding_ids": [],
            "alternative_hypotheses": [],
            "knowledge_base_consistency": "no_relevant_precedent",
            "knowledge_base_note": "",
        }
    )

    fake = _FakeClient(response)
    analysis = analyze_root_cause(investigation, client=fake)

    assert analysis.supporting_finding_ids == []
