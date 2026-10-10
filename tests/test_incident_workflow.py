import pytest

from agentic_ai.agents.investigation import InvalidConclusionError
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.investigation import (
    Evidence,
    Finding,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
from agentic_ai.llm.rca import InvalidRootCauseAnalysisError
from agentic_ai.llm.severity import SeverityAssessment
from agentic_ai.tools.mock_tools import (
    Scenario,
    get_mock_metrics,
    get_mock_pod_status,
)
from agentic_ai.workflows import incident as workflow_module
from agentic_ai.workflows.incident import handle_incident, route_by_severity


def _assessment(severity: Severity) -> SeverityAssessment:
    return SeverityAssessment(severity=severity, reasoning="test reasoning")


def _evidence(scenario: Scenario) -> list[Evidence]:
    """Real tool output, so the escalation policy has numbers to read."""
    return [
        Evidence(
            source_tool=name,
            tool_arguments={"service": "payment"},
            raw_output=tool("payment", scenario),
            observation=f"{name} returned data",
        )
        for name, tool in (
            ("get_mock_metrics", get_mock_metrics),
            ("get_mock_pod_status", get_mock_pod_status),
        )
    ]


def _result(
    termination: TerminationReason = TerminationReason.COMPLETED,
    scenario: Scenario = Scenario.HEALTHY,
    findings: list[Finding] | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        summary="Investigation summary.",
        model_assessed_confidence=0.8,
        recommended_next_action="No action required.",
        termination_reason=termination,
        evidence=_evidence(scenario),
        findings=findings or [],
    )


def _finding() -> Finding:
    return Finding(
        statement="Connection pool utilisation is 100%.",
        model_assessed_confidence=0.9,
    )


def _analysis() -> RootCauseAnalysis:
    return RootCauseAnalysis(
        root_cause="Database connection pool exhaustion.",
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )


@pytest.fixture
def stub_dependencies(monkeypatch):
    """Replace the LLM-backed steps with deterministic stubs."""

    def _apply(
        severity: Severity,
        investigation=None,
        error: Exception | None = None,
        rca_result=None,
        rca_error: Exception | None = None,
    ):
        monkeypatch.setattr(
            workflow_module,
            "assess_severity",
            lambda description: _assessment(severity),
        )

        def fake_investigate(description, scenario=None):
            if error is not None:
                raise error
            return investigation if investigation is not None else _result()

        monkeypatch.setattr(workflow_module, "investigate", fake_investigate)

        def fake_analyze_root_cause(result):
            if rca_error is not None:
                raise rca_error
            return rca_result if rca_result is not None else _analysis()

        monkeypatch.setattr(
            workflow_module, "analyze_root_cause", fake_analyze_root_cause
        )

    return _apply


# --- flow ---------------------------------------------------------------


def test_every_incident_is_investigated_before_routing(stub_dependencies) -> None:
    """Even a critical incident is investigated, so escalation carries evidence."""
    stub_dependencies(Severity.CRITICAL)

    state = handle_incident("Payment service is down")

    assert state["investigation"] is not None
    assert state["next_step"] == "escalate"


def test_high_severity_escalates(stub_dependencies) -> None:
    stub_dependencies(Severity.HIGH)

    assert handle_incident("Elevated errors")["next_step"] == "escalate"


def test_low_severity_with_complete_investigation_is_monitored(
    stub_dependencies,
) -> None:
    stub_dependencies(Severity.LOW)

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["investigation"].is_complete


def test_severity_assessment_is_carried_into_state(stub_dependencies) -> None:
    stub_dependencies(Severity.MEDIUM)

    state = handle_incident("Slow checkout")

    assert state["severity_assessment"].severity is Severity.MEDIUM


# --- degradation --------------------------------------------------------


def test_incomplete_investigation_escalates_even_when_severity_is_low(
    stub_dependencies,
) -> None:
    """An unexplained incident gets human eyes regardless of severity."""
    stub_dependencies(
        Severity.LOW,
        investigation=_result(TerminationReason.MAX_STEPS_REACHED),
    )

    assert handle_incident("Minor logging delay")["next_step"] == "escalate"


def test_failed_investigation_does_not_sink_the_workflow(
    stub_dependencies,
) -> None:
    """The incident still needs routing even if investigation failed."""
    stub_dependencies(
        Severity.LOW,
        error=InvalidConclusionError("model produced nothing usable"),
    )

    state = handle_incident("Minor logging delay")

    assert state["investigation"] is None
    assert "nothing usable" in state["investigation_error"]
    assert state["next_step"] == "escalate"


def test_unexpected_errors_are_not_swallowed(stub_dependencies) -> None:
    """Only investigation failures are handled; real bugs must surface."""
    stub_dependencies(Severity.LOW, error=RuntimeError("connection refused"))

    with pytest.raises(RuntimeError, match="connection refused"):
        handle_incident("Minor logging delay")


# --- routing in isolation -----------------------------------------------


def test_router_requires_a_severity_assessment() -> None:
    with pytest.raises(ValueError, match="Severity assessment is missing"):
        route_by_severity(
            {
                "incident_description": "x",
                "scenario": "connection_pool_exhaustion",
                "severity_assessment": None,
                "investigation": _result(),
                "investigation_error": None,
                "next_step": None,
            }
        )


# --- escalation on evidence ---------------------------------------------


def test_evidence_escalates_even_when_severity_looks_mild(
    stub_dependencies,
) -> None:
    """A mildly-worded report with alarming numbers must still escalate.

    Severity is assessed from the description before any data exists, so it
    cannot be the only input to the decision.
    """
    stub_dependencies(
        Severity.LOW,
        investigation=_result(scenario=Scenario.UPSTREAM_DEPENDENCY_FAILURE),
    )

    assert handle_incident("Occasional checkout hiccups")["next_step"] == "escalate"


def test_no_evidence_escalates(stub_dependencies) -> None:
    """An incident nobody could gather data on needs a human."""
    stub_dependencies(
        Severity.LOW,
        investigation=InvestigationResult(
            summary="Nothing gathered.",
            model_assessed_confidence=0.1,
            recommended_next_action="Escalate.",
            termination_reason=TerminationReason.COMPLETED,
        ),
    )

    assert handle_incident("Something odd")["next_step"] == "escalate"


# --- root cause analysis integration -------------------------------------


def test_rca_runs_on_the_escalation_path_when_findings_exist(
    stub_dependencies,
) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is not None
    assert state["root_cause_analysis"].root_cause == "Database connection pool exhaustion."
    assert state["root_cause_analysis_error"] is None


def test_rca_does_not_run_on_the_monitor_path(stub_dependencies) -> None:
    stub_dependencies(
        Severity.LOW,
        investigation=_result(findings=[_finding()]),
    )

    state = handle_incident("Minor logging delay")

    assert state["next_step"] == "monitor"
    assert state["root_cause_analysis"] is None


def test_rca_is_skipped_when_investigation_produced_no_findings(
    stub_dependencies,
) -> None:
    """Nothing to synthesize - RCA must not be asked to invent a conclusion."""
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[]),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is None
    assert state["root_cause_analysis_error"] is None


def test_rca_is_skipped_when_investigation_itself_failed(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        error=InvalidConclusionError("model produced nothing usable"),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["investigation"] is None
    assert state["root_cause_analysis"] is None


def test_rca_failure_does_not_block_escalation(stub_dependencies) -> None:
    """A human is already being paged; a missing RCA should not stop that."""
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_error=InvalidRootCauseAnalysisError("model could not produce a valid analysis"),
    )

    state = handle_incident("Payment service is down")

    assert state["next_step"] == "escalate"
    assert state["root_cause_analysis"] is None
    assert "could not produce" in state["root_cause_analysis_error"]


def test_unexpected_rca_errors_are_not_swallowed(stub_dependencies) -> None:
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_error=RuntimeError("connection refused"),
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        handle_incident("Payment service is down")


def test_routing_is_unaffected_by_rca_regardless_of_its_content(
    stub_dependencies,
) -> None:
    """Routing is decided before RCA runs - this is enforced by graph order,
    not by RCA's content, but a passing test here is a regression guard if
    that ever changes.
    """
    low_confidence_analysis = RootCauseAnalysis(
        root_cause="Unclear.",
        root_cause_confidence=0.05,
        knowledge_base_consistency=KnowledgeBaseConsistency.NO_RELEVANT_PRECEDENT,
    )
    stub_dependencies(
        Severity.CRITICAL,
        investigation=_result(findings=[_finding()]),
        rca_result=low_confidence_analysis,
    )

    # Severity alone already routes this to escalate; a low-confidence RCA
    # conclusion must not change that, since RCA runs strictly after routing.
    assert handle_incident("Payment service is down")["next_step"] == "escalate"
