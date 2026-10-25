"""Tests for operational incident persistence (Phase 16 - Step 1).

Covers:
- IncidentRecord immutability and from_workflow_state mappings
- Secret/credential redaction in persistence records
- Protocol adherence of InMemoryIncidentRepository
- CRUD, pagination, filtering, count, and duplicate handling
- Terminal state immutability and conflict detection
- Thread safety of InMemoryIncidentRepository
- Settings database_url configuration and validation
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from agentic_ai.config import ConfigurationError, Settings
from agentic_ai.domain.incident import Severity
from agentic_ai.domain.persistence import (
    IncidentLifecycleStatus,
    IncidentRecord,
    IncidentRemediationRecord,
    IncidentStageRecord,
)
from agentic_ai.persistence import (
    InMemoryIncidentRepository,
    IncidentConflictError,
    IncidentRepository,
)


# ==============================================================================
# Domain Model & Mapping Tests
# ==============================================================================


def test_incident_record_immutability() -> None:
    """IncidentRecord and sub-records must be immutable (frozen)."""
    record = IncidentRecord(
        incident_id="INC-001",
        correlation_id="CORR-001",
        scenario="connection_pool_exhaustion",
        incident_description="Test pool exhaustion",
        severity=Severity.HIGH,
        status=IncidentLifecycleStatus.DETECTED,
    )

    with pytest.raises(ValidationError):
        # Direct mutation attempt should fail on frozen Pydantic model
        record.incident_id = "INC-MUTATED"  # type: ignore[misc]


def test_incident_stage_and_remediation_redaction() -> None:
    """Sensitive parameters and metadata must be automatically sanitized."""
    stage = IncidentStageRecord(
        stage_name="investigation",
        status="completed",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        duration_ms=120.5,
        metadata={"api_key": "sk-secret123", "safe_field": "public"},
    )
    assert stage.metadata["api_key"] == "[REDACTED]"
    assert stage.metadata["safe_field"] == "public"

    remediation = IncidentRemediationRecord(
        step_order=1,
        action_type="scale_pool",
        target_service="db-pool",
        disposition="auto_executed",
        execution_status="success",
        parameters={"db_password": "super-secret-pw", "pool_size": 25},
    )
    assert remediation.parameters["db_password"] == "[REDACTED]"
    assert remediation.parameters["pool_size"] == 25


def test_from_workflow_state_resolved() -> None:
    """Mapping a resolved workflow state produces a RESOLVED IncidentRecord with resolved_at set."""
    dummy_state = {
        "incident_id": "INC-RESOLVED-1",
        "correlation_id": "CORR-RESOLVED-1",
        "scenario": "connection_pool_exhaustion",
        "incident_description": "Database pool saturated",
        "severity_assessment": {"severity": "high"},
        "next_step": "resolve",
    }
    record = IncidentRecord.from_workflow_state(dummy_state)

    assert record.incident_id == "INC-RESOLVED-1"
    assert record.correlation_id == "CORR-RESOLVED-1"
    assert record.severity == Severity.HIGH
    assert record.status == IncidentLifecycleStatus.RESOLVED
    assert record.final_outcome == "resolve"
    assert record.resolved_at is not None


def test_from_workflow_state_escalated() -> None:
    """Mapping an escalated workflow state produces an ESCALATED IncidentRecord."""
    dummy_state = {
        "incident_id": "INC-ESC-1",
        "correlation_id": "CORR-ESC-1",
        "scenario": "upstream_dependency_failure",
        "incident_description": "Payment gateway down",
        "severity_assessment": {"severity": "critical"},
        "next_step": "escalate",
    }
    record = IncidentRecord.from_workflow_state(dummy_state)

    assert record.status == IncidentLifecycleStatus.ESCALATED
    assert record.final_outcome == "escalate"
    assert record.severity == Severity.CRITICAL
    assert record.resolved_at is not None


def test_from_workflow_state_monitored() -> None:
    """Mapping a healthy/low-severity state produces a MONITORED IncidentRecord."""
    dummy_state = {
        "incident_id": "INC-MON-1",
        "correlation_id": "CORR-MON-1",
        "scenario": "healthy",
        "incident_description": "Routine health check",
        "severity_assessment": {"severity": "low"},
        "next_step": "monitor",
    }
    record = IncidentRecord.from_workflow_state(dummy_state)

    assert record.status == IncidentLifecycleStatus.MONITORED
    assert record.final_outcome == "monitor"
    assert record.severity == Severity.LOW
    assert record.resolved_at is not None


def test_from_workflow_state_in_progress_states() -> None:
    """Mapping intermediate workflow states maps to DETECTED, INVESTIGATING, or MITIGATION."""
    # Detected only
    rec_detected = IncidentRecord.from_workflow_state({"incident_id": "INC-D"})
    assert rec_detected.status == IncidentLifecycleStatus.DETECTED
    assert rec_detected.resolved_at is None

    # Investigating
    rec_investigating = IncidentRecord.from_workflow_state(
        {
            "incident_id": "INC-I",
            "investigation": {"findings": "some findings"},
        }
    )
    assert rec_investigating.status == IncidentLifecycleStatus.INVESTIGATING

    # Mitigation
    rec_mitigation = IncidentRecord.from_workflow_state(
        {
            "incident_id": "INC-M",
            "remediation_plan": {"steps": []},
        }
    )
    assert rec_mitigation.status == IncidentLifecycleStatus.MITIGATION


# ==============================================================================
# Repository Contract & InMemory Implementation Tests
# ==============================================================================


def test_in_memory_repository_implements_protocol() -> None:
    """InMemoryIncidentRepository must satisfy the runtime IncidentRepository protocol."""
    repo = InMemoryIncidentRepository()
    assert isinstance(repo, IncidentRepository)


def test_save_and_get_incident() -> None:
    """Basic save and retrieve by incident_id."""
    repo = InMemoryIncidentRepository()
    record = IncidentRecord(
        incident_id="INC-100",
        correlation_id="CORR-100",
        scenario="healthy",
        incident_description="Health check",
        severity=Severity.LOW,
        status=IncidentLifecycleStatus.DETECTED,
    )

    saved = repo.save_incident(record)
    assert saved == record

    retrieved = repo.get_incident("INC-100")
    assert retrieved == record

    missing = repo.get_incident("INC-NONEXISTENT")
    assert missing is None


def test_get_by_correlation_id() -> None:
    """Retrieving incidents by correlation ID returns all matching records."""
    repo = InMemoryIncidentRepository()
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc)

    rec1 = IncidentRecord(
        incident_id="INC-A1",
        correlation_id="CORR-MATCH",
        scenario="test",
        incident_description="desc 1",
        severity=Severity.LOW,
        status=IncidentLifecycleStatus.DETECTED,
        created_at=t1,
    )
    rec2 = IncidentRecord(
        incident_id="INC-A2",
        correlation_id="CORR-MATCH",
        scenario="test",
        incident_description="desc 2",
        severity=Severity.HIGH,
        status=IncidentLifecycleStatus.INVESTIGATING,
        created_at=t2,
    )
    rec3 = IncidentRecord(
        incident_id="INC-OTHER",
        correlation_id="CORR-OTHER",
        scenario="test",
        incident_description="desc 3",
        severity=Severity.LOW,
        status=IncidentLifecycleStatus.DETECTED,
    )

    repo.save_incident(rec1)
    repo.save_incident(rec2)
    repo.save_incident(rec3)

    matching = repo.get_by_correlation_id("CORR-MATCH")
    assert len(matching) == 2
    # Should be sorted descending by created_at
    assert matching[0].incident_id == "INC-A2"
    assert matching[1].incident_id == "INC-A1"


def test_list_incidents_pagination_and_filtering() -> None:
    """list_incidents handles limit, offset, and status filtering properly."""
    repo = InMemoryIncidentRepository()
    for i in range(10):
        repo.save_incident(
            IncidentRecord(
                incident_id=f"INC-{i:02d}",
                correlation_id=f"CORR-{i:02d}",
                scenario="scenario",
                incident_description=f"Incident {i}",
                severity=Severity.LOW,
                status=(
                    IncidentLifecycleStatus.RESOLVED
                    if i % 2 == 0
                    else IncidentLifecycleStatus.ESCALATED
                ),
                created_at=datetime(2026, 1, 1, 10, i, tzinfo=timezone.utc),
            )
        )

    assert repo.count() == 10
    assert repo.count(status=IncidentLifecycleStatus.RESOLVED) == 5
    assert repo.count(status=IncidentLifecycleStatus.ESCALATED) == 5

    # Page 1 (limit 3, offset 0)
    page1 = repo.list_incidents(limit=3, offset=0)
    assert len(page1) == 3
    assert page1[0].incident_id == "INC-09"
    assert page1[1].incident_id == "INC-08"
    assert page1[2].incident_id == "INC-07"

    # Page 2 (limit 3, offset 3)
    page2 = repo.list_incidents(limit=3, offset=3)
    assert len(page2) == 3
    assert page2[0].incident_id == "INC-06"
    assert page2[1].incident_id == "INC-05"
    assert page2[2].incident_id == "INC-04"

    # Filtered listing
    resolved = repo.list_incidents(status=IncidentLifecycleStatus.RESOLVED)
    assert len(resolved) == 5
    assert all(r.status == IncidentLifecycleStatus.RESOLVED for r in resolved)


def test_list_incidents_invalid_pagination_bounds() -> None:
    """Invalid limit or offset values must raise ValueError."""
    repo = InMemoryIncidentRepository()
    with pytest.raises(ValueError, match="limit must be >= 1"):
        repo.list_incidents(limit=0)

    with pytest.raises(ValueError, match="offset must be >= 0"):
        repo.list_incidents(offset=-1)


def test_terminal_incident_conflict_detection() -> None:
    """Terminal incident records cannot be modified with conflicting outcomes or reverted."""
    repo = InMemoryIncidentRepository()
    terminal_rec = IncidentRecord(
        incident_id="INC-TERM-1",
        correlation_id="CORR-TERM-1",
        scenario="pool",
        incident_description="Initial description",
        severity=Severity.HIGH,
        status=IncidentLifecycleStatus.RESOLVED,
        final_outcome="resolve",
    )
    repo.save_incident(terminal_rec)

    # 1. Idempotent re-save with identical data should succeed
    repo.save_incident(terminal_rec)
    assert repo.count() == 1

    # 2. Conflicting outcome must raise IncidentConflictError
    conflicting_outcome = IncidentRecord(
        incident_id="INC-TERM-1",
        correlation_id="CORR-TERM-1",
        scenario="pool",
        incident_description="Initial description",
        severity=Severity.HIGH,
        status=IncidentLifecycleStatus.ESCALATED,
        final_outcome="escalate",
    )
    with pytest.raises(IncidentConflictError, match="conflicting outcome"):
        repo.save_incident(conflicting_outcome)

    # 3. Conflicting severity must raise IncidentConflictError
    conflicting_severity = IncidentRecord(
        incident_id="INC-TERM-1",
        correlation_id="CORR-TERM-1",
        scenario="pool",
        incident_description="Initial description",
        severity=Severity.CRITICAL,
        status=IncidentLifecycleStatus.RESOLVED,
        final_outcome="resolve",
    )
    with pytest.raises(IncidentConflictError, match="severity"):
        repo.save_incident(conflicting_severity)

    # 4. Attempting to revert terminal status to non-terminal must raise IncidentConflictError
    revert_status = IncidentRecord(
        incident_id="INC-TERM-1",
        correlation_id="CORR-TERM-1",
        scenario="pool",
        incident_description="Initial description",
        severity=Severity.HIGH,
        status=IncidentLifecycleStatus.INVESTIGATING,
    )
    with pytest.raises(IncidentConflictError, match="Cannot revert terminal incident"):
        repo.save_incident(revert_status)


def test_non_terminal_incident_transition() -> None:
    """Non-terminal incident records can transition across lifecycle states."""
    repo = InMemoryIncidentRepository()
    initial_rec = IncidentRecord(
        incident_id="INC-PROG-1",
        correlation_id="CORR-PROG-1",
        scenario="pool",
        incident_description="Alert received",
        severity=Severity.MEDIUM,
        status=IncidentLifecycleStatus.DETECTED,
    )
    repo.save_incident(initial_rec)

    # Transition to INVESTIGATING
    investigating_rec = IncidentRecord(
        incident_id="INC-PROG-1",
        correlation_id="CORR-PROG-1",
        scenario="pool",
        incident_description="Alert received",
        severity=Severity.MEDIUM,
        status=IncidentLifecycleStatus.INVESTIGATING,
    )
    repo.save_incident(investigating_rec)
    assert repo.get_incident("INC-PROG-1").status == IncidentLifecycleStatus.INVESTIGATING

    # Transition to RESOLVED (terminal)
    resolved_rec = IncidentRecord(
        incident_id="INC-PROG-1",
        correlation_id="CORR-PROG-1",
        scenario="pool",
        incident_description="Alert received",
        severity=Severity.MEDIUM,
        status=IncidentLifecycleStatus.RESOLVED,
        final_outcome="resolve",
    )
    repo.save_incident(resolved_rec)
    assert repo.get_incident("INC-PROG-1").status == IncidentLifecycleStatus.RESOLVED


def test_in_memory_repository_thread_safety() -> None:
    """Concurrent writes across multiple threads must remain safe and consistent."""
    repo = InMemoryIncidentRepository()
    num_threads = 8
    items_per_thread = 25

    def worker(thread_idx: int) -> None:
        for i in range(items_per_thread):
            rec = IncidentRecord(
                incident_id=f"INC-T{thread_idx}-{i}",
                correlation_id=f"CORR-T{thread_idx}",
                scenario="scenario",
                incident_description=f"Thread {thread_idx} item {i}",
                severity=Severity.LOW,
                status=IncidentLifecycleStatus.DETECTED,
            )
            repo.save_incident(rec)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, t) for t in range(num_threads)]
        for f in futures:
            f.result()

    total_expected = num_threads * items_per_thread
    assert repo.count() == total_expected
    listed = repo.list_incidents(limit=total_expected + 10)
    assert len(listed) == total_expected


# ==============================================================================
# Settings & Configuration Tests
# ==============================================================================


def test_database_url_optional_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """database_url defaults to None when not configured in environment."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings.from_env()
    assert settings.database_url is None
    assert settings.get_database_url_value() is None


def test_database_url_valid_postgresql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid PostgreSQL connection URLs are masked with SecretStr."""
    url = "postgresql://user:pass123@localhost:5432/incident_db"
    monkeypatch.setenv("DATABASE_URL", url)
    settings = Settings.from_env()

    assert settings.database_url is not None
    assert settings.get_database_url_value() == url
    # Ensure SecretStr masks credentials when converted to str/repr
    assert "pass123" not in str(settings.database_url)
    assert "pass123" not in repr(settings.database_url)


def test_database_url_invalid_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid database URL schemes fail validation with clear error."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    with pytest.raises(ConfigurationError, match="database_url"):
        Settings.from_env()

    with pytest.raises(ValidationError, match="DATABASE_URL must use postgresql"):
        Settings(database_url="sqlite:///test.db")


def test_from_workflow_state_maps_a_real_guarded_plan_and_execution() -> None:
    """Regression: the mapping once read ``step_order`` and ``parameters`` off
    RemediationStep, which has neither, so any real run with a plan crashed.
    Built from the real planner, guardrail and executor objects, not stand-ins.
    """
    from agentic_ai.domain.guardrails import guard_plan
    from agentic_ai.domain.rca import KnowledgeBaseConsistency, RootCauseAnalysis
    from agentic_ai.domain.remediation import RemediationPlan, RemediationStep
    from agentic_ai.tools.mock_remediation import execute_guarded_plan

    plan = RemediationPlan(
        steps=[
            RemediationStep(action_type="scale_up", target_service="payment", rationale="r"),
            RemediationStep(
                action_type="rollback_deployment", target_service="payment", rationale="r"
            ),
        ]
    )
    analysis = RootCauseAnalysis(
        root_cause="x",
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.SUPPORTED,
    )
    guarded = guard_plan(plan, analysis, {"payment"})

    record = IncidentRecord.from_workflow_state(
        {
            "incident_description": "d",
            "guarded_plan": guarded,
            "remediation_execution": execute_guarded_plan(guarded),
            "next_step": "escalate",
        }
    )

    assert [r.step_order for r in record.remediations] == [0, 1]
    assert record.remediations[0].execution_status == "executed"
    assert record.remediations[0].disposition == "allowed"
    # HIGH risk and unapproved: skipped, with its disposition preserved.
    assert record.remediations[1].execution_status == "skipped"
    assert record.remediations[1].disposition == "requires_approval"
