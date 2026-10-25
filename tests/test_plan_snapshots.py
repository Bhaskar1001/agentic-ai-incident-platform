"""Domain tests for immutable plan snapshots and the execution-ledger record."""

from datetime import datetime, timezone

import pytest

from agentic_ai.domain.approvals import (
    apply_persisted_decisions,
    decide,
    pending_requests_for,
    plan_fingerprint,
    plan_fingerprint_of,
)
from agentic_ai.domain.guardrails import ApprovalState, guard_plan
from agentic_ai.domain.plan_snapshots import (
    SNAPSHOT_VERSION,
    PlanSnapshot,
    SnapshotIntegrityError,
    SnapshotStep,
    build_snapshot,
)
from agentic_ai.domain.rca import KnowledgeBaseConsistency
from agentic_ai.domain.step_executions import (
    ExecutionSource,
    StepExecutionError,
    StepExecutionRecord,
    StepExecutionStatus,
)
from tests.approval_helpers import HIGH_FAILOVER, HIGH_ROLLBACK, LOW, analysis, guarded


def snapshot(*actions, confidence=0.9, services=("payment",)) -> PlanSnapshot:
    return build_snapshot(
        "INC-1", guarded(*actions, confidence=confidence), analysis(confidence), services
    )


# --- identity ---------------------------------------------------------------------------


def test_the_snapshot_keeps_the_fingerprint_approvals_are_bound_to() -> None:
    plan = guarded(LOW, HIGH_ROLLBACK)
    snap = build_snapshot("INC-1", plan, analysis(), ["payment"])
    assert snap.plan_fingerprint == plan_fingerprint_of(plan)  # existing approvals still match


def test_a_fresh_snapshot_verifies() -> None:
    snapshot(LOW, HIGH_ROLLBACK).verify()


def test_the_snapshot_records_the_plan_in_order() -> None:
    snap = snapshot(LOW, HIGH_ROLLBACK, HIGH_FAILOVER)
    assert [(s.action_type, s.target_service) for s in snap.steps] == [
        LOW,
        HIGH_ROLLBACK,
        HIGH_FAILOVER,
    ]
    assert snap.version == SNAPSHOT_VERSION


def test_services_are_stored_sorted_and_deduplicated() -> None:
    snap = snapshot(HIGH_ROLLBACK, services=["payment", "db", "payment"])
    assert snap.investigated_services == ["db", "payment"]


def test_prose_is_display_only_and_not_part_of_identity() -> None:
    """Like Phase 17's approvals: rationale and root-cause text are not execution parameters."""
    snap = snapshot(HIGH_ROLLBACK)
    reworded = snap.model_copy(
        update={
            "steps": [SnapshotStep(action_type=s.action_type, target_service=s.target_service, rationale="different")
                      for s in snap.steps],
            "root_cause": "a completely different explanation",
        }
    )  # fmt: skip
    reworded.verify()
    assert reworded.checksum == snap.checksum and reworded.plan_fingerprint == snap.plan_fingerprint


def test_different_guardrail_inputs_make_a_different_snapshot_of_the_same_plan() -> None:
    strong, weak = snapshot(HIGH_ROLLBACK, confidence=0.9), snapshot(HIGH_ROLLBACK, confidence=0.3)
    assert strong.plan_fingerprint == weak.plan_fingerprint
    assert strong.checksum != weak.checksum


# --- integrity ---------------------------------------------------------------------------


def _tamper(snap: PlanSnapshot, **changes) -> PlanSnapshot:
    return snap.model_copy(update=changes)


def test_changed_steps_are_detected() -> None:
    snap = snapshot(HIGH_ROLLBACK)
    swapped = _tamper(snap, steps=[SnapshotStep(action_type="failover", target_service="payment")])
    with pytest.raises(SnapshotIntegrityError, match="fingerprint"):
        swapped.verify()


def test_an_extra_step_is_detected() -> None:
    snap = snapshot(HIGH_ROLLBACK)
    longer = _tamper(
        snap, steps=[*snap.steps, SnapshotStep(action_type="failover", target_service="payment")]
    )
    with pytest.raises(SnapshotIntegrityError):
        longer.verify()


@pytest.mark.parametrize(
    "change",
    [
        {"root_cause_confidence": 0.99},
        {"knowledge_base_consistency": KnowledgeBaseConsistency.CONTRADICTED},
        {"investigated_services": ["payment", "extra"]},
    ],
)
def test_changed_guardrail_inputs_are_detected(change) -> None:
    with pytest.raises(SnapshotIntegrityError, match="checksum"):
        _tamper(snapshot(HIGH_ROLLBACK, confidence=0.5), **change).verify()


def test_a_changed_checksum_is_detected() -> None:
    with pytest.raises(SnapshotIntegrityError, match="checksum"):
        _tamper(snapshot(HIGH_ROLLBACK), checksum="0" * 64).verify()


def test_an_unsupported_version_is_refused() -> None:
    with pytest.raises(SnapshotIntegrityError, match="version"):
        _tamper(snapshot(HIGH_ROLLBACK), version=SNAPSHOT_VERSION + 1).verify()


def test_a_tampered_plan_cannot_inherit_approvals_from_the_original() -> None:
    """The point of the fingerprint: edit the plan and every approval stops matching."""
    original_plan = guarded(HIGH_ROLLBACK)
    (request,) = pending_requests_for("INC-1", original_plan)
    approved = decide(request, ApprovalState.APPROVED, "alice")

    tampered = snapshot(HIGH_FAILOVER)  # someone swaps the stored action
    assert tampered.plan_fingerprint != approved.plan_fingerprint

    rebuilt = guard_plan(
        tampered.to_plan(), tampered.to_analysis(), set(tampered.investigated_services)
    )
    assert apply_persisted_decisions(rebuilt, [approved]) == rebuilt  # nothing was approved


# --- rebuilding what the guardrails need -------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.9, 0.5, 0.2])
def test_reguarding_a_snapshot_reproduces_the_original_verdicts(confidence) -> None:
    """The property resume rests on: stored inputs + deterministic policy = same verdicts."""
    plan = guarded(LOW, HIGH_ROLLBACK, HIGH_FAILOVER, confidence=confidence)
    snap = build_snapshot("INC-1", plan, analysis(confidence), ["payment"])

    again = guard_plan(snap.to_plan(), snap.to_analysis(), set(snap.investigated_services))

    assert [(g.disposition, g.effective_tier) for g in again.steps] == [
        (g.disposition, g.effective_tier) for g in plan.steps
    ]
    assert plan_fingerprint_of(again) == snap.plan_fingerprint


def test_a_step_this_code_cannot_represent_is_an_integrity_error() -> None:
    bad = PlanSnapshot(
        snapshot_id="SNP-x",
        incident_id="INC-1",
        plan_fingerprint=plan_fingerprint([("format_disk", "payment")]),
        steps=[SnapshotStep(action_type="format_disk", target_service="payment")],
        root_cause_confidence=0.9,
        knowledge_base_consistency=KnowledgeBaseConsistency.SUPPORTED,
        investigated_services=["payment"],
        checksum="x" * 64,
    )
    with pytest.raises(SnapshotIntegrityError, match="cannot represent"):
        bad.to_plan()


def test_the_snapshot_cannot_be_changed_in_place() -> None:
    snap = snapshot(HIGH_ROLLBACK)
    with pytest.raises(Exception):
        snap.root_cause_confidence = 0.1  # frozen


# =============================================================================
# The execution record
# =============================================================================


def _record(**overrides) -> StepExecutionRecord:
    values = {
        "execution_id": "EXE-1",
        "incident_id": "INC-1",
        "plan_fingerprint": "a" * 64,
        "step_index": 0,
        "step_fingerprint": "b" * 64,
        "action_type": "rollback_deployment",
        "target_service": "payment",
        "source": ExecutionSource.WORKFLOW,
        **overrides,
    }
    return StepExecutionRecord(**values)


def test_a_new_execution_is_started_and_unknown_until_finished() -> None:
    record = _record()
    assert record.status == StepExecutionStatus.STARTED and record.is_unknown
    assert record.finished_at is None


@pytest.mark.parametrize("outcome", [StepExecutionStatus.SUCCEEDED, StepExecutionStatus.FAILED])
def test_a_started_execution_can_be_finished(outcome) -> None:
    at = datetime(2026, 10, 26, tzinfo=timezone.utc)
    done = _record().finish(outcome, "ok", None, at=at)
    assert done.status == outcome and done.finished_at == at and not done.is_unknown
    assert done.output == "ok"


@pytest.mark.parametrize("first", [StepExecutionStatus.SUCCEEDED, StepExecutionStatus.FAILED])
@pytest.mark.parametrize("second", [StepExecutionStatus.SUCCEEDED, StepExecutionStatus.FAILED])
def test_a_finished_execution_is_never_rewritten(first, second) -> None:
    done = _record().finish(first, "x", None)
    with pytest.raises(StepExecutionError, match="never rewritten"):
        done.finish(second, "y", None)


def test_an_execution_cannot_be_finished_as_started() -> None:
    with pytest.raises(StepExecutionError):
        _record().finish(StepExecutionStatus.STARTED, None, None)


def test_the_record_lifecycle_must_be_coherent() -> None:
    with pytest.raises(ValueError, match="started"):
        _record(finished_at=datetime.now(timezone.utc))
    with pytest.raises(ValueError, match="finished"):
        _record(status=StepExecutionStatus.SUCCEEDED)
