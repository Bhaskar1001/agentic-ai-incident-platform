"""HTTP tests: resume, the cross-incident pending queue, and the execution ledger view.

Run over every available backend. ``forbid_llm`` is armed before each resume so a resume
that touched the planner would fail the test, not merely pass quietly.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from agentic_ai.api import create_app
from agentic_ai.domain.approvals import step_fingerprint
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.domain.persistence import IncidentRecord
from agentic_ai.domain.step_executions import (
    ExecutionSource,
    StepExecutionRecord,
    new_execution_id,
)
from agentic_ai.persistence.repository import IncidentPersistenceError
from tests.approval_helpers import (
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    LOW,
    WrappedStore,
    boom,
    decide,
    forbid_llm,
    guarded,
    install_pipeline_stubs,
    new_incident_id,
    saved_incident,
    state_for,
)
from tests.approval_helpers import submit as store_submit
from tests.test_ledger_repository import _store_tampered_snapshot


@pytest.fixture
def pipeline(monkeypatch):
    return install_pipeline_stubs(monkeypatch)


@pytest.fixture
def client(incident_store, pipeline):
    return TestClient(create_app(repository=incident_store))


def submit(client, incident_id=None, *, post="healthy"):
    response = client.post(
        "/incidents",
        json={
            "incident_description": "Payment service is degraded",
            "scenario": "connection_pool_exhaustion",
            "post_remediation_scenario": post,
            "incident_id": incident_id or new_incident_id(),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def approve_all(client, incident_id, action="approve"):
    for approval in client.get(f"/incidents/{incident_id}/approvals").json()["approvals"]:
        r = client.post(
            f"/incidents/{incident_id}/approvals/{approval['approval_id']}/{action}",
            json={"approver": "alice"},
        )
        assert r.status_code == 200, r.text


def resume(client, incident_id, **body):
    return client.post(f"/incidents/{incident_id}/resume", json=body)


def awaiting(client, pipeline, *actions):
    pipeline.actions = list(actions) or [HIGH_ROLLBACK]
    return submit(client)["incident_id"]


# =============================================================================
# POST /incidents/{id}/resume
# =============================================================================


def test_resume_runs_the_approved_plan_without_asking_the_model(client, pipeline, monkeypatch):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    forbid_llm(monkeypatch)

    response = resume(client, incident_id)  # no body at all

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "resolve" and body["status"] == "resolved"
    assert body["persisted"] is True and body["error"] is None
    assert body["verification"]["status"] == "recovered"
    (step,) = body["steps"]
    assert (step["action_type"], step["execution_status"], step["ran_now"]) == (
        "rollback_deployment",
        "executed",
        True,
    )
    assert len(body["plan_fingerprint"]) == 64 and body["snapshot_id"].startswith("SNP-")
    assert body["requested_by_verified"] is False  # a caller-supplied label, not an identity

    record = client.get(f"/incidents/{incident_id}").json()
    assert (
        record["status"] == "resolved"
        and record["stages"][-1]["stage_name"] == "resume_remediation"
    )
    (execution,) = client.get(f"/incidents/{incident_id}/executions").json()["executions"]
    assert execution["source"] == "resume" and execution["status"] == "succeeded"


def test_resume_accepts_an_operator_label_and_a_post_remediation_scenario(
    client, pipeline, monkeypatch
):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    forbid_llm(monkeypatch)

    body = resume(
        client,
        incident_id,
        requested_by="  operator-3  ",
        post_remediation_scenario="connection_pool_exhaustion",
    ).json()

    assert body["steps"][0]["execution_status"] == "executed"  # approval did run the step...
    assert body["outcome"] == "escalate" and body["status"] == "escalated"  # ...it did not fix it
    record = client.get(f"/incidents/{incident_id}").json()
    assert record["stages"][-1]["metadata"]["requested_by"] == "operator-3"


@pytest.mark.parametrize(
    ("decision", "code"), [(None, "approvals_pending"), ("reject", "approval_rejected")]
)
def test_resume_is_refused_while_approval_is_pending_or_rejected(
    client, pipeline, monkeypatch, decision, code
):
    incident_id = awaiting(client, pipeline)
    if decision:
        approve_all(client, incident_id, decision)
    forbid_llm(monkeypatch)

    response = resume(client, incident_id)

    assert response.status_code == 409 and response.json()["detail"].startswith(f"{code}:")
    assert client.get(f"/incidents/{incident_id}/executions").json()["executions"] == []
    assert client.get(f"/incidents/{incident_id}").json()["status"] == "awaiting_approval"


def test_resuming_twice_is_refused_the_second_time(client, pipeline, monkeypatch):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    forbid_llm(monkeypatch)
    assert resume(client, incident_id).status_code == 200

    response = resume(client, incident_id)

    assert response.status_code == 409 and response.json()["detail"].startswith("incident_closed:")
    assert len(client.get(f"/incidents/{incident_id}/executions").json()["executions"]) == 1


def test_an_incident_with_no_stored_plan_is_refused(incident_store, client, monkeypatch):
    record, _, (approval,) = saved_incident(incident_store, HIGH_ROLLBACK)  # saved with no snapshot
    client.post(
        f"/incidents/{record.incident_id}/approvals/{approval.approval_id}/approve",
        json={"approver": "alice"},
    )

    response = resume(client, record.incident_id)

    assert response.status_code == 409 and response.json()["detail"].startswith("no_plan_snapshot:")


def test_a_tampered_plan_is_refused(incident_store, client, pipeline, monkeypatch):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    fingerprint = incident_store.list_approvals(incident_id)[0].plan_fingerprint
    _store_tampered_snapshot(
        incident_store, incident_store.get_plan_snapshot(incident_id, fingerprint)
    )
    forbid_llm(monkeypatch)

    response = resume(client, incident_id)

    assert response.status_code == 409 and response.json()["detail"].startswith(
        "snapshot_integrity:"
    )
    assert client.get(f"/incidents/{incident_id}/executions").json()["executions"] == []


def test_a_step_with_an_unknown_outcome_blocks_the_resume(
    incident_store, client, pipeline, monkeypatch
):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    approval = incident_store.list_approvals(incident_id)[0]
    incident_store.claim_step_execution(  # an earlier attempt that started and never finished
        StepExecutionRecord(
            execution_id=new_execution_id(),
            incident_id=incident_id,
            plan_fingerprint=approval.plan_fingerprint,
            step_index=0,
            step_fingerprint=step_fingerprint(*HIGH_ROLLBACK),
            action_type=HIGH_ROLLBACK[0],
            target_service=HIGH_ROLLBACK[1],
            source=ExecutionSource.RESUME,
        )
    )
    forbid_llm(monkeypatch)

    response = resume(client, incident_id)

    assert response.status_code == 409 and response.json()["detail"].startswith("unknown_outcome:")
    (entry,) = client.get(f"/incidents/{incident_id}/executions").json()["executions"]
    assert entry["outcome_unknown"] is True and entry["status"] == "started"


def test_resuming_an_unknown_incident_is_404(client):
    response = resume(client, "INC-NOPE")
    assert response.status_code == 404 and "INC-NOPE" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": True},
        {"requested_by": "x" * 129},
        {"post_remediation_scenario": "not_a_scenario"},
        {"requested_by": 7},
    ],
)
def test_malformed_resume_requests_are_422_and_change_nothing(client, pipeline, body):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)

    assert client.post(f"/incidents/{incident_id}/resume", json=body).status_code == 422
    assert client.get(f"/incidents/{incident_id}/executions").json()["executions"] == []


def test_a_storage_failure_during_resume_is_503_and_leaks_nothing(incident_store, pipeline):
    def down(*args, **kwargs):
        raise IncidentPersistenceError("password=hunter2 connection refused")

    client = TestClient(create_app(repository=WrappedStore(incident_store, get_incident=down)))

    response = client.post("/incidents/INC-1/resume", json={})

    assert response.status_code == 503
    assert "hunter2" not in response.text and "refused" not in response.text


def test_an_unexpected_failure_during_resume_is_a_generic_500(incident_store, pipeline):
    client = TestClient(
        create_app(repository=WrappedStore(incident_store, get_incident=boom)),
        raise_server_exceptions=False,
    )

    response = client.post("/incidents/INC-1/resume", json={})

    assert response.status_code == 500
    assert "database down" not in response.text


def test_a_failed_final_save_is_reported_not_hidden(incident_store, pipeline, monkeypatch):
    client = TestClient(create_app(repository=WrappedStore(incident_store, save_incident=boom)))
    pipeline.actions = [HIGH_ROLLBACK]
    incident_id = new_incident_id()
    store_submit(incident_store, incident_id)
    decide(incident_store, incident_id, 0, ApprovalState.APPROVED)
    forbid_llm(monkeypatch)

    response = client.post(f"/incidents/{incident_id}/resume", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["persisted"] is False and "incident record not saved" in body["error"]
    assert body["steps"][0]["ran_now"] is True  # it did run, and the ledger knows
    assert len(incident_store.list_step_executions(incident_id)) == 1


def test_the_resume_docs_state_the_guarantees_and_the_missing_authentication(
    pipeline, incident_store
):
    spec = TestClient(create_app(repository=incident_store)).get("/openapi.json").json()
    text = spec["paths"]["/incidents/{incident_id}/resume"]["post"]["description"]
    assert "Never calls the LLM planner" in text
    assert "does not guarantee exactly-once" in text
    assert "NOT authenticated" in text


# =============================================================================
# GET /approvals/pending
# =============================================================================


def _queue(client, **params):
    response = client.get("/approvals/pending", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _svc() -> str:
    return f"svc-{uuid.uuid4().hex[:8]}"


def _make(incident_store, n=1, *, service=None, action="rollback_deployment"):
    return [saved_incident(incident_store, (action, service or _svc())) for _ in range(n)]


def test_the_queue_lists_minimal_fields_only(incident_store, client):
    svc = _svc()
    _make(incident_store, service=svc)

    body = _queue(client, target_service=svc)

    (item,) = body["approvals"]
    assert set(item) == {
        "approval_id",
        "incident_id",
        "step_index",
        "action_type",
        "target_service",
        "status",
        "requested_at",
    }  # no rationale, fingerprints or execution detail
    assert item["status"] == "pending" and item["target_service"] == svc
    assert (body["total"], body["limit"], body["offset"], body["has_more"]) == (1, 50, 0, False)


def test_the_queue_pages_through_everything_exactly_once(incident_store, client):
    svc = _svc()
    made = _make(incident_store, 5, service=svc)

    pages = [_queue(client, target_service=svc, limit=2, offset=o) for o in (0, 2, 4)]

    assert [len(p["approvals"]) for p in pages] == [2, 2, 1]
    assert [p["has_more"] for p in pages] == [True, True, False]
    assert all(p["total"] == 5 for p in pages)
    flat = [a["approval_id"] for p in pages for a in p["approvals"]]
    assert len(set(flat)) == 5 and set(flat) == {m[2][0].approval_id for m in made}
    assert _queue(client, target_service=svc, limit=2, offset=10)["approvals"] == []


def test_the_queue_is_oldest_first(incident_store, client):
    svc = _svc()
    _make(incident_store, 4, service=svc)
    times = [a["requested_at"] for a in _queue(client, target_service=svc)["approvals"]]
    assert times == sorted(times)


def test_the_queue_filters_by_action_and_target(incident_store, client):
    svc = _svc()
    _make(incident_store, 2, service=svc, action="rollback_deployment")
    _make(incident_store, 1, service=svc, action="failover")

    assert _queue(client, target_service=svc)["total"] == 3
    only_failover = _queue(client, target_service=svc, action_type="failover")
    assert (
        only_failover["total"] == 1 and only_failover["approvals"][0]["action_type"] == "failover"
    )
    assert _queue(client, target_service=_svc())["approvals"] == []


def test_decided_approvals_leave_the_queue(incident_store, client):
    svc = _svc()
    (first_record, _, (first,)), (second_record, _, (second,)) = _make(
        incident_store, 2, service=svc
    )

    incident_store.decide_approval(
        first_record.incident_id, first.approval_id, ApprovalState.APPROVED, "alice"
    )

    body = _queue(client, target_service=svc)
    assert [a["approval_id"] for a in body["approvals"]] == [second.approval_id] and body[
        "total"
    ] == 1


def test_closed_incidents_are_left_out_unless_requested(incident_store, client):
    svc = _svc()
    open_one = saved_incident(incident_store, ("rollback_deployment", svc))[0]
    closed = saved_incident(incident_store, ("rollback_deployment", svc))[0]
    incident_store.save_incident(
        IncidentRecord.from_workflow_state(
            state_for(closed.incident_id, guarded(("rollback_deployment", svc)), "resolve")
        )
    )

    assert [a["incident_id"] for a in _queue(client, target_service=svc)["approvals"]] == [
        open_one.incident_id
    ]
    both = _queue(client, target_service=svc, include_closed="true")
    assert both["total"] == 2


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 201}, {"limit": "many"}, {"offset": -1}, {"include_closed": "maybe"}],
)
def test_bad_queue_parameters_are_422(client, params):
    assert client.get("/approvals/pending", params=params).status_code == 422


def test_a_storage_failure_listing_the_queue_is_503(incident_store):
    client = TestClient(
        create_app(repository=WrappedStore(incident_store, list_pending_approvals=boom))
    )
    response = client.get("/approvals/pending")
    assert response.status_code == 503 and "database down" not in response.text


def test_the_queue_docs_do_not_pretend_there_is_authentication(client):
    spec = client.get("/openapi.json").json()
    assert "no authentication" in spec["paths"]["/approvals/pending"]["get"]["description"]


# =============================================================================
# GET /incidents/{id}/executions
# =============================================================================


def test_the_ledger_view_hides_output_and_error_text(client, pipeline, monkeypatch):
    incident_id = awaiting(client, pipeline)
    approve_all(client, incident_id)
    forbid_llm(monkeypatch)
    resume(client, incident_id)

    (entry,) = client.get(f"/incidents/{incident_id}/executions").json()["executions"]

    assert "output" not in entry and "error" not in entry
    assert entry["outcome_unknown"] is False and entry["finished_at"] is not None


def test_the_ledger_view_of_an_unknown_incident_is_404(client):
    assert client.get("/incidents/INC-NOPE/executions").status_code == 404


def test_low_risk_steps_run_by_the_workflow_appear_in_the_ledger(client, pipeline):
    incident_id = awaiting(client, pipeline, LOW, HIGH_FAILOVER)  # LOW runs now; HIGH waits
    entries = client.get(f"/incidents/{incident_id}/executions").json()["executions"]
    assert [(e["step_index"], e["source"], e["status"]) for e in entries] == [
        (0, "workflow", "succeeded")
    ]
