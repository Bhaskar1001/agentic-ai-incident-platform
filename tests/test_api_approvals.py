"""HTTP tests for the approval endpoints, over every available backend.

``POST /incidents`` runs the real planner, guardrails, executor and verification (only
the LLM-backed steps are controlled), so an approval here is created and then enforced
by the same code that runs in production.
"""

import pytest
from fastapi.testclient import TestClient

from agentic_ai.api import create_app
from agentic_ai.domain.guardrails import ApprovalState
from agentic_ai.persistence.in_memory import InMemoryIncidentRepository
from tests.approval_helpers import (
    HIGH_FAILOVER,
    HIGH_ROLLBACK,
    LOW,
    install_pipeline_stubs,
    new_incident_id,
)


@pytest.fixture
def pipeline(monkeypatch):
    return install_pipeline_stubs(monkeypatch)


@pytest.fixture
def client(incident_store, pipeline):
    return TestClient(create_app(repository=incident_store))


def submit(client, incident_id, *, post="healthy"):
    response = client.post(
        "/incidents",
        json={
            "incident_description": "Payment service is degraded",
            "scenario": "connection_pool_exhaustion",
            "post_remediation_scenario": post,
            "incident_id": incident_id,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def approvals_url(incident_id, approval_id=None, action=None):
    base = f"/incidents/{incident_id}/approvals"
    return base if approval_id is None else f"{base}/{approval_id}/{action}"


def pending_one(client):
    """Submit a high-risk incident; return (incident_id, approval_id)."""
    incident_id = new_incident_id()
    body = submit(client, incident_id)
    assert len(body["pending_approval_ids"]) == 1
    return incident_id, body["pending_approval_ids"][0]


def decide(client, incident_id, approval_id, action, **body):
    return client.post(
        approvals_url(incident_id, approval_id, action),
        json={"approver": "alice", **body},
    )


# --- POST /incidents reports what it is waiting on ------------------------------------------


def test_a_high_risk_run_reports_its_pending_approvals(client):
    incident_id = new_incident_id()
    body = submit(client, incident_id)

    assert body["outcome"] == "escalate"
    assert body["persisted"] is True
    assert len(body["pending_approval_ids"]) == 1


def test_a_low_risk_run_has_no_pending_approvals(client, pipeline):
    pipeline.actions = [LOW]
    body = submit(client, new_incident_id())
    assert body["pending_approval_ids"] == [] and body["outcome"] == "resolve"


# --- GET approvals -------------------------------------------------------------------------------


def test_listing_shows_what_each_approval_authorizes(client):
    incident_id, approval_id = pending_one(client)

    response = client.get(approvals_url(incident_id))

    assert response.status_code == 200
    body = response.json()
    assert body["incident_id"] == incident_id
    (approval,) = body["approvals"]
    assert approval["approval_id"] == approval_id
    assert approval["status"] == "pending"
    assert (approval["action_type"], approval["target_service"]) == HIGH_ROLLBACK
    assert approval["step_index"] == 0 and approval["rationale"]
    assert approval["requested_at"] and approval["decided_at"] is None
    assert approval["approver"] is None and approval["decision_reason"] is None
    assert approval["stale"] is False and approval["decidable"] is True


def test_listing_can_filter_by_status(client):
    incident_id, approval_id = pending_one(client)

    def ids(status):
        r = client.get(approvals_url(incident_id), params={"status": status})
        assert r.status_code == 200
        return [a["approval_id"] for a in r.json()["approvals"]]

    assert ids("pending") == [approval_id]
    assert ids("approved") == [] and ids("rejected") == []

    assert decide(client, incident_id, approval_id, "approve").status_code == 200
    assert ids("pending") == [] and ids("approved") == [approval_id]


def test_an_unknown_status_filter_is_rejected(client):
    incident_id, _ = pending_one(client)
    response = client.get(approvals_url(incident_id), params={"status": "maybe"})
    assert response.status_code == 422


def test_listing_an_unknown_incident_is_404(client):
    response = client.get(approvals_url("INC-NOPE"))
    assert response.status_code == 404 and "INC-NOPE" in response.json()["detail"]


# --- approve / reject ------------------------------------------------------------------------------


def test_approving_records_the_decision_and_says_the_identity_is_unverified(client):
    incident_id, approval_id = pending_one(client)

    response = decide(client, incident_id, approval_id, "approve", reason="  ticket 42  ")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved" and body["approver"] == "alice"
    assert body["decision_reason"] == "ticket 42"
    assert body["decided_at"] is not None
    assert body["approver_verified"] is False  # the name is caller-supplied, not authenticated
    assert body["decidable"] is False  # decided: nothing further to decide


def test_rejecting_records_the_decision(client):
    incident_id, approval_id = pending_one(client)

    response = decide(client, incident_id, approval_id, "reject", reason="too risky")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected" and body["decision_reason"] == "too risky"
    assert body["approver_verified"] is False


def test_a_decision_does_not_execute_or_resolve_anything(client):
    incident_id, approval_id = pending_one(client)

    decide(client, incident_id, approval_id, "approve")

    record = client.get(f"/incidents/{incident_id}").json()
    assert record["status"] == "awaiting_approval"
    assert [r["execution_status"] for r in record["remediations"]] == ["skipped"]


def test_resubmitting_after_approval_executes_verifies_and_resolves(client):
    incident_id, approval_id = pending_one(client)
    decide(client, incident_id, approval_id, "approve")

    body = submit(client, incident_id, post="healthy")

    assert body["outcome"] == "resolve" and body["persisted"] is True
    assert body["pending_approval_ids"] == []
    assert body["verification"]["status"] == "recovered"
    record = client.get(f"/incidents/{incident_id}").json()
    assert record["status"] == "resolved"
    assert [r["execution_status"] for r in record["remediations"]] == ["executed"]


def test_resubmitting_after_rejection_still_does_not_execute(client):
    incident_id, approval_id = pending_one(client)
    decide(client, incident_id, approval_id, "reject")

    body = submit(client, incident_id)

    assert body["outcome"] == "escalate" and body["verification"]["status"] == "skipped"
    record = client.get(f"/incidents/{incident_id}").json()
    assert [r["execution_status"] for r in record["remediations"]] == ["skipped"]


# --- not found --------------------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_deciding_on_an_unknown_incident_is_404(client, action):
    response = decide(client, "INC-NOPE", "APR-x", action)
    assert response.status_code == 404 and "INC-NOPE" in response.json()["detail"]


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_deciding_on_an_unknown_approval_is_404(client, action):
    incident_id, _ = pending_one(client)
    response = decide(client, incident_id, "APR-nope", action)
    assert response.status_code == 404 and "APR-nope" in response.json()["detail"]


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_an_approval_id_cannot_be_used_through_another_incident(client, action):
    mine, approval_id = pending_one(client)
    other, _ = pending_one(client)

    response = decide(client, other, approval_id, action)

    assert response.status_code == 404
    # ...and it is untouched.
    listed = client.get(approvals_url(mine)).json()["approvals"]
    assert listed[0]["status"] == "pending"


# --- conflicts -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [("approve", "approve"), ("approve", "reject"), ("reject", "approve"), ("reject", "reject")],
)
def test_a_second_decision_is_409_and_the_first_stands(client, first, second):
    incident_id, approval_id = pending_one(client)
    won = decide(client, incident_id, approval_id, first, reason="first").json()

    response = decide(client, incident_id, approval_id, second, reason="second")

    assert response.status_code == 409
    assert "already" in response.json()["detail"]
    (kept,) = client.get(approvals_url(incident_id)).json()["approvals"]
    assert kept["status"] == won["status"] and kept["decision_reason"] == "first"


def test_a_decision_on_a_closed_incident_is_409(client, pipeline):
    """A step left pending when its incident resolves can no longer be decided."""
    pipeline.actions = [HIGH_ROLLBACK, HIGH_FAILOVER]
    incident_id = new_incident_id()
    first, second = submit(client, incident_id)["pending_approval_ids"]
    # pending_approval_ids follow plan order: first is the rollback, second the failover.
    assert decide(client, incident_id, first, "approve").status_code == 200

    # Executing the approved rollback recovers the service, which resolves the incident
    # and leaves the failover approval pending on a closed incident.
    assert submit(client, incident_id, post="healthy")["outcome"] == "resolve"
    assert client.get(f"/incidents/{incident_id}").json()["status"] == "resolved"

    response = decide(client, incident_id, second, "approve")

    assert response.status_code == 409 and "closed" in response.json()["detail"]
    leftover = {
        a["approval_id"]: a for a in client.get(approvals_url(incident_id)).json()["approvals"]
    }
    assert leftover[second]["status"] == "pending" and leftover[second]["decidable"] is False


def test_an_approval_for_a_replaced_plan_is_stale_and_cannot_be_decided(client, pipeline):
    incident_id, old_id = pending_one(client)
    pipeline.actions = [HIGH_FAILOVER]
    new_id = submit(client, incident_id)["pending_approval_ids"][0]

    approvals = {
        a["approval_id"]: a for a in client.get(approvals_url(incident_id)).json()["approvals"]
    }
    assert approvals[old_id]["stale"] is True and approvals[old_id]["decidable"] is False
    assert approvals[new_id]["stale"] is False and approvals[new_id]["decidable"] is True

    response = decide(client, incident_id, old_id, "approve")
    assert response.status_code == 409 and "superseded" in response.json()["detail"]
    assert decide(client, incident_id, new_id, "approve").status_code == 200


# --- invalid requests ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},  # no approver
        {"approver": ""},
        {"approver": "   "},
        {"approver": "x" * 129},
        {"approver": "alice", "reason": "y" * 1001},
        {"approver": "alice", "unexpected": True},
        {"approver": 7},
    ],
)
@pytest.mark.parametrize("action", ["approve", "reject"])
def test_malformed_decision_payloads_are_422_and_change_nothing(client, body, action):
    incident_id, approval_id = pending_one(client)

    response = client.post(approvals_url(incident_id, approval_id, action), json=body)

    assert response.status_code == 422
    assert client.get(approvals_url(incident_id)).json()["approvals"][0]["status"] == "pending"


def test_the_longest_acceptable_payload_is_accepted(client):
    incident_id, approval_id = pending_one(client)
    response = decide(
        client, incident_id, approval_id, "approve", approver="x" * 128, reason="y" * 1000
    )
    assert response.status_code == 200


# --- storage failures ---------------------------------------------------------------------------------------------


class _Down(InMemoryIncidentRepository):
    """Every approval operation fails with a driver-looking error."""

    def _fail(self, *args, **kwargs):
        raise RuntimeError("password=hunter2 connection refused")

    list_approvals = decide_approval = get_incident = _fail


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/incidents/INC-1/approvals", None),
        ("post", "/incidents/INC-1/approvals/APR-1/approve", {"approver": "alice"}),
        ("post", "/incidents/INC-1/approvals/APR-1/reject", {"approver": "alice"}),
    ],
)
def test_a_storage_failure_is_503_and_leaks_nothing(method, path, body):
    client = TestClient(create_app(repository=_Down()))

    response = client.request(method, path, json=body)

    assert response.status_code == 503
    assert "hunter2" not in response.text and "refused" not in response.text


def test_the_decision_stands_even_if_the_follow_up_read_fails(monkeypatch, pipeline):
    store = InMemoryIncidentRepository()
    client = TestClient(create_app(repository=store))
    incident_id, approval_id = pending_one(client)
    monkeypatch.setattr(
        store, "get_incident", lambda _id: (_ for _ in ()).throw(RuntimeError("down"))
    )

    response = decide(client, incident_id, approval_id, "approve")

    assert response.status_code == 200  # the committed decision is not turned into an error
    body = response.json()
    assert body["status"] == "approved"
    assert body["stale"] is None and body["decidable"] is None  # hints unknown, not guessed
    monkeypatch.undo()
    assert store.get_approval(incident_id, approval_id).status == ApprovalState.APPROVED


def test_the_openapi_description_warns_that_identity_is_not_authenticated(pipeline):
    spec = (
        TestClient(create_app(repository=InMemoryIncidentRepository())).get("/openapi.json").json()
    )
    for action in ("approve", "reject"):
        operation = spec["paths"][f"/incidents/{{incident_id}}/approvals/{{approval_id}}/{action}"][
            "post"
        ]
        assert "NOT authenticated" in operation["description"]
