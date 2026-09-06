from datetime import datetime, timezone
from time import sleep

import pytest
from pydantic import ValidationError

from agentic_ai.domain.incident import (
    Incident,
    IncidentStatus,
    Severity,
)


def test_incident_creation():
    incident = Incident(
        title="API latency incident",
        severity=Severity.HIGH,
    )

    assert incident.id is not None
    assert incident.title == "API latency incident"
    assert incident.severity == Severity.HIGH
    assert incident.status == IncidentStatus.DETECTED
    assert incident.created_at.tzinfo is not None
    assert incident.updated_at.tzinfo is not None


def test_empty_title_is_rejected():
    with pytest.raises(ValidationError):
        Incident(
            title="",
            severity=Severity.HIGH,
        )


def test_whitespace_title_is_rejected():
    with pytest.raises(ValidationError):
        Incident(
            title="   ",
            severity=Severity.HIGH,
        )


def test_title_is_trimmed():
    incident = Incident(
        title="  API failure  ",
        severity=Severity.MEDIUM,
    )

    assert incident.title == "API failure"


def test_title_over_200_characters_is_rejected():
    with pytest.raises(ValidationError):
        Incident(
            title="A" * 201,
            severity=Severity.LOW,
        )


def test_illegal_transition_from_detected_to_closed():
    incident = Incident(
        title="Database failure",
        severity=Severity.CRITICAL,
    )

    original_updated_at = incident.updated_at

    with pytest.raises(ValueError):
        incident.transition_to(IncidentStatus.CLOSED)

    assert incident.status == IncidentStatus.DETECTED
    assert incident.updated_at == original_updated_at


def test_legal_transition_updates_status_and_timestamp():
    incident = Incident(
        title="API failure",
        severity=Severity.HIGH,
    )

    original_updated_at = incident.updated_at

    sleep(0.001)

    incident.transition_to(IncidentStatus.INVESTIGATING)

    assert incident.status == IncidentStatus.INVESTIGATING
    assert incident.updated_at > original_updated_at


def test_resolved_incident_can_be_reopened():
    incident = Incident(
        title="Service outage",
        severity=Severity.CRITICAL,
        status=IncidentStatus.RESOLVED,
    )

    original_updated_at = incident.updated_at

    sleep(0.001)

    incident.transition_to(IncidentStatus.INVESTIGATING)

    assert incident.status == IncidentStatus.INVESTIGATING
    assert incident.updated_at > original_updated_at


def test_complete_legal_lifecycle():
    incident = Incident(
        title="Production incident",
        severity=Severity.HIGH,
    )

    incident.transition_to(IncidentStatus.INVESTIGATING)
    assert incident.status == IncidentStatus.INVESTIGATING

    incident.transition_to(IncidentStatus.MITIGATION)
    assert incident.status == IncidentStatus.MITIGATION

    incident.transition_to(IncidentStatus.RESOLVED)
    assert incident.status == IncidentStatus.RESOLVED

    incident.transition_to(IncidentStatus.CLOSED)
    assert incident.status == IncidentStatus.CLOSED

def test_status_cannot_be_changed_directly():
    incident = Incident(
        title="API failure",
        severity=Severity.HIGH,
    )

    with pytest.raises(AttributeError):
        incident.status = IncidentStatus.CLOSED

    assert incident.status == IncidentStatus.DETECTED


def test_private_status_can_still_be_modified_directly():
    incident = Incident(
        title="API failure",
        severity=Severity.HIGH,
    )

    incident._status = IncidentStatus.CLOSED

    assert incident.status == IncidentStatus.CLOSED