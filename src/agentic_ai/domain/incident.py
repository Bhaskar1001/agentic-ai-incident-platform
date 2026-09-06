from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    MITIGATION = "mitigation"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Incident(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    severity: Severity

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    _status: IncidentStatus = PrivateAttr(default=IncidentStatus.DETECTED)

    @model_validator(mode="before")
    @classmethod
    def extract_status(cls, values):
        if isinstance(values, dict) and "status" in values:
            status = values["status"]
            values = values.copy()
            values.pop("status")
            values["_status"] = status

        return values

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Incident title cannot be empty")

        if len(value) > 200:
            raise ValueError("Incident title cannot exceed 200 characters")

        return value

    @property
    def status(self) -> IncidentStatus:
        return self._status

    _LEGAL_TRANSITIONS = {
        IncidentStatus.DETECTED: {
            IncidentStatus.INVESTIGATING,
        },
        IncidentStatus.INVESTIGATING: {
            IncidentStatus.MITIGATION,
        },
        IncidentStatus.MITIGATION: {
            IncidentStatus.RESOLVED,
        },
        IncidentStatus.RESOLVED: {
            IncidentStatus.CLOSED,
            IncidentStatus.INVESTIGATING,
        },
        IncidentStatus.CLOSED: set(),
    }

    def transition_to(self, new_status: IncidentStatus) -> None:
        allowed_statuses = self._LEGAL_TRANSITIONS.get(self.status)

        if allowed_statuses is None:
            raise ValueError(
                f"No transition rules defined for status: {self.status.value}"
            )

        if new_status not in allowed_statuses:
            raise ValueError(
                f"Illegal incident status transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self._status = new_status
        self.updated_at = datetime.now(timezone.utc)