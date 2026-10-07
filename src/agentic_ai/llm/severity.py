from pydantic import BaseModel, ValidationError

from agentic_ai.domain.incident import Severity
from agentic_ai.llm.client import LLMClient, get_client


class SeverityAssessmentError(Exception):
    """Base exception for severity assessment failures."""


class InvalidAssessmentError(SeverityAssessmentError):
    """Raised when the LLM repeatedly returns invalid structured output."""


class SeverityAssessment(BaseModel):
    severity: Severity
    reasoning: str


def assess_severity(
    incident_description: str, *, client: LLMClient | None = None
) -> SeverityAssessment:
    client = client or get_client()
    schema = SeverityAssessment.model_json_schema()

    for attempt in range(2):
        try:
            response = client.complete(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze the following incident and determine its "
                            "severity. Return the severity and reasoning.\n\n"
                            f"Incident: {incident_description}"
                        ),
                    }
                ],
                json_schema=schema,
            )

            return SeverityAssessment.model_validate_json(response.content)

        except ValidationError as exc:
            if attempt == 1:
                raise InvalidAssessmentError(
                    "LLM returned invalid severity assessment after retry."
                ) from exc
