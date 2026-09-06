
import ollama
from pydantic import BaseModel, ValidationError

from agentic_ai.domain.incident import Severity


class SeverityAssessmentError(Exception):
    """Base exception for severity assessment failures."""


class InvalidAssessmentError(SeverityAssessmentError):
    """Raised when the LLM repeatedly returns invalid structured output."""


class SeverityAssessment(BaseModel):
    severity: Severity
    reasoning: str


def assess_severity(incident_description: str) -> SeverityAssessment:
    schema = SeverityAssessment.model_json_schema()

    for attempt in range(2):
        try:
            response = ollama.chat(
                model="llama3.2",
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
                format=schema,
            )

            raw_content = response.message.content

            return SeverityAssessment.model_validate_json(raw_content)

        except ValidationError as exc:
            if attempt == 1:
                raise InvalidAssessmentError(
                    "LLM returned invalid severity assessment after retry."
                ) from exc

