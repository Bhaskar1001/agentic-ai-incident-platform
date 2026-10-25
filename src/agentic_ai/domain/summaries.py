"""Small, framework-free summaries of a finished workflow run.

These describe *what happened* (the terminal diagnostic decision and the
post-remediation verification outcome). They live in the domain so persistence
can store them without depending on the API layer; ``api.schemas`` re-exports
them under the same names so the HTTP contract is unchanged.
"""

from pydantic import BaseModel, ConfigDict


class DiagnosticSummary(BaseModel):
    """High-level summary of the workflow's terminal diagnostic decision."""

    decision: str
    trigger_type: str
    primary_reason: str

    model_config = ConfigDict(frozen=True)


class VerificationSummary(BaseModel):
    """High-level summary of the post-remediation verification results."""

    status: str
    attempts: int
    reasons: list[str]

    model_config = ConfigDict(frozen=True)


__all__ = ["DiagnosticSummary", "VerificationSummary"]
