"""Domain model for root cause analysis.

RCA is a synthesis step over a *finished* investigation, not another
evidence-gathering loop: it takes the Findings an investigation already
produced and reasons about which one (or combination) actually explains the
incident, what else could explain it, and whether any knowledge-base
precedent agrees. It never calls a tool and never re-observes anything -
investigation gathers; RCA synthesizes. Keeping those as different jobs is
the same reasoning that kept Evidence and Finding themselves separate.

Free of LLM and framework imports: this describes what an RCA conclusion is,
not how one gets produced.
"""

from enum import Enum

from pydantic import BaseModel, Field
from uuid import UUID


class KnowledgeBaseConsistency(str, Enum):
    """How a retrieved past-incident precedent relates to this conclusion.

    A separate enum rather than folding this into prose, for the same reason
    termination_reason is its own field on InvestigationResult: whether a
    precedent was even consulted, and whether it agreed, are facts a reader
    (or an evaluation harness) needs to check without parsing a paragraph.
    """

    SUPPORTED = "supported"
    """A retrieved precedent's root cause matches this conclusion."""

    CONTRADICTED = "contradicted"
    """A retrieved precedent exists but points to a different root cause -
    worth surfacing explicitly rather than silently picking one account."""

    NO_RELEVANT_PRECEDENT = "no_relevant_precedent"
    """Either the knowledge base was not consulted, or nothing it returned
    was relevant enough to bear on this conclusion."""


class AlternativeHypothesis(BaseModel):
    """A root cause that was considered and ranked below the primary one.

    Recording *why* it was ranked lower is what makes the conclusion
    auditable rather than a single confident-sounding sentence with no
    visible reasoning behind it - the same instinct behind
    unresolved_questions on InvestigationResult.
    """

    statement: str
    why_ranked_lower: str


class RootCauseAnalysis(BaseModel):
    """The synthesized conclusion of a root cause analysis."""

    root_cause: str
    root_cause_confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the *synthesized* conclusion, not any single finding.
    Named distinctly from Finding.model_assessed_confidence and
    InvestigationResult.model_assessed_confidence - this is a judgement about
    the combination of findings, one level up from either."""

    supporting_finding_ids: list[UUID] = Field(default_factory=list)
    """Which Findings from the investigation this conclusion actually rests
    on. Validated against the real Finding ids present in the investigation
    before being trusted - an unresolvable id is treated as invalid
    structured output, the same way an invalid enum value is, not quietly
    accepted."""

    alternative_hypotheses: list[AlternativeHypothesis] = Field(
        default_factory=list
    )

    knowledge_base_consistency: KnowledgeBaseConsistency
    knowledge_base_note: str = ""
    """Short explanation for knowledge_base_consistency. Empty when
    NO_RELEVANT_PRECEDENT with nothing further to say."""
