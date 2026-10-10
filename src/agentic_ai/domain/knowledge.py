"""Domain model for the engineering knowledge base.

A ``KnowledgeEntry`` is a past incident, deliberately structured rather than a
freeform paragraph, so retrieval has stable fields to search over and the
agent has a consistent shape to cite from. This mirrors the same discipline
as ``investigation.py``: observation (symptom), inference (root cause), and
action (resolution) stay in distinct fields rather than one undifferentiated
block of prose - the same reasoning that kept Evidence and Finding separate.

Free of LLM, embedding, and framework imports: this describes *what a
knowledge entry is*, not how it gets embedded or searched.
"""

from pydantic import BaseModel, Field


class KnowledgeEntry(BaseModel):
    """One past incident recorded in the knowledge base.

    ``symptom_description`` is what retrieval actually matches against - it
    should read the way an engineer would describe the problem, not the way
    the root cause would be described, since that is what an incoming query
    will resemble.
    """

    id: str
    title: str
    symptom_description: str
    root_cause: str
    resolution: str
    tags: list[str] = Field(default_factory=list)

    def as_search_text(self) -> str:
        """The text actually embedded and matched against.

        Title and symptom only - not root_cause or resolution. Retrieval
        should find entries whose *symptoms* resemble the query; if the
        answer text itself were embedded, a query that happened to share
        vocabulary with a resolution (e.g. "restart the pod") could outrank
        an entry with a genuinely similar symptom but a differently-worded
        fix.
        """
        return f"{self.title}. {self.symptom_description}"


class KnowledgeSearchResult(BaseModel):
    """One retrieval hit: the entry plus how well it matched."""

    entry: KnowledgeEntry
    similarity: float = Field(ge=0.0, le=1.0)
