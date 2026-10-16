"""Long-term incident memory: past analyses persisted across runs.

The Phase 6 knowledge base was static, and that phase's own note named the
condition for revisiting it: a corpus that changes at runtime and needs
persistence. This is that. Still no vector database - a JSON-lines file holds
the entries and retrieval stays brute-force cosine similarity, because the
volume is still tiny. Revisit when the file is large enough that loading and
embedding it per process is noticeable.

Entries written here are ``source="learned"``: model-authored, and not yet
confirmed by a human. They are retrievable, but never indistinguishable from
the curated corpus - otherwise the system would cite its own earlier guesses
as established precedent.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

from agentic_ai.domain.knowledge import KnowledgeEntry

MEMORY_PATH_ENV = "AGENTIC_MEMORY_PATH"
DEFAULT_MEMORY_PATH = ".agentic_memory/incidents.jsonl"


def memory_path() -> Path:
    return Path(os.environ.get(MEMORY_PATH_ENV, DEFAULT_MEMORY_PATH))


def load_memory() -> list[KnowledgeEntry]:
    """Read every remembered entry. A missing file is simply empty memory.

    A corrupt line is skipped rather than raised: one bad record must not
    take down retrieval for everything else.
    """
    path = memory_path()
    if not path.exists():
        return []

    entries: list[KnowledgeEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(KnowledgeEntry.model_validate_json(line))
        except ValidationError:
            continue
    return entries


def remember(entry: KnowledgeEntry) -> bool:
    """Append an entry. Returns False if an equivalent one already exists."""
    if entry.source != "learned":
        raise ValueError("Only learned entries may be written to memory.")

    for existing in load_memory():
        if (
            existing.symptom_description == entry.symptom_description
            and existing.root_cause == entry.root_cause
        ):
            return False

    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")
    return True
