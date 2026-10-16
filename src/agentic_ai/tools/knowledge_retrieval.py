"""Retrieval over the synthetic engineering knowledge base.

In-memory, embedded once at import time, searched with brute-force cosine
similarity. Deliberately not a vector database: the corpus is ~20 static
documents, and brute-force search over that many vectors costs microseconds.
A real vector store would be justified by a concrete problem this does not
have yet - a corpus too large to hold in memory, or one that changes at
runtime and needs persistence. Revisit if either becomes true.

The embedding model is loaded lazily (on first search, not at import time) so
that importing this module - which test collection and every other import in
the package does - never pays the model-load cost unless retrieval is
actually used.
"""

from __future__ import annotations

import numpy as np

from agentic_ai.domain.knowledge import KnowledgeEntry, KnowledgeSearchResult
from agentic_ai.tools.incident_memory import load_memory
from agentic_ai.tools.knowledge_corpus import KNOWLEDGE_BASE

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_embedding_cache: dict[str, np.ndarray] = {}


def _get_model():
    """Load the embedding model once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _all_entries() -> list[KnowledgeEntry]:
    """The curated corpus plus whatever long-term memory holds right now.

    Memory is re-read on every search because it changes at runtime; ids are
    unique, so a learned entry can never shadow a curated one.
    """
    curated_ids = {e.id for e in KNOWLEDGE_BASE}
    learned = [e for e in load_memory() if e.id not in curated_ids]
    return [*KNOWLEDGE_BASE, *learned]


def _embeddings_for(entries: list[KnowledgeEntry]) -> np.ndarray:
    """Embed each entry once, keyed by id plus text so an edit re-embeds."""
    missing = [
        e for e in entries if _cache_key(e) not in _embedding_cache
    ]
    if missing:
        vectors = _get_model().encode(
            [e.as_search_text() for e in missing], normalize_embeddings=True
        )
        for entry, vector in zip(missing, vectors):
            _embedding_cache[_cache_key(entry)] = vector
    return np.stack([_embedding_cache[_cache_key(e)] for e in entries])


def _cache_key(entry: KnowledgeEntry) -> str:
    return f"{entry.id}:{entry.as_search_text()}"


def search(query: str, *, top_k: int = 3) -> list[KnowledgeSearchResult]:
    """Return the top_k most similar knowledge entries to ``query``.

    Cosine similarity, computed as a plain dot product since both the query
    and entry embeddings are L2-normalised at encode time.
    """
    if not query.strip():
        return []

    entries = _all_entries()
    embeddings = _embeddings_for(entries)

    query_embedding = _get_model().encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_embedding

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        KnowledgeSearchResult(
            entry=entries[i], similarity=float(max(0.0, min(1.0, similarities[i])))
        )
        for i in ranked_indices
    ]


def get_entry_by_id(entry_id: str) -> KnowledgeEntry | None:
    """Look up a single entry by id, for tests and debugging."""
    return next((e for e in _all_entries() if e.id == entry_id), None)
