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
from agentic_ai.tools.knowledge_corpus import KNOWLEDGE_BASE

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_corpus_embeddings: np.ndarray | None = None


def _get_model():
    """Load the embedding model once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_corpus_embeddings() -> np.ndarray:
    """Embed the corpus once, cached for the process lifetime.

    The corpus is a static Python list, not loaded from a file or database,
    so there is no staleness concern here: a code change to the corpus
    requires a process restart anyway, which naturally re-embeds it.
    """
    global _corpus_embeddings
    if _corpus_embeddings is None:
        model = _get_model()
        texts = [entry.as_search_text() for entry in KNOWLEDGE_BASE]
        _corpus_embeddings = model.encode(texts, normalize_embeddings=True)
    return _corpus_embeddings


def search(query: str, *, top_k: int = 3) -> list[KnowledgeSearchResult]:
    """Return the top_k most similar knowledge entries to ``query``.

    Cosine similarity, computed as a plain dot product since both the query
    and corpus embeddings are L2-normalised at encode time.
    """
    if not query.strip():
        return []

    model = _get_model()
    corpus_embeddings = _get_corpus_embeddings()

    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = corpus_embeddings @ query_embedding

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        KnowledgeSearchResult(
            entry=KNOWLEDGE_BASE[i], similarity=float(similarities[i])
        )
        for i in ranked_indices
    ]


def get_entry_by_id(entry_id: str) -> KnowledgeEntry | None:
    """Look up a single entry by id, for tests and debugging."""
    return next((e for e in KNOWLEDGE_BASE if e.id == entry_id), None)
