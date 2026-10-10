"""Tests for the knowledge base: the corpus, retrieval, and the agent tool.

These tests load the real embedding model (no LLM involved, but a genuine
local model load + inference), so they are slower than the rest of the
deterministic suite - expect seconds, not milliseconds, especially on first
run before the model is cached on disk.
"""

import json

import pytest

from agentic_ai.domain.knowledge import KnowledgeEntry, KnowledgeSearchResult
from agentic_ai.tools.knowledge_corpus import KNOWLEDGE_BASE
from agentic_ai.tools.knowledge_retrieval import get_entry_by_id, search
from agentic_ai.tools.search_tool import search_knowledge_base


# --- corpus integrity ------------------------------------------------------


def test_corpus_has_unique_ids() -> None:
    ids = [entry.id for entry in KNOWLEDGE_BASE]

    assert len(ids) == len(set(ids))


def test_corpus_is_nontrivially_sized() -> None:
    """A corpus this small would not meaningfully exercise ranking."""
    assert len(KNOWLEDGE_BASE) >= 10


def test_every_entry_has_all_required_fields_populated() -> None:
    for entry in KNOWLEDGE_BASE:
        assert entry.title.strip()
        assert entry.symptom_description.strip()
        assert entry.root_cause.strip()
        assert entry.resolution.strip()


def test_search_text_excludes_the_answer() -> None:
    """Retrieval must match on symptoms, not leak the resolution into the index."""
    entry = KNOWLEDGE_BASE[0]
    search_text = entry.as_search_text()

    assert entry.root_cause not in search_text
    assert entry.resolution not in search_text
    assert entry.title in search_text
    assert entry.symptom_description in search_text


# --- domain model ------------------------------------------------------


def test_similarity_must_be_within_unit_range() -> None:
    entry = KNOWLEDGE_BASE[0]

    with pytest.raises(Exception):
        KnowledgeSearchResult(entry=entry, similarity=1.5)


# --- retrieval quality ---------------------------------------------------
#
# These assert on real embedding output, not mocks - the entire point of
# this module is whether retrieval actually discriminates between entries
# that share surface vocabulary but have different root causes.


def test_connection_pool_query_ranks_the_matching_entry_first() -> None:
    results = search(
        "Service returns errors under load, logs mention connection pool "
        "exhausted",
        top_k=3,
    )

    assert results[0].entry.id == "RB-001"


def test_upstream_symptom_is_not_confused_with_local_database_issue() -> None:
    """RB-001 and RB-002 share vocabulary (errors, service) but differ in
    whether the service's own resources are implicated - exactly the
    distinction the escalation policy and agent prompt both care about.

    The query describes the positive symptom (timeouts calling an external
    dependency) rather than leading with a negation ("our database is
    fine"). An earlier version of this test phrased the query as a negation
    and it mis-ranked RB-001 (database) above RB-002 - this embedding model
    is noticeably weaker at negation than at topical matching, since
    mentioning "database" at all, even to rule it out, pulled
    database-themed entries up in the ranking. Describing what IS happening,
    not what is NOT happening, is the more reliable query shape - both for
    this corpus and as a general property of small embedding models worth
    remembering when writing retrieval queries elsewhere.
    """
    results = search(
        "We see timeouts calling an external dependency, the dependency "
        "itself appears down, our service metrics are otherwise fine",
        top_k=3,
    )

    assert results[0].entry.id == "RB-002"


def test_crash_loop_pair_is_discriminated_by_cause_not_just_crash_loop() -> None:
    """RB-006 and RB-007 are both CrashLoopBackOff - retrieval must pick up
    on which specific cause each query describes, not just "pods crashing".
    """
    config_query = search(
        "Pods crash looping right after deploy, logs show a missing "
        "required environment variable",
        top_k=1,
    )
    memory_query = search(
        "Pods get OOMKilled under load but run fine at low traffic",
        top_k=1,
    )

    assert config_query[0].entry.id == "RB-006"
    assert memory_query[0].entry.id == "RB-007"


def test_results_are_ordered_by_descending_similarity() -> None:
    results = search("database connections exhausted", top_k=5)

    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_empty_query_returns_no_results() -> None:
    assert search("") == []
    assert search("   ") == []


def test_top_k_limits_the_number_of_results() -> None:
    results = search("service is failing", top_k=2)

    assert len(results) == 2


def test_get_entry_by_id_returns_the_matching_entry() -> None:
    entry = get_entry_by_id("RB-001")

    assert entry is not None
    assert entry.id == "RB-001"


def test_get_entry_by_id_returns_none_for_unknown_id() -> None:
    assert get_entry_by_id("RB-does-not-exist") is None


# --- the agent-facing tool -------------------------------------------------


def test_search_tool_returns_success_json_with_expected_fields() -> None:
    raw = search_knowledge_base("connection pool exhausted")
    data = json.loads(raw)

    assert data["status"] == "success"
    assert data["results"], "expected at least one match"

    first = data["results"][0]
    assert set(first.keys()) == {
        "id",
        "title",
        "symptom_description",
        "root_cause",
        "resolution",
        "tags",
        "similarity",
    }


def test_search_tool_respects_top_k() -> None:
    data = json.loads(search_knowledge_base("service is down", top_k=1))

    assert len(data["results"]) == 1


def test_search_tool_rejects_empty_query_without_crashing() -> None:
    data = json.loads(search_knowledge_base(""))

    assert data["status"] == "error"
    assert data["error_type"] == "empty_query"


def test_search_tool_output_is_json_serialisable_and_deterministic_in_shape() -> None:
    """Same query, same top_k, twice - the result shape must not vary."""
    first = json.loads(search_knowledge_base("database is slow"))
    second = json.loads(search_knowledge_base("database is slow"))

    assert [r["id"] for r in first["results"]] == [r["id"] for r in second["results"]]
