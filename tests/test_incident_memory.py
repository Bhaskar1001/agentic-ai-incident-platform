import json

import pytest

from agentic_ai.domain.investigation import (
    Evidence,
    InvestigationResult,
    TerminationReason,
)
from agentic_ai.domain.knowledge import KnowledgeEntry
from agentic_ai.llm.rca import _knowledge_base_section
from agentic_ai.tools.incident_memory import load_memory, memory_path, remember
from agentic_ai.tools.knowledge_retrieval import get_entry_by_id, search
from agentic_ai.tools.search_tool import search_knowledge_base

_SYMPTOM = "Zebra cache stampede after nightly flush."


def _learned(entry_id: str = "MEM-1") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        title=_SYMPTOM[:40],
        symptom_description=_SYMPTOM,
        root_cause="Cache warmup missing.",
        resolution="Proposed (unexecuted): clear_cache on payment",
        source="learned",
    )


def test_missing_memory_file_is_empty_memory() -> None:
    assert load_memory() == []


def test_remembered_entry_round_trips() -> None:
    assert remember(_learned()) is True
    assert load_memory() == [_learned()]


def test_duplicate_is_not_stored_twice() -> None:
    remember(_learned("MEM-1"))
    assert remember(_learned("MEM-2")) is False
    assert len(load_memory()) == 1


def test_curated_entries_cannot_be_written_to_memory() -> None:
    with pytest.raises(ValueError):
        remember(_learned().model_copy(update={"source": "curated"}))


def test_corrupt_line_is_skipped_not_fatal() -> None:
    remember(_learned())
    with memory_path().open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    assert len(load_memory()) == 1


def test_remembered_entry_is_retrievable_and_marked_learned() -> None:
    remember(_learned())

    assert search(_SYMPTOM, top_k=1)[0].entry.id == "MEM-1"
    assert get_entry_by_id("MEM-1") is not None
    payload = json.loads(search_knowledge_base(_SYMPTOM))
    assert payload["results"][0]["source"] == "learned"


def test_curated_entries_are_marked_curated() -> None:
    payload = json.loads(search_knowledge_base("database connection pool exhausted"))
    assert payload["results"][0]["source"] == "curated"


def test_rca_prompt_marks_learned_results_unverified() -> None:
    evidence = Evidence(
        source_tool="search_knowledge_base",
        raw_output=json.dumps(
            {
                "results": [
                    {
                        "id": "MEM-1",
                        "title": "t",
                        "similarity": 0.9,
                        "root_cause": "r",
                        "source": "learned",
                    }
                ]
            }
        ),
        observation="x",
    )
    result = InvestigationResult(
        summary="s",
        evidence=[evidence],
        model_assessed_confidence=0.5,
        recommended_next_action="n",
        termination_reason=TerminationReason.COMPLETED,
    )

    assert "UNVERIFIED" in _knowledge_base_section(result)
