"""The search_knowledge_base tool: the agent's access to past incidents.

Wraps knowledge_retrieval.search() into the same shape as the mock
observability tools - a JSON string, for the same reasons as those tools:
preservable verbatim as Evidence.raw_output, directly readable as LLM
tool-result content, and deterministic to assert against in tests.

Unlike the mock tools, this one has no ``scenario`` parameter - retrieval
runs against the one real (synthetic) knowledge base regardless of which
mock incident scenario is active, since the knowledge base represents
institutional knowledge that exists independently of any specific incident.
"""

import json

from agentic_ai.tools.knowledge_retrieval import search

DEFAULT_TOP_K = 3


def search_knowledge_base(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    Search past-incident runbooks for entries similar to a description.

    Use this when the symptoms resemble a problem that may have occurred
    before. Returns the most similar past incidents, each with its
    previously-identified root cause and resolution - treat these as
    informed hypotheses to confirm against the actual evidence, not as a
    confirmed diagnosis for the current incident.

    Args:
        query: A description of the current symptoms, written the way an
            engineer would describe the problem (not a guess at the cause).
        top_k: How many results to return. Defaults to 3.

    Returns:
        A JSON-formatted string listing matching runbook entries ordered by
        similarity, most similar first. An empty query returns no results.
    """
    if not query or not query.strip():
        return json.dumps(
            {
                "status": "error",
                "error_type": "empty_query",
                "message": "query must not be empty.",
            }
        )

    results = search(query, top_k=top_k)

    return json.dumps(
        {
            "status": "success",
            "query": query,
            "results": [
                {
                    "id": result.entry.id,
                    "title": result.entry.title,
                    "symptom_description": result.entry.symptom_description,
                    "root_cause": result.entry.root_cause,
                    "resolution": result.entry.resolution,
                    "tags": result.entry.tags,
                    "similarity": round(result.similarity, 3),
                }
                for result in results
            ],
        }
    )
