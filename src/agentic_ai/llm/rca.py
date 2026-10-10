"""Root cause analysis: a single structured-output synthesis over a finished
investigation.

This is deliberately not another agent loop. It takes an InvestigationResult
and reasons over what has already been gathered - no tool calls, no further
observation. That boundary is enforced structurally, the same way
InvestigationResult's own summariser cannot author evidence or
termination_reason: the prompt built here never offers tools, so there is
nothing for the model to call even if it wanted to.
"""

import json

from pydantic import ValidationError

from agentic_ai.domain.investigation import InvestigationResult
from agentic_ai.domain.rca import RootCauseAnalysis
from agentic_ai.llm.client import LLMClient, get_client


class RootCauseAnalysisError(Exception):
    """Base exception for RCA failures."""


class InvalidRootCauseAnalysisError(RootCauseAnalysisError):
    """The model repeatedly failed to produce a valid, well-cited analysis."""


def _findings_section(result: InvestigationResult) -> str:
    """Render findings with their ids, so the model can cite them exactly."""
    if not result.findings:
        return "No findings were recorded."

    lines = [
        f"- id={finding.id}: {finding.statement} "
        f"(observation confidence {finding.model_assessed_confidence})"
        for finding in result.findings
    ]
    return "\n".join(lines)


def _knowledge_base_section(result: InvestigationResult) -> str:
    """Render any search_knowledge_base results found in the evidence.

    search_knowledge_base evidence is Evidence like any other tool's - this
    filters by source_tool and parses its own raw_output, rather than the
    domain model needing a dedicated field for one specific tool's results.
    """
    kb_evidence = [
        item for item in result.evidence if item.source_tool == "search_knowledge_base"
    ]
    if not kb_evidence:
        return "No knowledge base search was performed during the investigation."

    lines: list[str] = []
    for item in kb_evidence:
        try:
            payload = json.loads(item.raw_output)
        except json.JSONDecodeError:
            continue

        for entry in payload.get("results", []):
            lines.append(
                f"- {entry['id']} ({entry['title']}, similarity "
                f"{entry['similarity']}): root cause was \"{entry['root_cause']}\""
            )

    return "\n".join(lines) if lines else "No knowledge base matches were found."


_RCA_INSTRUCTION = (
    "You are performing root cause analysis for an already-completed "
    "incident investigation. You are not investigating further and have no "
    "tools - reason only over the findings and evidence already gathered "
    "below.\n\n"
    "INVESTIGATION SUMMARY:\n{summary}\n\n"
    "FINDINGS (cite by id in supporting_finding_ids):\n{findings}\n\n"
    "KNOWLEDGE BASE RESULTS FROM THIS INVESTIGATION:\n{knowledge_base}\n\n"
    "Produce a root cause analysis as JSON:\n"
    "- root_cause: the single most likely explanation, synthesized from the "
    "findings above - not a restatement of one finding, but your judgement "
    "of what they collectively show.\n"
    "- root_cause_confidence: 0.0 to 1.0, your confidence in this synthesis "
    "as a whole.\n"
    "- supporting_finding_ids: the exact id values (copy them verbatim) of "
    "the findings above that support your root cause. Cite only ids that "
    "appear above.\n"
    "- alternative_hypotheses: other plausible explanations you considered "
    "and why the evidence ranks them below your primary conclusion. Empty "
    "list if none are plausible.\n"
    "- knowledge_base_consistency: \"supported\" if a knowledge base result "
    "above names the same root cause, \"contradicted\" if a result exists "
    "but names a different root cause, \"no_relevant_precedent\" if no "
    "knowledge base search was performed or nothing returned was relevant.\n"
    "- knowledge_base_note: one sentence explaining that consistency "
    "judgement. Empty string if no_relevant_precedent with nothing further "
    "to add."
)


def analyze_root_cause(
    result: InvestigationResult, *, client: LLMClient | None = None
) -> RootCauseAnalysis:
    """Synthesize a root cause analysis from a completed investigation."""
    client = client or get_client()
    schema = RootCauseAnalysis.model_json_schema()

    prompt = _RCA_INSTRUCTION.format(
        summary=result.summary,
        findings=_findings_section(result),
        knowledge_base=_knowledge_base_section(result),
    )

    valid_finding_ids = {str(finding.id) for finding in result.findings}

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.complete(
                messages=[{"role": "user", "content": prompt}],
                json_schema=schema,
            )
            analysis = RootCauseAnalysis.model_validate_json(response.content)

            cited = {str(finding_id) for finding_id in analysis.supporting_finding_ids}
            unresolvable = cited - valid_finding_ids
            if unresolvable:
                raise ValueError(
                    f"RCA cited finding id(s) not present in the "
                    f"investigation: {sorted(unresolvable)}"
                )

            return analysis

        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == 1:
                raise InvalidRootCauseAnalysisError(
                    "LLM produced an invalid or badly-cited root cause "
                    "analysis after retry."
                ) from last_error
