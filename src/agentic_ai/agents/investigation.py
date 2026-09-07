"""Investigation agent: a bounded ReAct-style LangGraph subgraph.

The loop is ``reason -> select tool -> execute -> observe``, repeated until the
model stops requesting tools or the step budget is exhausted.

Division of responsibility:
    * The LLM decides which tool to call and when it has enough evidence.
    * Deterministic code enforces the step budget, validates tool names,
      records the timeline, and preserves raw tool output verbatim.
"""

import json
from typing import Annotated, Any, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agentic_ai.domain.investigation import (
    Evidence,
    InvestigationResult,
    TerminationReason,
    TimelineEntry,
    TimelineEventType,
)
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario
from agentic_ai.tools.registry import (
    INVESTIGATION_TOOLS,
    ToolNotFoundError,
    build_tool_schemas,
    dispatch_tool,
)

MODEL = "llama3.2"

MAX_INVESTIGATION_STEPS = 8
"""One step = reason -> select tool -> execute -> observe.

Sized to leave headroom above MIN_DISTINCT_TOOLS. A model that tries to
conclude after every single call needs 3 tool calls plus 2 rejections just to
satisfy coverage; a budget of 5 would leave no room for a failed call or a
hallucinated tool name.
"""

MIN_DISTINCT_TOOLS = 3
"""Evidence coverage required before the agent may conclude.

The model has repeatedly judged one source sufficient and then reported high
confidence on partial evidence - including asserting metric values it never
retrieved. Sufficiency of evidence is a boundary, so deterministic code owns
it rather than the model's discretion.
"""


class InvestigationError(Exception):
    """Base exception for investigation failures."""


class InvalidConclusionError(InvestigationError):
    """The model repeatedly failed to produce a valid structured conclusion."""


def _append(existing: list, new: list) -> list:
    """Reducer: accumulate list state across nodes instead of overwriting."""
    return existing + new


class InvestigationState(TypedDict):
    """State threaded through the investigation subgraph."""

    incident_description: str
    scenario: str
    messages: Annotated[list[dict[str, Any]], _append]
    evidence: Annotated[list[Evidence], _append]
    timeline: Annotated[list[TimelineEntry], _append]
    steps_taken: int
    termination_reason: TerminationReason | None


_SYSTEM_PROMPT = (
    "You are an incident investigation agent for a production system.\n\n"
    "Your job is to find the ROOT CAUSE, not just restate the symptom. A "
    "single source of data is never enough: logs show what errors occurred, "
    "metrics show their magnitude and the health of anything this service "
    "depends on, and pod status shows whether the service's own instances "
    "are healthy. Corroborate across all of them before concluding.\n\n"
    "Be careful to distinguish a fault in this service from a fault "
    "somewhere else that this service is merely reporting. Evidence that the "
    "service's own resources are healthy is meaningful, not a dead end.\n\n"
    "You MUST gather evidence from all three of these tools before drawing "
    "any conclusion:\n"
    "- get_mock_logs\n"
    "- get_mock_metrics\n"
    "- get_mock_pod_status\n\n"
    "You may call them in whatever order the evidence suggests, but you must "
    "call all three.\n\n"
    "Rules:\n"
    "- Call ONE tool per turn.\n"
    "- Do NOT conclude until all three tools have been called.\n"
    "- Do NOT repeat a tool call you have already made.\n"
    "- You have at most {max_steps} tool calls in total.\n"
    "- Only state facts you actually observed in a tool result. Do not claim "
    "a metric value or a log message you have not seen.\n"
    "- Once all three tools have been called, reply in plain text explaining "
    "the root cause and the specific observations supporting it."
)


def reason_node(state: InvestigationState) -> dict[str, Any]:
    """Ask the model what to do next, given everything observed so far."""
    response = ollama.chat(
        model=MODEL,
        messages=state["messages"],
        tools=build_tool_schemas(),
    )

    message = response.message
    tool_calls = message.tool_calls or []

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
            }
            for call in tool_calls
        ]

    update: dict[str, Any] = {"messages": [assistant_message]}

    if not tool_calls:
        update["termination_reason"] = TerminationReason.COMPLETED
        update["timeline"] = [
            TimelineEntry(
                event_type=TimelineEventType.INVESTIGATION_CONCLUDED,
                description="Agent stopped requesting tools.",
            )
        ]

    return update


def execute_tools_node(state: InvestigationState) -> dict[str, Any]:
    """Execute the tool calls the model just requested.

    A tool that fails is recorded as an observation the agent can react to -
    it does not terminate the investigation.
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.get("tool_calls", [])

    messages: list[dict[str, Any]] = []
    evidence: list[Evidence] = []
    timeline: list[TimelineEntry] = []

    for call in tool_calls:
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]

        timeline.append(
            TimelineEntry(
                event_type=TimelineEventType.TOOL_SELECTED,
                description=f"Agent selected {name}.",
                metadata={"tool": name, "arguments": arguments},
            )
        )

        try:
            raw_output = dispatch_tool(name, arguments, scenario=state["scenario"])
        except ToolNotFoundError:
            raw_output = json.dumps(
                {
                    "status": "error",
                    "error_type": "tool_not_found",
                    "message": f"No tool named '{name}' exists.",
                    "available_tools": sorted(INVESTIGATION_TOOLS),
                }
            )
            timeline.append(
                TimelineEntry(
                    event_type=TimelineEventType.TOOL_FAILED,
                    description=f"Model requested unknown tool '{name}'.",
                    metadata={"tool": name},
                )
            )
        except TypeError as exc:
            # Wrong/missing arguments - also an observation, not a crash.
            raw_output = json.dumps(
                {
                    "status": "error",
                    "error_type": "invalid_arguments",
                    "message": str(exc),
                }
            )
            timeline.append(
                TimelineEntry(
                    event_type=TimelineEventType.TOOL_FAILED,
                    description=f"Invalid arguments for '{name}'.",
                    metadata={"tool": name, "arguments": arguments},
                )
            )
        else:
            timeline.append(
                TimelineEntry(
                    event_type=TimelineEventType.TOOL_SUCCEEDED,
                    description=f"{name} returned a result.",
                    metadata={"tool": name},
                )
            )
            evidence.append(
                Evidence(
                    source_tool=name,
                    tool_arguments=arguments,
                    raw_output=raw_output,
                    observation=f"{name}({arguments}) returned: {raw_output}",
                )
            )

        messages.append({"role": "tool", "content": raw_output, "name": name})

    return {
        "messages": messages,
        "evidence": evidence,
        "timeline": timeline,
        "steps_taken": state["steps_taken"] + 1,
    }


def _distinct_tools_used(state: InvestigationState) -> set[str]:
    """Tool names that actually returned evidence (failed calls don't count)."""
    return {item.source_tool for item in state["evidence"]}


def should_continue(state: InvestigationState) -> str:
    """Decide whether to keep investigating.

    The model's judgement is honoured only within deterministic bounds: it may
    not conclude before minimum evidence coverage is met, and it may not
    continue past the step budget.
    """
    if state["steps_taken"] >= MAX_INVESTIGATION_STEPS:
        return "budget_exhausted"

    if state["termination_reason"] is not None:
        if len(_distinct_tools_used(state)) >= MIN_DISTINCT_TOOLS:
            return "stop"
        return "insufficient_evidence"

    return "continue"


def require_more_evidence_node(state: InvestigationState) -> dict[str, Any]:
    """Reject a premature conclusion and send the agent back for more evidence.

    Clears ``termination_reason`` so the cycle resumes, and tells the model
    specifically which tools it has not yet used.
    """
    remaining = sorted(set(INVESTIGATION_TOOLS) - _distinct_tools_used(state))

    return {
        "termination_reason": None,
        # A rejection consumes budget. Without this, a model that repeatedly
        # claims to be done would cycle here forever: steps_taken is only
        # advanced by tool execution, so nothing would ever bound the loop.
        "steps_taken": state["steps_taken"] + 1,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You have not gathered enough evidence to conclude. You "
                    f"have not yet called: {', '.join(remaining)}. Call the "
                    "next one now."
                ),
            }
        ],
        "timeline": [
            TimelineEntry(
                event_type=TimelineEventType.INSUFFICIENT_EVIDENCE,
                description=(
                    "Agent tried to conclude with "
                    f"{len(_distinct_tools_used(state))} of "
                    f"{MIN_DISTINCT_TOOLS} required sources."
                ),
                metadata={"remaining_tools": remaining},
            )
        ],
    }


def budget_exhausted_node(state: InvestigationState) -> dict[str, Any]:
    """Record that deterministic code, not the agent, ended the investigation."""
    return {
        "termination_reason": TerminationReason.MAX_STEPS_REACHED,
        "timeline": [
            TimelineEntry(
                event_type=TimelineEventType.STEP_LIMIT_REACHED,
                description=(
                    f"Stopped after {MAX_INVESTIGATION_STEPS} steps without "
                    "the agent concluding."
                ),
            )
        ],
    }


def build_investigation_graph():
    """Wire the bounded reason/act cycle."""
    builder = StateGraph(InvestigationState)

    builder.add_node("reason", reason_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("require_more_evidence", require_more_evidence_node)
    builder.add_node("budget_exhausted", budget_exhausted_node)

    builder.add_edge(START, "reason")

    builder.add_conditional_edges(
        "reason",
        should_continue,
        path_map={
            "continue": "execute_tools",
            "stop": END,
            "insufficient_evidence": "require_more_evidence",
            "budget_exhausted": "budget_exhausted",
        },
    )

    # The cycle: observations feed back into reasoning.
    builder.add_edge("execute_tools", "reason")
    # A rejected conclusion also re-enters the cycle. The step budget still
    # bounds this, so it cannot loop forever.
    builder.add_edge("require_more_evidence", "reason")
    builder.add_edge("budget_exhausted", END)

    return builder.compile()


investigation_graph = build_investigation_graph()


def investigate(
    incident_description: str,
    scenario: Scenario | str = DEFAULT_SCENARIO,
) -> InvestigationResult:
    """Run a bounded investigation and return a structured result.

    ``scenario`` selects which mock fixture set the tools draw from. It exists
    only because the tools are mocks; a real deployment would have one reality.
    """
    initial_state: InvestigationState = {
        "incident_description": incident_description,
        "scenario": Scenario(scenario).value,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(max_steps=MAX_INVESTIGATION_STEPS),
            },
            {"role": "user", "content": f"Incident: {incident_description}"},
        ],
        "evidence": [],
        "timeline": [
            TimelineEntry(
                event_type=TimelineEventType.INVESTIGATION_STARTED,
                description="Investigation started.",
                metadata={"incident": incident_description},
            )
        ],
        "steps_taken": 0,
        "termination_reason": None,
    }

    final_state = investigation_graph.invoke(initial_state)

    return _summarise(final_state)


_SUMMARY_INSTRUCTION = (
    "Now write your investigation report as JSON, based only on the tool "
    "results above. Do not invent facts.\n\n"
    "- summary: two or three sentences stating what is wrong with the "
    "service and why, citing the specific numbers and error messages you "
    "observed.\n"
    "- findings: one entry per conclusion you drew from the evidence. Each "
    "statement should name the concrete signal that supports it (for example "
    "an exact metric value or log message). If the evidence supports a root "
    "cause, say so explicitly.\n"
    "- model_assessed_confidence: 0.0 to 1.0, how strongly the evidence "
    "supports your conclusion.\n"
    "- unresolved_questions: what you still could not determine.\n"
    "- recommended_next_action: the single most useful next step."
)


def _summarise(state: InvestigationState) -> InvestigationResult:
    """Turn the accumulated transcript into a validated InvestigationResult."""
    termination_reason = state["termination_reason"] or TerminationReason.MAX_STEPS_REACHED

    schema = InvestigationResult.model_json_schema()
    # Evidence, timeline and termination_reason are owned by deterministic
    # code - the model must not author them.
    for owned in ("evidence", "timeline", "termination_reason"):
        schema["properties"].pop(owned, None)
    schema["required"] = [
        field for field in schema.get("required", []) if field not in
        ("evidence", "timeline", "termination_reason")
    ]

    instruction = _SUMMARY_INSTRUCTION
    used = _distinct_tools_used(state)
    if len(used) < MIN_DISTINCT_TOOLS:
        missing = sorted(set(INVESTIGATION_TOOLS) - used)
        instruction += (
            "\n\nIMPORTANT: this investigation is INCOMPLETE. You never "
            f"retrieved data from: {', '.join(missing)}. Do not state facts "
            "from those sources. Keep model_assessed_confidence at or below "
            "0.5, and list what you could not check in unresolved_questions."
        )

    messages = state["messages"] + [{"role": "user", "content": instruction}]

    last_error: Exception | None = None
    for _ in range(2):
        response = ollama.chat(model=MODEL, messages=messages, format=schema)
        try:
            partial = json.loads(response.message.content)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        try:
            return InvestigationResult(
                **partial,
                evidence=state["evidence"],
                timeline=state["timeline"],
                termination_reason=termination_reason,
            )
        except (ValidationError, TypeError) as exc:
            last_error = exc

    raise InvalidConclusionError(
        "Model failed to produce a valid investigation conclusion after retry."
    ) from last_error
