"""Provider-agnostic LLM client interface.

Two call sites need an LLM: a one-shot structured-output call
(``llm/severity.py``) and a multi-turn tool-calling loop
(``agents/investigation.py``). Both only ever need one operation -
"given these messages, optionally these tools, optionally this output
schema, give me back a response" - so that is the entire interface.

Each provider translates its own wire format into the shapes below and
back. Callers never see Ollama's ``response.message`` or OpenRouter's
``response.choices[0].message`` - only ``ChatResponse``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the model requested.

    ``arguments`` is always a parsed dict here, regardless of whether the
    underlying provider handed back a dict (Ollama) or a JSON string
    (OpenRouter/OpenAI-style) - that normalisation is the provider's job,
    not the caller's.

    ``id`` must round-trip back to the provider as ``tool_call_id`` on the
    matching tool-result message. OpenAI-compatible APIs (OpenRouter) reject
    a request that omits it. Ollama's tool calls carry no id at all, so that
    provider synthesises a stable one - callers must not assume the id means
    anything beyond "pairs this result with that call".
    """

    name: str
    arguments: dict[str, Any]
    id: str


@dataclass(frozen=True)
class ChatResponse:
    """A normalised model response, regardless of provider."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    """What every provider must implement."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Run one completion.

        Args:
            messages: OpenAI-shaped message dicts (role/content, plus
                tool_calls/tool_call_id where relevant). Both providers
                already speak this dialect in this codebase.
            tools: OpenAI-shaped tool definitions
                (``{"type": "function", "function": {...}}``), as produced
                by ``tools.registry.build_tool_schemas()``.
            json_schema: A JSON Schema the response content must conform
                to. When set, the provider constrains generation to match
                it (Ollama's ``format=``, OpenRouter's
                ``response_format={"type": "json_schema", ...}``).
                Mutually exclusive with expecting tool calls in practice,
                but that is a caller concern, not this interface's.
        """
        ...


class LLMConfigurationError(Exception):
    """Raised when the configured provider is missing required setup."""


class LLMRateLimitError(Exception):
    """Raised when a provider refuses a request due to a rate or quota limit.

    Deliberately not retried automatically anywhere in this codebase. The
    case that motivated this - OpenRouter's free-model daily quota - resets
    once every 24 hours; a short backoff-and-retry would fail silently for
    up to that long rather than surface the real problem. ``retry_after``
    carries whatever the provider reported about when the limit clears, so a
    caller can decide what "come back later" actually means here, rather
    than guessing from a generic HTTP error.
    """

    def __init__(self, message: str, *, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def get_client() -> LLMClient:
    """Build the configured provider from the environment.

    ``LLM_PROVIDER`` selects the provider ("openrouter" or "ollama"),
    defaulting to "openrouter". This is read fresh on every call rather
    than cached at import time, so tests can monkeypatch the environment
    without import-order issues.
    """
    provider = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()

    if provider == "openrouter":
        from agentic_ai.llm.providers.openrouter_provider import (
            OpenRouterClient,
        )

        return OpenRouterClient()

    if provider == "ollama":
        from agentic_ai.llm.providers.ollama_provider import OllamaClient

        return OllamaClient()

    raise LLMConfigurationError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected 'openrouter' or 'ollama'."
    )
