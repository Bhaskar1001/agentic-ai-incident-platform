"""OpenRouter provider: an OpenAI-compatible HTTP client behind LLMClient.

OpenRouter's wire format differs from Ollama's in ways that matter:

- Content lives at ``choices[0].message``, not ``message`` directly.
- Tool call arguments arrive as a JSON *string* that must be parsed, not an
  already-parsed dict.
- Structured output is requested via ``response_format`` (a nested
  ``json_schema`` object with a ``strict`` flag), not a bare ``format=``
  schema dict.

None of that should leak past this file - ``complete()`` returns the same
``ChatResponse`` shape as every other provider.
"""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

from agentic_ai.llm.client import (
    ChatResponse,
    LLMConfigurationError,
    LLMRateLimitError,
    ToolCall,
)

load_dotenv()

DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60.0


class OpenRouterClient:
    """LLMClient backed by OpenRouter's chat completions API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
                "and fill in a key from https://openrouter.ai/keys."
            )

        self.model = model or os.environ.get(
            "OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL
        )

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
    ) -> ChatResponse:
        payload: dict = {"model": self.model, "messages": messages}

        if tools:
            payload["tools"] = tools

        if json_schema:
            # OpenRouter (OpenAI-style) wants the schema nested under a
            # named json_schema object, not passed bare. "strict" asks the
            # provider to reject output that does not conform, where the
            # underlying model supports it.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        response = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 429:
            raise _rate_limit_error(response)

        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]

        tool_calls = [
            ToolCall(
                name=call["function"]["name"],
                # Arguments arrive as a JSON string over this wire format,
                # unlike Ollama which hands back an already-parsed dict.
                arguments=json.loads(call["function"]["arguments"]),
                id=call["id"],
            )
            for call in (message.get("tool_calls") or [])
        ]

        return ChatResponse(content=message.get("content"), tool_calls=tool_calls)


def _rate_limit_error(response: httpx.Response) -> LLMRateLimitError:
    """Build a clear error from a 429, distinguishing OpenRouter's daily
    free-model quota from any other rate limiting, since the two call for
    completely different responses (wait until tomorrow vs. a short retry).
    """
    try:
        body = response.json()
        error = body.get("error", {})
        metadata = error.get("metadata", {})
    except (json.JSONDecodeError, AttributeError):
        error, metadata = {}, {}

    limit_source = metadata.get("limit_source", "")
    reset_ms = metadata.get("headers", {}).get("X-RateLimit-Reset")
    remedy = metadata.get("remedy_hint", "")

    if limit_source == "openrouter_free_tier_daily":
        reset_note = ""
        if reset_ms:
            import datetime

            reset_time = datetime.datetime.fromtimestamp(
                int(reset_ms) / 1000, tz=datetime.timezone.utc
            )
            reset_note = f" Resets at {reset_time.isoformat()}."

        return LLMRateLimitError(
            "OpenRouter's free-model daily quota (50 requests/day) is "
            f"exhausted.{reset_note} {remedy}".strip(),
            retry_after=reset_note.strip() or None,
        )

    # A rate limit that is not the known daily cap - still surfaced as a
    # distinct type so callers can tell "LLM refused due to volume" apart
    # from an arbitrary HTTP failure, even though this codebase does not
    # yet know how to distinguish a case worth a short automatic retry.
    message = error.get("message") or "OpenRouter rate limit exceeded."
    return LLMRateLimitError(message)
