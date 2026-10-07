"""Ollama provider: wraps the local ``ollama`` package behind LLMClient.

Kept alongside the OpenRouter provider (rather than deleted) specifically so
the abstraction has two real implementations to prove it against - an
interface validated by only ever having one provider behind it is not
actually validated.
"""

from __future__ import annotations

import os
from typing import Any

import ollama

from agentic_ai.llm.client import ChatResponse, ToolCall

DEFAULT_OLLAMA_MODEL = "llama3.2"


class OllamaClient:
    """LLMClient backed by a local Ollama server."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get(
            "OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> ChatResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if json_schema:
            kwargs["format"] = json_schema

        response = ollama.chat(**kwargs)
        message = response.message

        tool_calls = [
            ToolCall(
                name=call.function.name,
                arguments=dict(call.function.arguments),
                # Ollama's tool calls carry no id; synthesise a positional one
                # stable within this single response, which is all a
                # tool_call_id needs to be.
                id=f"call_{index}",
            )
            for index, call in enumerate(message.tool_calls or [])
        ]

        return ChatResponse(content=message.content, tool_calls=tool_calls)
