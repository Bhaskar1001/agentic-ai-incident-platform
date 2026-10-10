"""Tests for OpenRouterClient's HTTP behaviour, especially rate-limit handling.

Uses httpx's mock transport rather than the real network: these must be
deterministic and must never consume real API quota, which is exactly the
scarce resource this file is testing the handling of.
"""

import httpx
import pytest

from agentic_ai.llm.client import LLMRateLimitError
from agentic_ai.llm.providers.openrouter_provider import OpenRouterClient


def _patch_post(monkeypatch, status_code: int, body: dict):
    """Patch httpx.post globally for the duration of one test."""

    def fake_post(url, *, headers, json, timeout):
        return httpx.Response(status_code, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(
        "agentic_ai.llm.providers.openrouter_provider.httpx.post", fake_post
    )


_DAILY_QUOTA_BODY = {
    "error": {
        "message": (
            "Rate limit exceeded: free-models-per-day. Add 10 credits to "
            "unlock 1000 free model requests per day"
        ),
        "code": 429,
        "metadata": {
            "headers": {
                "X-RateLimit-Limit": "50",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1791676800000",
            },
            "limit_source": "openrouter_free_tier_daily",
            "remedy_hint": (
                "Wait for the daily reset (see X-RateLimit-Reset), or "
                "purchase credits to raise your free-model daily limit."
            ),
        },
    }
}


def test_daily_quota_429_raises_llm_rate_limit_error(monkeypatch) -> None:
    _patch_post(monkeypatch, 429, _DAILY_QUOTA_BODY)
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.complete(messages=[{"role": "user", "content": "hi"}])

    assert "daily quota" in str(exc_info.value)
    assert "50 requests/day" in str(exc_info.value)


def test_daily_quota_error_includes_the_reset_time(monkeypatch) -> None:
    _patch_post(monkeypatch, 429, _DAILY_QUOTA_BODY)
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.complete(messages=[{"role": "user", "content": "hi"}])

    assert exc_info.value.retry_after is not None
    assert "2026-10-11" in exc_info.value.retry_after


def test_non_daily_429_still_raises_rate_limit_error_without_fabricating_a_reset(
    monkeypatch,
) -> None:
    """A 429 without the known daily-quota shape must still be distinguishable
    from an arbitrary HTTP failure, but must not invent a reset time it was
    not actually given.
    """
    body = {"error": {"message": "Too many requests, please slow down."}}
    _patch_post(monkeypatch, 429, body)
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.complete(messages=[{"role": "user", "content": "hi"}])

    assert "Too many requests" in str(exc_info.value)
    assert exc_info.value.retry_after is None


def test_429_with_unparseable_body_still_raises_rate_limit_error(monkeypatch) -> None:
    """Even a malformed error body must not crash while handling a 429 -
    that would replace a clear rate-limit signal with a confusing one.
    """

    def fake_post(url, *, headers, json, timeout):
        request = httpx.Request("POST", url)
        return httpx.Response(429, content=b"not json", request=request)

    monkeypatch.setattr(
        "agentic_ai.llm.providers.openrouter_provider.httpx.post", fake_post
    )
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(LLMRateLimitError):
        client.complete(messages=[{"role": "user", "content": "hi"}])


def test_non_429_errors_still_raise_their_ordinary_http_error(monkeypatch) -> None:
    """A 500 or similar must not be reclassified as a rate limit."""
    body = {"error": {"message": "internal server error"}}
    _patch_post(monkeypatch, 500, body)
    client = OpenRouterClient(api_key="test-key")

    with pytest.raises(httpx.HTTPStatusError):
        client.complete(messages=[{"role": "user", "content": "hi"}])


def test_successful_response_is_unaffected_by_the_429_handling(monkeypatch) -> None:
    body = {
        "choices": [{"message": {"role": "assistant", "content": "all good"}}]
    }
    _patch_post(monkeypatch, 200, body)
    client = OpenRouterClient(api_key="test-key")

    response = client.complete(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "all good"
