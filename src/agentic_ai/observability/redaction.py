"""Deterministic sanitization and secret redaction engine for operational observability.

Provides utilities to redact credentials, API keys, bearer tokens, authorization
headers, passwords, and sensitive keys from structured payloads and text strings.
Fails safely if an unexpected object or cyclic reference is encountered.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

DEFAULT_MAX_STRING_LENGTH = 500
DEFAULT_MAX_DEPTH = 10
REDACTED_PLACEHOLDER = "[REDACTED]"
MAX_DEPTH_PLACEHOLDER = "[MAX_DEPTH_EXCEEDED]"
UNSAFE_FALLBACK_PLACEHOLDER = "[UNSERIALIZABLE_PAYLOAD_REDACTED]"

# Sensitive key keywords (case-insensitive substring match)
SENSITIVE_KEY_PATTERNS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "private_key",
        "privatekey",
        "access_key",
        "accesskey",
        "client_secret",
        "clientsecret",
        "cert",
    }
)

# Inline patterns for regex-based string masking
_URL_BASIC_AUTH_REGEX = re.compile(r"(https?://)([^:\s]+):([^@\s]+)@")
_BASIC_AUTH_HEADER_REGEX = re.compile(
    r"(?i)\b(authorization\s*:\s*basic\s+)([^\s,\"\'&]+)",
)
_GENERIC_AUTH_HEADER_REGEX = re.compile(
    r"(?i)\b(authorization\s*:\s*)(?!bearer\s+|basic\s+)([^\s,\"\'&]+)",
)
_BEARER_TOKEN_REGEX = re.compile(
    r"(?i)\b(bearer\s+)([A-Za-z0-9_\-\.~+/]+=*)",
)
# Note: sk-ant- is ordered before sk- to avoid partial prefix shadowing
_KEY_PREFIX_REGEXES = (
    re.compile(r"\b(sk-ant-[A-Za-z0-9_\-]{10,})\b"),  # Anthropic-style
    re.compile(r"\b(sk-[A-Za-z0-9_\-]{10,})\b"),  # OpenAI-style
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),  # AWS Access Key ID
    re.compile(r"\b(ghp_[A-Za-z0-9]{36})\b"),  # GitHub Personal Access Token
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9_\-]{10,})\b"),  # Slack tokens
)
_INLINE_KEY_VALUE_REGEX = re.compile(
    r"(?i)([\?&]?(?:x[_-]?api[_-]?key|password|passwd|pwd|secret|auth[_-]?token|token|api[_-]?key|apikey|credentials?|private[_-]?key|access[_-]?key|client[_-]?secret)\s*[:=]\s*)([\"']?)([^\s,\"\'&]+)([\"']?)",
)
_CAMEL_TO_SNAKE_REGEX = re.compile(r"([a-z0-9])([A-Z])")


def _is_sensitive_key(key: str) -> bool:
    """Determine whether a dictionary key is sensitive across camelCase, snake_case, and kebab-case."""
    # Convert camelCase (e.g. privateKey, accessKey, clientSecret, PassWord) to snake_case
    converted = _CAMEL_TO_SNAKE_REGEX.sub(r"\1_\2", key)
    normalized = converted.lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return any(p in normalized or p in compact for p in SENSITIVE_KEY_PATTERNS)


def mask_secrets(text: str) -> str:
    """Redact embedded credentials, API keys, tokens, and basic auth from a string.

    Guarantees that non-sensitive text is preserved while secrets are masked.
    Fails safely if an unexpected error occurs.
    """
    if not isinstance(text, str):
        return text

    try:
        # 1. URL basic auth
        masked = _URL_BASIC_AUTH_REGEX.sub(
            rf"\1{REDACTED_PLACEHOLDER}:{REDACTED_PLACEHOLDER}@", text
        )

        # 2. Authorization headers (Basic or raw custom token)
        masked = _BASIC_AUTH_HEADER_REGEX.sub(rf"\1{REDACTED_PLACEHOLDER}", masked)
        masked = _GENERIC_AUTH_HEADER_REGEX.sub(rf"\1{REDACTED_PLACEHOLDER}", masked)

        # 3. Standalone Bearer tokens (including in Authorization headers)
        masked = _BEARER_TOKEN_REGEX.sub(rf"\1{REDACTED_PLACEHOLDER}", masked)

        # 4. Known API key / token prefixes (Anthropic sk-ant- before OpenAI sk- to avoid prefix shadowing)
        for regex in _KEY_PREFIX_REGEXES:
            masked = regex.sub(REDACTED_PLACEHOLDER, masked)

        # 5. Inline key=value or key: value assignments and query parameters
        masked = _INLINE_KEY_VALUE_REGEX.sub(rf"\1\2{REDACTED_PLACEHOLDER}\4", masked)

        return masked
    except Exception:
        return REDACTED_PLACEHOLDER


def _truncate_string(text: str, max_str_len: int) -> str:
    """Truncate text exceeding maximum allowed length."""
    if len(text) > max_str_len:
        return text[:max_str_len] + "... [TRUNCATED]"
    return text


def sanitize_payload(
    payload: Any,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_str_len: int = DEFAULT_MAX_STRING_LENGTH,
    _current_depth: int = 0,
    _visited: set[int] | None = None,
) -> Any:
    """Recursively sanitize any Python payload, redacting secrets and truncating strings.

    Fails safely: if an unexpected exception occurs or a cyclic structure is detected,
    returns a safe placeholder string without leaking unsanitized content.
    """
    try:
        if _visited is None:
            _visited = set()

        if _current_depth > max_depth:
            return MAX_DEPTH_PLACEHOLDER

        # Handle None and booleans (bool is a subclass of int in Python)
        if payload is None or isinstance(payload, bool):
            return payload

        # Handle numeric primitives
        if isinstance(payload, (int, float)):
            return payload

        # Handle strings
        if isinstance(payload, str):
            masked = mask_secrets(payload)
            return _truncate_string(masked, max_str_len=max_str_len)

        # Handle Pydantic models
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()

        obj_id = id(payload)
        if obj_id in _visited:
            return "[CYCLIC_REFERENCE_REDACTED]"
        _visited.add(obj_id)

        # Handle dictionaries
        if isinstance(payload, dict):
            sanitized_dict: dict[str, Any] = {}
            for key, val in payload.items():
                str_key = str(key)
                if _is_sensitive_key(str_key):
                    sanitized_dict[str_key] = REDACTED_PLACEHOLDER
                else:
                    sanitized_dict[str_key] = sanitize_payload(
                        val,
                        max_depth=max_depth,
                        max_str_len=max_str_len,
                        _current_depth=_current_depth + 1,
                        _visited=_visited,
                    )
            return sanitized_dict

        # Handle lists, tuples, sets
        if isinstance(payload, (list, tuple, set)):
            sanitized_list = [
                sanitize_payload(
                    item,
                    max_depth=max_depth,
                    max_str_len=max_str_len,
                    _current_depth=_current_depth + 1,
                    _visited=_visited,
                )
                for item in payload
            ]
            return sanitized_list

        # Other types: attempt string conversion, sanitize, and truncate
        str_repr = str(payload)
        masked_str = mask_secrets(str_repr)
        return _truncate_string(masked_str, max_str_len=max_str_len)

    except Exception:
        # Strict fail-safe guarantee: never emit raw unsanitized content on failure
        return UNSAFE_FALLBACK_PLACEHOLDER


def sanitize_dict(
    data: dict[str, Any],
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_str_len: int = DEFAULT_MAX_STRING_LENGTH,
) -> dict[str, Any]:
    """Sanitize a dictionary ensuring the return value is a dict[str, Any].

    Guarantees safe dictionary return type even if sanitization encounters anomalies.
    """
    try:
        res = sanitize_payload(data, max_depth=max_depth, max_str_len=max_str_len)
        if isinstance(res, dict):
            return res
        return {"_sanitization_fallback": UNSAFE_FALLBACK_PLACEHOLDER}
    except Exception:
        return {"_sanitization_fallback": UNSAFE_FALLBACK_PLACEHOLDER}


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_STRING_LENGTH",
    "MAX_DEPTH_PLACEHOLDER",
    "REDACTED_PLACEHOLDER",
    "UNSAFE_FALLBACK_PLACEHOLDER",
    "mask_secrets",
    "sanitize_dict",
    "sanitize_payload",
]
