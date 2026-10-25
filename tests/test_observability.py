"""Tests for Phase 14 Step 1: Observability domain models and redaction engine.

Verifies:
- Observability event models validation, frozen immutability, and JSON serialization.
- Masking of embedded secrets, API keys, Bearer tokens, and URL basic auth in strings.
- Redaction of sensitive dictionary keys (case-insensitive) across nested dicts and lists.
- Truncation of oversized payloads and depth protection.
- Preservation of normal non-sensitive operational telemetry.
- Fail-safe handling of unsupported types, cyclic structures, and edge-case exceptions.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_ai.observability.events import (
    DiagnosticReason,
    ErrorCategory,
    IncidentTrace,
    StageStatus,
    StageTrace,
    ToolInvocationTrace,
)
from agentic_ai.observability.redaction import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_STRING_LENGTH,
    MAX_DEPTH_PLACEHOLDER,
    REDACTED_PLACEHOLDER,
    UNSAFE_FALLBACK_PLACEHOLDER,
    mask_secrets,
    sanitize_dict,
    sanitize_payload,
)


# --- 1. Domain Model Tests ---------------------------------------------------


def test_tool_invocation_trace_model() -> None:
    tool_trace = ToolInvocationTrace(
        tool_name="get_mock_metrics",
        target_service="payment",
        duration_ms=45.2,
        status="success",
        sanitized_arguments={"service": "payment"},
    )
    assert tool_trace.tool_name == "get_mock_metrics"
    assert tool_trace.duration_ms == 45.2
    assert tool_trace.status == "success"
    assert tool_trace.error_category is None

    # Immutability check
    with pytest.raises(ValidationError):
        tool_trace.status = "failure"  # type: ignore[misc]

    # JSON roundtrip
    dumped_json = tool_trace.model_dump_json()
    loaded = ToolInvocationTrace.model_validate_json(dumped_json)
    assert loaded == tool_trace


def test_stage_trace_model() -> None:
    now = datetime.now(timezone.utc)
    stage_trace = StageTrace(
        stage_name="investigate",
        status=StageStatus.COMPLETED,
        start_time=now,
        end_time=now,
        duration_ms=120.5,
        metadata={"service": "payment", "steps_taken": 3},
    )
    assert stage_trace.stage_name == "investigate"
    assert stage_trace.status == StageStatus.COMPLETED
    assert stage_trace.duration_ms == 120.5

    # Immutability
    with pytest.raises(ValidationError):
        stage_trace.duration_ms = 200.0  # type: ignore[misc]


def test_diagnostic_reason_model() -> None:
    diag = DiagnosticReason(
        decision="resolve",
        primary_reason="Verification confirmed health recovery.",
        trigger_type="verification_recovered",
        details={"attempts": 1, "service": "payment"},
    )
    assert diag.decision == "resolve"
    assert diag.trigger_type == "verification_recovered"

    dumped = diag.model_dump()
    assert dumped["decision"] == "resolve"
    assert dumped["trigger_type"] == "verification_recovered"


def test_incident_trace_model() -> None:
    now = datetime.now(timezone.utc)
    stage = StageTrace(
        stage_name="assess_severity",
        status=StageStatus.COMPLETED,
        start_time=now,
        end_time=now,
        duration_ms=10.0,
    )
    tool = ToolInvocationTrace(
        tool_name="get_mock_logs",
        target_service="payment",
        duration_ms=25.0,
        status="success",
    )
    trace = IncidentTrace(
        incident_id="INC-1001",
        correlation_id="CORR-9999",
        start_time=now,
        stages=[stage],
        tools=[tool],
        final_outcome="escalate",
    )
    assert trace.incident_id == "INC-1001"
    assert trace.correlation_id == "CORR-9999"
    assert len(trace.stages) == 1
    assert len(trace.tools) == 1

    # Immutability
    with pytest.raises(ValidationError):
        trace.final_outcome = "resolve"  # type: ignore[misc]

    # JSON roundtrip
    dumped_json = trace.model_dump_json()
    loaded = IncidentTrace.model_validate_json(dumped_json)
    assert loaded.incident_id == trace.incident_id
    assert loaded.stages[0].stage_name == "assess_severity"


def test_error_category_and_stage_status_enums() -> None:
    assert ErrorCategory.APP_ERROR == "app_error"
    assert ErrorCategory.TOOL_ERROR == "tool_error"
    assert ErrorCategory.TIMEOUT == "timeout"
    assert ErrorCategory.SAFETY_GATE_BLOCKED == "safety_gate_blocked"
    assert ErrorCategory.VALIDATION_ERROR == "validation_error"

    assert StageStatus.STARTED == "started"
    assert StageStatus.COMPLETED == "completed"
    assert StageStatus.FAILED == "failed"
    assert StageStatus.SKIPPED == "skipped"
    assert StageStatus.BLOCKED == "blocked"


# --- 2. String Secret Masking Tests ------------------------------------------


def test_mask_secrets_bearer_token() -> None:
    text = "Authorization header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz"
    masked = mask_secrets(text)
    assert "Bearer [REDACTED]" in masked
    assert "eyJhbGci" not in masked


def test_mask_secrets_url_basic_auth() -> None:
    url = (
        "Failed to connect to https://admin_user:super_secret_pass@cluster.internal:8080/v1/health"
    )
    masked = mask_secrets(url)
    assert "https://[REDACTED]:[REDACTED]@cluster.internal:8080/v1/health" in masked
    assert "super_secret_pass" not in masked
    assert "admin_user" not in masked


def test_mask_secrets_api_key_patterns() -> None:
    samples = [
        ("OpenAI key sk-proj-1234567890abcdef1234567890 is invalid", "sk-proj-"),
        ("Anthropic key sk-ant-api03-abcdef1234567890-xyz failed", "sk-ant-"),
        ("AWS Key AKIAIOSFODNN7EXAMPLE used", "AKIAIOSFODNN7EXAMPLE"),
        ("GitHub Token ghp_123456789012345678901234567890123456", "ghp_"),
        ("Slack bot xoxb-1234567890-abcdef123456-secret", "xoxb-"),
    ]
    for text, secret in samples:
        masked = mask_secrets(text)
        assert secret not in masked
        assert REDACTED_PLACEHOLDER in masked


def test_mask_secrets_inline_assignments() -> None:
    text = "Database connection error: password=secret_password_123, api_key: 'top_secret_key', token=\"jwt_token_here\""
    masked = mask_secrets(text)
    assert "secret_password_123" not in masked
    assert "top_secret_key" not in masked
    assert "jwt_token_here" not in masked
    assert "password=[REDACTED]" in masked
    assert "api_key: '[REDACTED]'" in masked
    assert 'token="[REDACTED]"' in masked


def test_mask_secrets_preserves_non_sensitive_strings() -> None:
    ordinary = "Payment service active connections: 85/100, error rate: 0.02, pods ready: 3/3."
    assert mask_secrets(ordinary) == ordinary


# --- 3. Dictionary & Nested Payload Sanitization Tests -----------------------


def test_sanitize_dict_sensitive_keys_case_insensitive() -> None:
    data = {
        "password": "plain_password",
        "PASSWORD": "upper_password",
        "user_secret": "my_secret",
        "api_key": "key_123",
        "ApiKey": "key_456",
        "AUTH_TOKEN": "token_789",
        "client_secret": "cs_000",
        "private_key": "pk_111",
        "credentials": {"user": "admin", "pass": "secret"},
        "safe_field": "unaltered_value",
        "status_code": 200,
    }
    sanitized = sanitize_dict(data)

    assert sanitized["password"] == REDACTED_PLACEHOLDER
    assert sanitized["PASSWORD"] == REDACTED_PLACEHOLDER
    assert sanitized["user_secret"] == REDACTED_PLACEHOLDER
    assert sanitized["api_key"] == REDACTED_PLACEHOLDER
    assert sanitized["ApiKey"] == REDACTED_PLACEHOLDER
    assert sanitized["AUTH_TOKEN"] == REDACTED_PLACEHOLDER
    assert sanitized["client_secret"] == REDACTED_PLACEHOLDER
    assert sanitized["private_key"] == REDACTED_PLACEHOLDER
    assert sanitized["credentials"] == REDACTED_PLACEHOLDER
    # Safe fields preserved
    assert sanitized["safe_field"] == "unaltered_value"
    assert sanitized["status_code"] == 200


def test_sanitize_nested_structures_and_lists() -> None:
    nested = {
        "service": "payment",
        "config": {
            "retries": 3,
            "security": {
                "auth_header": "Bearer secret_jwt",
                "cert_path": "/etc/ssl/cert.pem",
            },
        },
        "endpoints": [
            {"path": "/health", "is_public": True},
            {"path": "/admin", "token": "admin_token_xyz"},
            "URL: https://app:secret123@internal.net/sync",
        ],
    }
    sanitized = sanitize_dict(nested)

    assert sanitized["service"] == "payment"
    assert sanitized["config"]["retries"] == 3
    assert sanitized["config"]["security"]["auth_header"] == REDACTED_PLACEHOLDER
    assert sanitized["config"]["security"]["cert_path"] == REDACTED_PLACEHOLDER

    # Endpoints list check
    endpoints = sanitized["endpoints"]
    assert endpoints[0]["path"] == "/health"
    assert endpoints[0]["is_public"] is True
    assert endpoints[1]["token"] == REDACTED_PLACEHOLDER
    assert "secret123" not in endpoints[2]
    assert "https://[REDACTED]:[REDACTED]@internal.net/sync" in endpoints[2]


def test_sanitize_payload_with_pydantic_model() -> None:
    class SubConfig(ToolInvocationTrace):
        pass

    trace = ToolInvocationTrace(
        tool_name="get_mock_metrics",
        target_service="payment",
        duration_ms=10.0,
        status="success",
        sanitized_arguments={"token": "should_be_redacted", "service": "payment"},
    )

    sanitized = sanitize_payload(trace)
    assert isinstance(sanitized, dict)
    assert sanitized["tool_name"] == "get_mock_metrics"
    assert sanitized["sanitized_arguments"]["token"] == REDACTED_PLACEHOLDER
    assert sanitized["sanitized_arguments"]["service"] == "payment"


def test_sanitize_truncation_limits() -> None:
    long_string = "A" * 1000
    sanitized = sanitize_payload(long_string, max_str_len=100)
    assert len(sanitized) == 100 + len("... [TRUNCATED]")
    assert sanitized.endswith("... [TRUNCATED]")


def test_sanitize_depth_limits() -> None:
    # Build deeply nested dictionary
    curr: dict[str, Any] = {"leaf": "deep"}
    for _ in range(15):
        curr = {"inner": curr}

    sanitized = sanitize_payload(curr, max_depth=5)
    # At depth > 5 it must be replaced by MAX_DEPTH_PLACEHOLDER
    d = sanitized
    for _ in range(6):
        d = d["inner"]
    assert d == MAX_DEPTH_PLACEHOLDER


# --- 4. Robustness & Fail-Safe Handling Tests --------------------------------


def test_sanitize_cyclic_reference_handling() -> None:
    cyclic_dict: dict[str, Any] = {"service": "payment"}
    cyclic_dict["self"] = cyclic_dict

    sanitized = sanitize_payload(cyclic_dict)
    assert sanitized["service"] == "payment"
    assert sanitized["self"] == "[CYCLIC_REFERENCE_REDACTED]"


def test_sanitize_unexpected_unserializable_input() -> None:
    class ExplosiveObject:
        def __str__(self) -> str:
            raise RuntimeError("Explosion on str conversion!")

    # Must fail safely without leaking raw object or raising exception
    result = sanitize_payload(ExplosiveObject())
    assert result == UNSAFE_FALLBACK_PLACEHOLDER


def test_sanitize_dict_safe_return_on_failure() -> None:
    class MalformedDictKey:
        def __hash__(self) -> int:
            return 1

        def __str__(self) -> str:
            raise ValueError("Key str conversion failed")

    bad_dict = {MalformedDictKey(): "secret_value"}
    res = sanitize_dict(bad_dict)
    assert isinstance(res, dict)
    assert "_sanitization_fallback" in res
    assert res["_sanitization_fallback"] == UNSAFE_FALLBACK_PLACEHOLDER
    assert "secret_value" not in str(res)


def test_error_message_containing_credentials_sanitized() -> None:
    err = (
        "InvestigationError: Failed calling get_mock_logs for service 'payment': "
        "Error connecting to https://user:pass999@metrics.infra.internal/logs?api_key=sk-1234567890abcdef1234567890: "
        "Bearer token_xyz expired."
    )
    sanitized = mask_secrets(err)
    assert "pass999" not in sanitized
    assert "sk-1234567890abcdef1234567890" not in sanitized
    assert "token_xyz" not in sanitized
    assert "https://[REDACTED]:[REDACTED]@" in sanitized
    assert "api_key=[REDACTED]" in sanitized
    assert "Bearer [REDACTED]" in sanitized


# --- 5. Security & Correctness Audit Tests -----------------------------------


def test_mask_secrets_overlapping_prefixes() -> None:
    """Audit: Overlapping token prefixes (sk-ant- and sk-) must both be masked completely."""
    text = (
        "Keys in trace: Anthropic sk-ant-api03-abcdef1234567890-xyz and "
        "OpenAI sk-1234567890abcdef1234567890."
    )
    masked = mask_secrets(text)
    assert "sk-ant-" not in masked
    assert "sk-1234" not in masked
    assert "abcdef1234567890" not in masked
    assert masked.count(REDACTED_PLACEHOLDER) == 2


def test_mask_secrets_authorization_header_variants() -> None:
    """Audit: Authorization header variations (Bearer, Basic, raw token, X-API-Key) must be masked."""
    headers = [
        ("Authorization: Bearer jwt_secret_token_123", "Authorization: Bearer [REDACTED]"),
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "Authorization: Basic [REDACTED]"),
        ("Authorization: raw_custom_auth_token", "Authorization: [REDACTED]"),
        ("X-API-Key: custom_secret_api_key_456", "X-API-Key: [REDACTED]"),
    ]
    for raw, expected in headers:
        assert mask_secrets(raw) == expected


def test_mask_secrets_url_query_parameters() -> None:
    """Audit: Sensitive URL query parameters must be masked while non-sensitive parameters are preserved."""
    url = (
        "https://service.internal/v1/telemetry"
        "?token=secret_token_abc&apiKey=secret_key_def&accessKey=secret_acc_ghi"
        "&clientSecret=secret_cs_jkl&service=payment&status=degraded"
    )
    masked = mask_secrets(url)
    assert "secret_token_abc" not in masked
    assert "secret_key_def" not in masked
    assert "secret_acc_ghi" not in masked
    assert "secret_cs_jkl" not in masked
    assert "service=payment" in masked
    assert "status=degraded" in masked


def test_sanitize_dict_naming_variations() -> None:
    """Audit: Naming variations (camelCase, snake_case, kebab-case, mixed case) must be redacted."""
    data = {
        "X-API-Key": "secret_key_1",
        "clientSecret": "secret_cs_2",
        "privateKey": "secret_pk_3",
        "accessKey": "secret_ak_4",
        "Authorization": "secret_auth_5",
        "auth_token": "secret_token_6",
        "PassWord": "secret_pass_7",
        "passwd": "secret_pwd_8",
        # Non-sensitive keys
        "service_name": "payment",
        "key_metrics_count": 5,
        "is_active": True,
    }
    sanitized = sanitize_dict(data)

    assert sanitized["X-API-Key"] == REDACTED_PLACEHOLDER
    assert sanitized["clientSecret"] == REDACTED_PLACEHOLDER
    assert sanitized["privateKey"] == REDACTED_PLACEHOLDER
    assert sanitized["accessKey"] == REDACTED_PLACEHOLDER
    assert sanitized["Authorization"] == REDACTED_PLACEHOLDER
    assert sanitized["auth_token"] == REDACTED_PLACEHOLDER
    assert sanitized["PassWord"] == REDACTED_PLACEHOLDER
    assert sanitized["passwd"] == REDACTED_PLACEHOLDER

    # Non-sensitive keys and values preserved
    assert sanitized["service_name"] == "payment"
    assert sanitized["key_metrics_count"] == 5
    assert sanitized["is_active"] is True


def test_model_validation_rejects_invalid_values() -> None:
    """Audit: Models must reject invalid enum values, malformed timestamps, and invalid durations."""
    now = datetime.now(timezone.utc)

    # Invalid tool status
    with pytest.raises(ValidationError):
        ToolInvocationTrace(
            tool_name="get_mock_logs",
            duration_ms=10.0,
            status="unknown_status",  # type: ignore[arg-type]
        )

    # Invalid stage status
    with pytest.raises(ValidationError):
        StageTrace(
            stage_name="investigate",
            status="not_a_valid_status",  # type: ignore[arg-type]
            start_time=now,
            end_time=now,
            duration_ms=10.0,
        )

    # Malformed timestamp string
    with pytest.raises(ValidationError):
        StageTrace(
            stage_name="investigate",
            status=StageStatus.COMPLETED,
            start_time="not-a-timestamp",  # type: ignore[arg-type]
            end_time=now,
            duration_ms=10.0,
        )

    # Non-numeric duration
    with pytest.raises(ValidationError):
        ToolInvocationTrace(
            tool_name="get_mock_logs",
            duration_ms="non_numeric_duration",  # type: ignore[arg-type]
            status="success",
        )


def test_fail_safe_exception_text_never_leaked() -> None:
    """Audit: Exception text that itself contains secrets must NEVER be leaked in fallback output."""

    class LeakyExceptionObject:
        def __str__(self) -> str:
            raise RuntimeError("Database connection password=leaked_secret_password failed!")

    sanitized = sanitize_payload(LeakyExceptionObject())
    assert sanitized == UNSAFE_FALLBACK_PLACEHOLDER
    assert "leaked_secret_password" not in str(sanitized)
    assert "RuntimeError" not in str(sanitized)
