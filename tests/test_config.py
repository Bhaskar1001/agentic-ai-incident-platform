"""Tests for centralized configuration and startup validation (Phase 15 Step 1).

Validates:
1. Default settings match existing repository defaults.
2. Environment-variable overrides across all supported fields.
3. Case-insensitivity and whitespace trimming.
4. Actionable error reporting for invalid providers, log levels, ports, and scenarios.
5. Secret masking in str, repr, model dumps, and validation errors.
6. Provider readiness validation for live vs mock/test usage.
7. Dynamic reflection of environment monkeypatching in get_settings().
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_ai.config import (
    ConfigurationError,
    LLMProviderType,
    LogLevel,
    Settings,
    get_settings,
)
from agentic_ai.llm.providers.ollama_provider import DEFAULT_OLLAMA_MODEL
from agentic_ai.llm.providers.openrouter_provider import DEFAULT_OPENROUTER_MODEL
from agentic_ai.tools.incident_memory import DEFAULT_MEMORY_PATH
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario


def test_default_settings() -> None:
    """Default Settings must preserve all existing repository conventions."""
    settings = Settings()

    assert settings.llm_provider == LLMProviderType.OPENROUTER
    assert settings.openrouter_api_key is None
    assert settings.openrouter_model == DEFAULT_OPENROUTER_MODEL
    assert settings.ollama_model == DEFAULT_OLLAMA_MODEL
    assert settings.incident_memory_path == Path(DEFAULT_MEMORY_PATH)
    assert settings.log_level == LogLevel.INFO
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.mcp_scenario == DEFAULT_SCENARIO


def test_settings_is_frozen() -> None:
    """Settings model must be immutable to prevent runtime configuration drift."""
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.api_port = 9000  # type: ignore[misc]


def test_environment_variable_overrides() -> None:
    """All supported environment variables override defaults correctly."""
    custom_env = {
        "LLM_PROVIDER": "ollama",
        "OPENROUTER_API_KEY": "sk-ant-test-override-key",
        "OPENROUTER_MODEL": "custom/nemotron-preview",
        "OLLAMA_MODEL": "mistral:7b",
        "AGENTIC_MEMORY_PATH": "custom_path/incidents.jsonl",
        "LOG_LEVEL": "DEBUG",
        "API_HOST": "127.0.0.1",
        "API_PORT": "9090",
        "AGENTIC_MCP_SCENARIO": "healthy",
    }

    settings = Settings.from_env(custom_env)

    assert settings.llm_provider == LLMProviderType.OLLAMA
    assert settings.get_openrouter_api_key_value() == "sk-ant-test-override-key"
    assert settings.openrouter_model == "custom/nemotron-preview"
    assert settings.ollama_model == "mistral:7b"
    assert settings.incident_memory_path == Path("custom_path/incidents.jsonl")
    assert settings.log_level == LogLevel.DEBUG
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 9090
    assert settings.mcp_scenario == Scenario.HEALTHY


def test_case_insensitivity_and_whitespace_trimming() -> None:
    """String enums and credentials are normalized and whitespace-trimmed."""
    env = {
        "LLM_PROVIDER": "  OLLAMA  ",
        "LOG_LEVEL": "  debug  ",
        "API_HOST": "  localhost  ",
        "AGENTIC_MCP_SCENARIO": " UPSTREAM_DEPENDENCY_FAILURE ",
        "OPENROUTER_API_KEY": "   ",  # whitespace only -> None
    }

    settings = Settings.from_env(env)

    assert settings.llm_provider == LLMProviderType.OLLAMA
    assert settings.log_level == LogLevel.DEBUG
    assert settings.api_host == "localhost"
    assert settings.mcp_scenario == Scenario.UPSTREAM_DEPENDENCY_FAILURE
    assert settings.openrouter_api_key is None


def test_invalid_llm_provider_raises_actionable_error() -> None:
    """Invalid provider names yield clear errors with expected values."""
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env({"LLM_PROVIDER": "unsupported_provider"})

    err_msg = str(exc_info.value)
    assert "Unknown LLM_PROVIDER" in err_msg
    assert "openrouter" in err_msg
    assert "ollama" in err_msg


def test_invalid_log_level_raises_actionable_error() -> None:
    """Invalid log level yields clear error listing allowed levels."""
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env({"LOG_LEVEL": "VERBOSE"})

    err_msg = str(exc_info.value)
    assert "Invalid LOG_LEVEL" in err_msg
    assert "DEBUG" in err_msg
    assert "INFO" in err_msg


def test_invalid_api_port_raises_actionable_error() -> None:
    """Non-integer or out-of-range ports are rejected with actionable errors."""
    with pytest.raises(ConfigurationError) as exc_info_nan:
        Settings.from_env({"API_PORT": "not-a-port"})
    assert "API_PORT must be an integer" in str(exc_info_nan.value)

    with pytest.raises(ConfigurationError) as exc_info_zero:
        Settings.from_env({"API_PORT": "0"})
    assert "API_PORT must be between 1 and 65535" in str(exc_info_zero.value)

    with pytest.raises(ConfigurationError) as exc_info_high:
        Settings.from_env({"API_PORT": "70000"})
    assert "API_PORT must be between 1 and 65535" in str(exc_info_high.value)


def test_invalid_mcp_scenario_raises_actionable_error() -> None:
    """Invalid scenario names yield descriptive error."""
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env({"AGENTIC_MCP_SCENARIO": "datacenter_fire"})

    err_msg = str(exc_info.value)
    assert "Invalid AGENTIC_MCP_SCENARIO" in err_msg
    assert "connection_pool_exhaustion" in err_msg


def test_secret_redaction_in_repr_str_and_dump() -> None:
    """Provider API keys are never exposed in string representations or dumps."""
    secret_value = "sk-ant-api03-live-secret-abcdefg1234567890"
    settings = Settings(openrouter_api_key=secret_value)

    repr_str = repr(settings)
    str_val = str(settings)
    dump_dict = settings.model_dump()
    json_str = settings.model_dump_json()

    # Raw secret must never appear anywhere in representations
    assert secret_value not in repr_str
    assert secret_value not in str_val
    assert secret_value not in str(dump_dict)
    assert secret_value not in json_str

    # SecretStr masking confirmation
    assert "**********" in repr_str
    assert "**********" in json_str

    # Explicit getter returns original value
    assert settings.get_openrouter_api_key_value() == secret_value


def test_secret_not_leaked_in_validation_errors() -> None:
    """Validation errors on other fields must never echo the secret token."""
    secret_value = "sk-live-super-secret-key-to-protect"

    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_env(
            {
                "OPENROUTER_API_KEY": secret_value,
                "API_PORT": "invalid-port",
            }
        )

    assert secret_value not in str(exc_info.value)


def test_provider_readiness_validation() -> None:
    """Validate provider credentials only when explicitly requested for live calls."""
    # OpenRouter without key is not live-ready
    openrouter_no_key = Settings(llm_provider=LLMProviderType.OPENROUTER, openrouter_api_key=None)
    assert openrouter_no_key.is_provider_ready() is False
    with pytest.raises(ConfigurationError) as exc_info:
        openrouter_no_key.validate_provider_ready()
    assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)

    # OpenRouter with key is live-ready
    openrouter_with_key = Settings(
        llm_provider=LLMProviderType.OPENROUTER,
        openrouter_api_key="sk-test-key",
    )
    assert openrouter_with_key.is_provider_ready() is True
    openrouter_with_key.validate_provider_ready()  # does not raise

    # Ollama does not require OpenRouter API key
    ollama_settings = Settings(llm_provider=LLMProviderType.OLLAMA)
    assert ollama_settings.is_provider_ready() is True
    ollama_settings.validate_provider_ready()  # does not raise


def test_mock_test_execution_without_live_credentials() -> None:
    """Default Settings does not raise errors or force live credentials for tests."""
    # Default instantiation must always succeed without throwing
    settings = Settings()
    assert settings.openrouter_api_key is None
    assert settings.llm_provider == LLMProviderType.OPENROUTER


def test_dynamic_get_settings_reflects_monkeypatch(monkeypatch) -> None:
    """get_settings() reads dynamically so test monkeypatching works without import caching."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("API_PORT", "8888")

    settings = get_settings()

    assert settings.llm_provider == LLMProviderType.OLLAMA
    assert settings.log_level == LogLevel.WARNING
    assert settings.api_port == 8888
