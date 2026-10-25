"""Centralized configuration and environment settings for agentic-ai.

Provides a strongly typed, validated Pydantic model for platform settings:
- LLM Provider ("openrouter" or "ollama") and model selection.
- Provider credentials stored as SecretStr to prevent accidental leakage in
  repr, str, logs, or serialization.
- Incident-memory path (Path).
- Log level (INFO, DEBUG, etc.).
- Future API host and port (for Phase 15 Step 2+).
- MCP scenario default.

Settings can be instantiated from explicit arguments, read dynamically from
the environment via ``Settings.from_env()``, or accessed through the
``get_settings()`` helper.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from agentic_ai.llm.providers.ollama_provider import DEFAULT_OLLAMA_MODEL
from agentic_ai.llm.providers.openrouter_provider import DEFAULT_OPENROUTER_MODEL
from agentic_ai.tools.incident_memory import DEFAULT_MEMORY_PATH
from agentic_ai.tools.mock_tools import DEFAULT_SCENARIO, Scenario


class ConfigurationError(ValueError):
    """Raised when configuration validation fails."""


class LLMProviderType(str, Enum):
    """Supported LLM providers."""

    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseModel):
    """Strongly typed application configuration settings."""

    llm_provider: LLMProviderType = LLMProviderType.OPENROUTER
    openrouter_api_key: SecretStr | None = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    incident_memory_path: Path = Field(default_factory=lambda: Path(DEFAULT_MEMORY_PATH))
    log_level: LogLevel = LogLevel.INFO
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    mcp_scenario: Scenario = Scenario(DEFAULT_SCENARIO.value)
    database_url: SecretStr | None = None

    model_config = {"frozen": True}

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _validate_llm_provider(cls, value: Any) -> LLMProviderType:
        if isinstance(value, LLMProviderType):
            return value
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in ("openrouter", "ollama"):
                return LLMProviderType(cleaned)
            raise ValueError(f"Unknown LLM_PROVIDER '{value}'. Expected 'openrouter' or 'ollama'.")
        raise ValueError(
            f"Invalid LLM_PROVIDER type '{type(value).__name__}'. Expected 'openrouter' or 'ollama'."
        )

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def _validate_openrouter_api_key(cls, value: Any) -> SecretStr | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value
        if not isinstance(value, str):
            raise ValueError("OPENROUTER_API_KEY must be a string or None.")
        cleaned = value.strip()
        return SecretStr(cleaned) if cleaned else None

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def _validate_openrouter_model(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("OPENROUTER_MODEL must be a non-empty string.")
        return value.strip()

    @field_validator("ollama_model", mode="before")
    @classmethod
    def _validate_ollama_model(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("OLLAMA_MODEL must be a non-empty string.")
        return value.strip()

    @field_validator("incident_memory_path", mode="before")
    @classmethod
    def _validate_incident_memory_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str) and value.strip():
            return Path(value.strip())
        raise ValueError("AGENTIC_MEMORY_PATH must be a non-empty path string or Path object.")

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: Any) -> LogLevel:
        if isinstance(value, LogLevel):
            return value
        if isinstance(value, str):
            cleaned = value.strip().upper()
            if cleaned in LogLevel.__members__:
                return LogLevel[cleaned]
            allowed = ", ".join(sorted(LogLevel.__members__.keys()))
            raise ValueError(f"Invalid LOG_LEVEL '{value}'. Expected one of: {allowed}.")
        raise ValueError(f"Invalid LOG_LEVEL type '{type(value).__name__}'. Expected string.")

    @field_validator("api_host", mode="before")
    @classmethod
    def _validate_api_host(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("API_HOST must be a non-empty string.")
        return value.strip()

    @field_validator("api_port", mode="before")
    @classmethod
    def _validate_api_port(cls, value: Any) -> int:
        try:
            port_num = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"API_PORT must be an integer between 1 and 65535, got '{value}'.")
        if not (1 <= port_num <= 65535):
            raise ValueError(f"API_PORT must be between 1 and 65535, got {port_num}.")
        return port_num

    @field_validator("mcp_scenario", mode="before")
    @classmethod
    def _validate_mcp_scenario(cls, value: Any) -> Scenario:
        if isinstance(value, Scenario):
            return value
        if isinstance(value, str):
            cleaned = value.strip().lower()
            try:
                return Scenario(cleaned)
            except ValueError:
                pass
        valid = ", ".join(f"'{s.value}'" for s in Scenario)
        raise ValueError(f"Invalid AGENTIC_MCP_SCENARIO '{value}'. Expected one of: {valid}.")

    @field_validator("database_url", mode="before")
    @classmethod
    def _validate_database_url(cls, value: Any) -> SecretStr | None:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            raw = value.get_secret_value()
        elif isinstance(value, str):
            raw = value.strip()
        else:
            raise ValueError("DATABASE_URL must be a string, SecretStr, or None.")

        if not raw:
            return None

        if raw.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL uses 'postgresql+asyncpg://', which is an async scheme "
                "incompatible with the synchronous driver. Use 'postgresql://' instead."
            )

        if not (raw.startswith("postgresql://") or raw.startswith("postgres://")):
            raise ValueError(
                "DATABASE_URL must use postgresql:// or postgres:// scheme "
                "(e.g. 'postgresql://user:pass@host:port/dbname')."
            )
        return SecretStr(raw)

    def validate_provider_ready(self) -> None:
        """Validate that the active LLM provider has all required credentials configured.

        Raises:
            ConfigurationError: If the active provider is missing required credentials.
        """
        if self.llm_provider == LLMProviderType.OPENROUTER:
            if not self.openrouter_api_key or not self.openrouter_api_key.get_secret_value():
                raise ConfigurationError(
                    "OPENROUTER_API_KEY is not set or empty. "
                    "Copy .env.example to .env and configure a valid key from https://openrouter.ai/keys."
                )

    def is_provider_ready(self) -> bool:
        """Check whether the active LLM provider has required credentials without raising."""
        try:
            self.validate_provider_ready()
            return True
        except ConfigurationError:
            return False

    def get_openrouter_api_key_value(self) -> str | None:
        """Return the raw secret value of OPENROUTER_API_KEY if present."""
        if self.openrouter_api_key:
            return self.openrouter_api_key.get_secret_value()
        return None

    def get_database_url_value(self) -> str | None:
        """Return the raw secret value of DATABASE_URL if present."""
        if self.database_url:
            return self.database_url.get_secret_value()
        return None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build Settings from an environment mapping, defaulting to os.environ.

        Translates standard environment variables:
        - LLM_PROVIDER
        - OPENROUTER_API_KEY
        - OPENROUTER_MODEL
        - OLLAMA_MODEL
        - AGENTIC_MEMORY_PATH
        - LOG_LEVEL
        - API_HOST
        - API_PORT
        - AGENTIC_MCP_SCENARIO
        - DATABASE_URL
        """
        source = os.environ if env is None else env
        kwargs: dict[str, Any] = {}

        mapping = {
            "LLM_PROVIDER": "llm_provider",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OPENROUTER_MODEL": "openrouter_model",
            "OLLAMA_MODEL": "ollama_model",
            "AGENTIC_MEMORY_PATH": "incident_memory_path",
            "LOG_LEVEL": "log_level",
            "API_HOST": "api_host",
            "API_PORT": "api_port",
            "AGENTIC_MCP_SCENARIO": "mcp_scenario",
            "DATABASE_URL": "database_url",
        }

        for env_key, field_name in mapping.items():
            if env_key in source:
                kwargs[field_name] = source[env_key]

        try:
            return cls(**kwargs)
        except ValidationError as err:
            errors = [f"{e['loc'][0]}: {e['msg']}" for e in err.errors()]
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            ) from err


def get_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Return platform settings read dynamically from the environment.

    Reads fresh from environment so that monkeypatched variables in tests
    are immediately reflected without caching or import-order side-effects.
    """
    return Settings.from_env(env=env)
