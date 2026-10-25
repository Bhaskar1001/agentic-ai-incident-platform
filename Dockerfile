# ==============================================================================
# Multi-stage production Dockerfile for AI Engineering & Incident Resolution Platform
# ==============================================================================

# Stage 1: Build virtual environment with uv
FROM ghcr.io/astral-sh/uv:0.12.10 AS uv-bin
FROM python:3.10-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy uv binaries from official release image
COPY --from=uv-bin /uv /uvx /bin/

# Copy dependency specifications for cached layer installation
COPY pyproject.toml uv.lock ./

# Install locked dependencies into /app/.venv without development packages or project code
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source tree and install project package
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev


# Stage 2: Minimal non-root runtime image
FROM python:3.10-slim AS runtime

# Standard production Python runtime environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    AGENTIC_MEMORY_PATH="/app/.agentic_memory/incidents.jsonl"

WORKDIR /app

# Create non-root service user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment and application code from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src

# Create writable domain memory directory for the non-root application user
RUN mkdir -p /app/.agentic_memory && \
    chown -R appuser:appgroup /app/.agentic_memory

# Switch to non-root user
USER appuser

# Expose default HTTP API port
EXPOSE 8000

# Liveness health check using Python standard library (no external curl/wget required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('API_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')" || exit 1

# Start Uvicorn ASGI server with signal forwarding
CMD ["sh", "-c", "exec uvicorn agentic_ai.api.app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]

