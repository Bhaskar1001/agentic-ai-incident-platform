import pytest

from agentic_ai.tools.incident_memory import MEMORY_PATH_ENV


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Never let a test read or write the real incident memory file."""
    monkeypatch.setenv(MEMORY_PATH_ENV, str(tmp_path / "memory.jsonl"))
