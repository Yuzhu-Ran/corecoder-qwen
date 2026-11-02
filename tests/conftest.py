"""Shared pytest fixtures and helpers."""

import pytest

from corecoder import Config, checkpoints, cli
from corecoder import config as config_module
from corecoder import llm as llm_module
from corecoder.tools import ALL_TOOLS


@pytest.fixture(autouse=True)
def _config_isolated(monkeypatch, request):
    """Unit tests never load a developer's real .env or enable remote tracing."""
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    if request.path.name != "test_live_smoke.py":
        for name in (
            "CORECODER_MODEL", "CORECODER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
            "OPENAI_BASE_URL", "CORECODER_BASE_URL", "CORECODER_PROVIDER", "CORECODER_MAX_TOKENS",
            "CORECODER_MAX_CONTEXT", "CORECODER_TEMPERATURE",
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _checkpoints_isolated(tmp_path, monkeypatch):
    """Checkpoint persistence must never touch the real ~/.corecoder."""
    monkeypatch.setattr(checkpoints, "CHECKPOINTS_FILE", tmp_path / "checkpoints.json")
    checkpoints.clear()
    yield
    checkpoints.clear()


@pytest.fixture(autouse=True)
def _pricing_isolated(tmp_path, monkeypatch):
    """Cost estimates must never read the real ~/.corecoder/pricing.json."""
    monkeypatch.setattr(llm_module, "PRICING_FILE", tmp_path / "pricing.json")


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None


def repl_with(monkeypatch, agent, inputs, config=None):
    """Drive the real REPL with a scripted list of inputs. Returns the config in use."""
    it = iter(inputs)
    monkeypatch.setattr(cli, "pt_prompt", lambda *a, **k: next(it))
    config = config or Config.from_env()
    cli._repl(agent, config)
    return config
