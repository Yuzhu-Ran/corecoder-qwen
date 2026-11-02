"""CLI model selection and saved-session restoration without network calls."""

from argparse import Namespace

import pytest

from corecoder import cli
from corecoder.config import Config
from corecoder.langchain_llm import LangChainLLM
from corecoder.llm import LLM, LiteLLM


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("langchain", LangChainLLM), ("openai", LangChainLLM), ("legacy", LLM), ("litellm", LiteLLM)],
)
def test_cli_chooses_expected_model_backend(provider, expected):
    model = cli._create_llm(Config(model="qwen-flash", api_key="test", provider=provider))
    assert type(model) is expected
    assert model.model == "qwen-flash"


@pytest.mark.parametrize(("override", "expected"), [(None, "saved-model"), ("qwen-flash", "qwen-flash")])
def test_cli_restores_model_on_real_langchain_backend(monkeypatch, override, expected):
    """Session model changes reach the adapter, while -m still wins."""
    config = Config(model="env-model", api_key="test", base_url="https://example.invalid/v1")
    args = Namespace(
        demo=False, model=override, base_url=None, api_key=None,
        yes=False, prompt="test", resume="saved-session",
    )
    monkeypatch.setattr(cli, "_parse_args", lambda: args)
    monkeypatch.setattr(cli.Config, "from_env", lambda: config)
    monkeypatch.setattr(cli, "load_mcp_tools", list)
    monkeypatch.setattr(cli, "load_hooks", lambda: None)
    monkeypatch.setattr(cli, "load_session", lambda sid: ([{"role": "user", "content": "earlier"}], "saved-model"))
    observed = []
    monkeypatch.setattr(cli, "_run_once", lambda agent, prompt: observed.append(agent))

    cli.main()

    assert len(observed) == 1
    agent = observed[0]
    assert isinstance(agent.llm, LangChainLLM)
    assert agent.llm.model == expected
    assert agent.llm._options()["model"] == expected
    assert not agent.permission.allow_all
    assert agent.messages == [{"role": "user", "content": "earlier"}]
