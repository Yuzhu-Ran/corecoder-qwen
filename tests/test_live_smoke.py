"""Opt-in live smoke test: one real end-to-end agent turn.

Skipped by default. Run it with a real key (OpenRouter works out of the box):

    CORECODER_LIVE_SMOKE=1 \
    OPENAI_API_KEY=your-api-key \
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    CORECODER_SMOKE_MODEL=qwen-flash \
    pytest tests/test_live_smoke.py -v

Override the model with CORECODER_SMOKE_MODEL (pick something cheap).
The point is not coverage — the unit suite has that. It is proof that the
whole loop (stream, tool call, tool result, final answer) works against a
live OpenAI-compatible endpoint.
"""

import os
from pathlib import Path

import pytest

from corecoder.agent import Agent
from corecoder.langchain_llm import LangChainLLM
from corecoder.tools.read import ReadFileTool

pytestmark = pytest.mark.skipif(
    not os.environ.get("CORECODER_LIVE_SMOKE"),
    reason="live smoke is opt-in: set CORECODER_LIVE_SMOKE=1 with a real API key",
)

MAGIC = "saffron-kettledrum-42"


def test_agent_reads_a_file_and_answers(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text(f"The magic phrase is {MAGIC}.\n")

    llm = LangChainLLM(
        model=os.environ.get("CORECODER_SMOKE_MODEL", "qwen-flash"),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=90,
    )
    class FixtureReadTool(ReadFileTool):
        """Expose only the generated test file to the external model."""

        def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str:
            if Path(file_path).resolve() != fixture.resolve():
                return "Error: this test only permits the generated fixture file"
            return super().execute(file_path, offset, limit)

    agent = Agent(llm=llm, tools=[FixtureReadTool()], max_rounds=6)

    tools_used: list[str] = []
    reply = agent.chat(
        f"Read the file at {fixture} with the read_file tool, then tell me "
        "the magic phrase it contains. Answer with just the phrase.",
        on_tool=lambda name, _args: tools_used.append(name),
    )

    assert "read_file" in tools_used, f"model never used the read tool: {reply!r}"
    assert MAGIC in reply, f"magic phrase missing from the reply: {reply!r}"
