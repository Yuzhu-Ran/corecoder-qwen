"""Exercise real file and Shell tools through the graph using a scripted model."""

import shlex
import sys
from pathlib import Path

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.tools.bash import BashTool, get_tracked_cwd, set_tracked_cwd
from corecoder.tools.edit import EditFileTool
from corecoder.tools.read import ReadFileTool
from corecoder.tools.write import WriteFileTool


def test_write_read_edit_and_verify_python_through_graph(tmp_path):
    path = tmp_path / "calculator.py"
    command = (
        f"cd {shlex.quote(tmp_path.as_posix())} && {shlex.quote(Path(sys.executable).as_posix())} -c "
        + shlex.quote('from calculator import add; assert add(2, 3) == 5; print("CHECK_PASSED")')
    )
    script = [
        LLMResponse(tool_calls=[ToolCall("write", "write_file", {
            "file_path": str(path), "content": "def add(a, b):\n    return a - b\n",
        })]),
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": str(path)})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {
            "file_path": str(path), "old_string": "return a - b", "new_string": "return a + b",
        })]),
        LLMResponse(tool_calls=[ToolCall("verify", "bash", {"command": command})]),
        LLMResponse(content="Verified."),
    ]
    agent = Agent(
        llm=ScriptedLLM(script),
        tools=[WriteFileTool(), ReadFileTool(), EditFileTool(), BashTool()],
        permission=Permission(allow_all=True),
    )
    before_cwd = get_tracked_cwd()
    try:
        assert agent.chat("Fix and verify the temporary calculator.") == "Verified."
        results = {m["tool_call_id"]: m["content"] for m in agent.messages if m["role"] == "tool"}
        assert "return a - b" in results["read"]
        assert "return a + b" in results["edit"]
        assert results["verify"] == "CHECK_PASSED"
        assert path.read_text() == "def add(a, b):\n    return a + b\n"
    finally:
        set_tracked_cwd(before_cwd)
