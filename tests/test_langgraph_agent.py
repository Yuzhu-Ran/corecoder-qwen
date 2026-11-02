"""Offline regressions for the LangGraph agent's existing public behavior."""

import shlex
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import pytest
from langgraph.graph.state import CompiledStateGraph

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.permissions import Permission
from corecoder.tools.agent import AgentTool
from corecoder.tools.base import Tool
from corecoder.tools.bash import BashTool, get_tracked_cwd, set_tracked_cwd
from corecoder.tools.read import ReadFileTool
from corecoder.tools.write import WriteFileTool


class _RecordingLLM(ScriptedLLM):
    def __init__(self, turns):
        super().__init__(list(turns))
        self.requests = []

    def chat(self, messages, tools=None, on_token=None, on_reasoning=None):
        self.requests.append(deepcopy(messages))
        return super().chat(messages, tools, on_token, on_reasoning)


class _CounterTool(Tool):
    name = "counter"
    description = "Return a deterministic counter without external side effects."
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def __init__(self):
        self.calls = 0

    def execute(self):
        self.calls += 1
        return str(self.calls)


class _InterruptTool(Tool):
    name = "interrupt"
    description = "Simulate Ctrl+C while a tool is running."
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def execute(self):
        raise KeyboardInterrupt


class _TypeErrorTool(Tool):
    name = "internal_type_error"
    description = "Fail inside the implementation, after arguments are accepted."
    parameters: ClassVar[dict] = {"type": "object", "properties": {}, "required": []}

    def execute(self):
        raise TypeError("failure inside the tool")


class _RecordingHooks:
    """In-memory hooks exercise ordering without spawning shell processes."""

    def __init__(self, events, blocked_paths=()):
        self.events = events
        self.blocked_paths = set(blocked_paths)

    def run_pre(self, tool_name, tool_input):
        path = tool_input.get("file_path")
        self.events.append(("pre", tool_name, path))
        if path in self.blocked_paths:
            return "Blocked by hook: blocked for this test"
        return None

    def run_post(self, tool_name, tool_input, tool_response):
        self.events.append(("post", tool_name, tool_input.get("file_path")))


def _turn(*calls):
    return LLMResponse(tool_calls=list(calls))


def _write(call_id, path, content="written by the test\n"):
    return ToolCall(
        id=call_id,
        name="write_file",
        arguments={"file_path": str(path), "content": content},
    )


def _results(agent):
    return {
        message["tool_call_id"]: message["content"]
        for message in agent.messages
        if message.get("role") == "tool"
    }


def _assert_paired_history(messages):
    requested = Counter(
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    )
    answered = Counter(
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    )
    assert requested == answered
    assert all(count == 1 for count in answered.values())


def test_agent_exposes_a_real_compiled_state_graph():
    agent = Agent(llm=_RecordingLLM([]), tools=[])

    assert isinstance(agent.graph, CompiledStateGraph)
    assert {"prepare", "model", "authorize", "tools", "compact", "limit"} <= set(
        agent.graph.get_graph().nodes
    )


def test_zero_round_budget_never_calls_the_model():
    llm = _RecordingLLM([LLMResponse(content="must not be requested")])
    agent = Agent(llm=llm, tools=[], max_rounds=0)

    result = agent.chat("a zero-budget task")

    assert "maximum tool-call rounds" in result
    assert llm.requests == []
    assert agent.messages == [{"role": "user", "content": "a zero-budget task"}]


def test_one_round_budget_executes_the_batch_without_an_extra_model_call():
    counter = _CounterTool()
    llm = _RecordingLLM([
        _turn(ToolCall(id="last-round", name=counter.name, arguments={})),
        LLMResponse(content="an extra request would exceed the budget"),
    ])
    agent = Agent(llm=llm, tools=[counter], max_rounds=1)

    result = agent.chat("use the counter")

    assert "maximum tool-call rounds" in result
    assert len(llm.requests) == 1
    assert counter.calls == 1
    assert _results(agent)["last-round"] == "1"
    _assert_paired_history(agent.messages)


def test_plain_text_on_the_last_allowed_round_is_returned_normally():
    llm = _RecordingLLM([LLMResponse(content="done")])
    agent = Agent(llm=llm, tools=[], max_rounds=1)

    assert agent.chat("finish without tools") == "done"
    assert len(llm.requests) == 1
    assert agent.messages[-1] == {"role": "assistant", "content": "done"}


@pytest.mark.parametrize("finish_with_text", [False, True])
def test_model_round_budget_is_not_the_graph_default_step_limit(finish_with_text):
    counter = _CounterTool()
    tool_rounds = 30
    turns = [
        _turn(ToolCall(id=f"round-{i}", name=counter.name, arguments={}))
        for i in range(tool_rounds)
    ]
    if finish_with_text:
        turns.append(LLMResponse(content="all rounds completed"))
    llm = _RecordingLLM(turns)
    agent = Agent(llm=llm, tools=[counter], max_rounds=len(turns))

    result = agent.chat("run more than twenty-five model rounds")

    assert counter.calls == tool_rounds
    assert len(llm.requests) == len(turns)
    assert len(_results(agent)) == tool_rounds
    if finish_with_text:
        assert result == "all rounds completed"
    else:
        assert "maximum tool-call rounds" in result
    _assert_paired_history(agent.messages)


def test_bash_cwd_survives_graph_nodes_and_the_next_chat(tmp_path):
    target = tmp_path / "project with spaces"
    target.mkdir()
    marker = "cwd survived the graph boundary"
    (target / "marker.txt").write_text(marker, encoding="utf-8")
    llm = _RecordingLLM([
        _turn(ToolCall(id="change-directory", name="bash", arguments={"command": f"cd {shlex.quote(target.as_posix())}"})),
        _turn(ToolCall(id="same-chat-read", name="bash", arguments={"command": "cat marker.txt"})),
        LLMResponse(content="first chat completed"),
        _turn(ToolCall(id="next-chat-read", name="bash", arguments={"command": "cat marker.txt"})),
        LLMResponse(content="second chat completed"),
    ])
    previous_cwd = get_tracked_cwd()
    set_tracked_cwd(str(tmp_path))
    try:
        agent = Agent(llm=llm, tools=[BashTool()], permission=Permission(allow_all=True))
        assert agent.chat("change directory, then read the marker") == "first chat completed"
        assert _results(agent)["same-chat-read"] == marker
        assert Path(get_tracked_cwd()).resolve() == target.resolve()

        assert agent.chat("read the marker again") == "second chat completed"
        assert _results(agent)["next-chat-read"] == marker
        assert Path(get_tracked_cwd()).resolve() == target.resolve()
    finally:
        set_tracked_cwd(previous_cwd)


def test_child_graph_cwd_does_not_leak_into_the_parent_graph(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent-project"
    child_dir = tmp_path / "child-project"
    parent_dir.mkdir()
    child_dir.mkdir()
    llm = _RecordingLLM([
        _turn(ToolCall(id="delegate", name="agent", arguments={"task": "work in the child directory"})),
        _turn(ToolCall(id="child-cd", name="bash", arguments={"command": f"cd {shlex.quote(child_dir.as_posix())}"})),
        _turn(ToolCall(id="child-pwd", name="bash", arguments={"command": "pwd"})),
        LLMResponse(content="child completed"),
        _turn(ToolCall(id="parent-pwd", name="bash", arguments={"command": "pwd"})),
        LLMResponse(content="parent completed"),
    ])
    instances = []
    original_init = Agent.__init__

    def capture_instance(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(Agent, "__init__", capture_instance)
    previous_cwd = get_tracked_cwd()
    set_tracked_cwd(str(parent_dir))
    try:
        parent = Agent(
            llm=llm,
            tools=[AgentTool(), BashTool()],
            permission=Permission(allow_all=True),
        )

        assert parent.chat("delegate a directory change, then check the parent cwd") == "parent completed"

        assert len(instances) == 2
        assert instances[0] is parent
        child = instances[1]
        assert isinstance(parent.graph, CompiledStateGraph)
        assert isinstance(child.graph, CompiledStateGraph)
        assert child.llm is parent.llm
        assert _results(child)["child-pwd"].replace("\\", "/").rstrip("/").endswith("/child-project")
        assert _results(parent)["parent-pwd"].replace("\\", "/").rstrip("/").endswith("/parent-project")
        assert _results(parent)["delegate"] == "[Sub-agent completed]\nchild completed"
        assert Path(get_tracked_cwd()).resolve() == parent_dir.resolve()
        _assert_paired_history(child.messages)
        _assert_paired_history(parent.messages)
    finally:
        set_tracked_cwd(previous_cwd)


def test_parallel_agent_tool_inherits_the_parent_cwd(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent-project"
    parent_dir.mkdir()
    counter = _CounterTool()
    # The sibling never calls the model, so the scripted parent/child turns
    # remain deterministic regardless of which pool worker finishes first.
    llm = _RecordingLLM([
        _turn(
            ToolCall(id="delegate", name="agent", arguments={"task": "report the inherited cwd"}),
            ToolCall(id="sibling", name=counter.name, arguments={}),
        ),
        _turn(ToolCall(id="child-pwd", name="bash", arguments={"command": "pwd"})),
        LLMResponse(content="child completed"),
        _turn(ToolCall(id="parent-pwd", name="bash", arguments={"command": "pwd"})),
        LLMResponse(content="parent completed"),
    ])
    instances = []
    original_init = Agent.__init__

    def capture_instance(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(Agent, "__init__", capture_instance)
    previous_cwd = get_tracked_cwd()
    set_tracked_cwd(str(parent_dir))
    try:
        parent = Agent(
            llm=llm,
            tools=[AgentTool(), BashTool(), counter],
            permission=Permission(allow_all=True),
        )

        assert parent.chat("delegate and run the counter in the same tool batch") == "parent completed"

        assert len(instances) == 2
        child = instances[1]
        assert child.llm is parent.llm
        assert isinstance(child.graph, CompiledStateGraph)
        assert _results(child)["child-pwd"].replace("\\", "/").rstrip("/").endswith("/parent-project")
        assert _results(parent)["parent-pwd"].replace("\\", "/").rstrip("/").endswith("/parent-project")
        assert _results(parent)["sibling"] == "1"
        assert counter.calls == 1
        assert Path(get_tracked_cwd()).resolve() == parent_dir.resolve()
        _assert_paired_history(child.messages)
        _assert_paired_history(parent.messages)
    finally:
        set_tracked_cwd(previous_cwd)


@pytest.mark.parametrize("parallel", [False, True])
def test_tool_interrupt_repairs_history_and_the_next_chat_can_continue(parallel):
    counter = _CounterTool()
    calls = [ToolCall(id="interrupted", name="interrupt", arguments={})]
    if parallel:
        calls.insert(0, ToolCall(id="sibling", name=counter.name, arguments={}))
    llm = _RecordingLLM([_turn(*calls), LLMResponse(content="recovered")])
    agent = Agent(llm=llm, tools=[counter, _InterruptTool()])

    with pytest.raises(KeyboardInterrupt):
        agent.chat("simulate Ctrl+C")

    assert _results(agent)["interrupted"] == "[interrupted]"
    _assert_paired_history(agent.messages)
    assert agent.chat("continue after the interruption") == "recovered"
    _assert_paired_history(llm.requests[-1])
    _assert_paired_history(agent.messages)
    assert counter.calls == int(parallel)


def test_graph_plan_mode_still_rejects_preapproved_writes(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("read-only result", encoding="utf-8")
    output = tmp_path / "must-not-be-written.txt"
    events = []
    llm = _RecordingLLM([
        _turn(
            _write("blocked-write", output),
            ToolCall(id="allowed-read", name="read_file", arguments={"file_path": str(note)}),
        ),
        LLMResponse(content="the plan is ready"),
    ])
    agent = Agent(
        llm=llm,
        tools=[WriteFileTool(), ReadFileTool()],
        permission=Permission(allow_all=True),
        hooks=_RecordingHooks(events),
    )
    agent.plan_mode = True

    assert agent.chat("investigate without changing files") == "the plan is ready"

    assert not output.exists()
    assert "Plan mode is on" in _results(agent)["blocked-write"]
    assert "read-only result" in _results(agent)["allowed-read"]
    assert events == [
        ("pre", "write_file", str(output)),
        ("pre", "read_file", str(note)),
        ("post", "read_file", str(note)),
    ]
    _assert_paired_history(agent.messages)


@pytest.mark.parametrize("plan_mode", [False, True])
def test_graph_hook_veto_runs_before_plan_mode_and_permission(tmp_path, plan_mode):
    output = tmp_path / "blocked.txt"
    events = []

    def ask(tool_name, arguments):
        events.append(("permission", tool_name, arguments.get("file_path")))
        return "once"

    agent = Agent(
        llm=_RecordingLLM([_turn(_write("blocked", output)), LLMResponse(content="noted")]),
        tools=[WriteFileTool()],
        permission=Permission(ask=ask),
        hooks=_RecordingHooks(events, blocked_paths=[str(output)]),
    )
    agent.plan_mode = plan_mode

    assert agent.chat("try the blocked action") == "noted"

    assert not output.exists()
    assert _results(agent)["blocked"] == "Blocked by hook: blocked for this test"
    assert events == [("pre", "write_file", str(output))]


def test_graph_parallel_permissions_decide_each_call_before_execution(tmp_path):
    denied = tmp_path / "denied.txt"
    allowed = tmp_path / "allowed.txt"
    events = []

    def ask(tool_name, arguments):
        path = arguments["file_path"]
        events.append(("permission", tool_name, path))
        return "deny" if path == str(denied) else "once"

    agent = Agent(
        llm=_RecordingLLM([
            _turn(_write("denied", denied), _write("allowed", allowed)),
            LLMResponse(content="decisions respected"),
        ]),
        tools=[WriteFileTool()],
        permission=Permission(ask=ask),
        hooks=_RecordingHooks(events),
    )

    assert agent.chat("request two different writes") == "decisions respected"

    assert not denied.exists()
    assert allowed.read_text(encoding="utf-8") == "written by the test\n"
    assert "Permission denied" in _results(agent)["denied"]
    assert events == [
        ("pre", "write_file", str(denied)),
        ("permission", "write_file", str(denied)),
        ("pre", "write_file", str(allowed)),
        ("permission", "write_file", str(allowed)),
        ("post", "write_file", str(allowed)),
    ]
    _assert_paired_history(agent.messages)


def test_graph_does_not_execute_a_tool_outside_the_agents_tool_set(tmp_path):
    output = tmp_path / "unregistered.txt"
    agent = Agent(
        llm=_RecordingLLM([_turn(_write("unknown", output)), LLMResponse(content="not available")]),
        tools=[ReadFileTool()],
        permission=Permission(allow_all=True),
    )

    assert agent.chat("try a tool that is not available") == "not available"

    assert not output.exists()
    assert "unknown tool 'write_file'" in _results(agent)["unknown"]
    _assert_paired_history(agent.messages)


def test_graph_preserves_bad_argument_and_internal_error_classification():
    tool = _TypeErrorTool()
    agent = Agent(
        llm=_RecordingLLM([
            _turn(
                ToolCall(id="bad-arguments", name=tool.name, arguments={"unexpected": True}),
                ToolCall(id="internal-error", name=tool.name, arguments={}),
            ),
            LLMResponse(content="both failures were returned"),
        ]),
        tools=[tool],
    )

    assert agent.chat("classify tool errors") == "both failures were returned"

    results = _results(agent)
    assert "bad arguments" in results["bad-arguments"]
    assert "Error executing internal_type_error" in results["internal-error"]
    assert "bad arguments" not in results["internal-error"]
    _assert_paired_history(agent.messages)
