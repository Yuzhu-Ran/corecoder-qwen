"""LangGraph workflow for model calls, authorization and LangChain tools.

The public chat/history interface stays compatible with CoreCoder's CLI,
session files and scripted demos. Graph state is per invocation; saved file
snapshots and conversation persistence remain separate concerns.
"""

import concurrent.futures
import inspect

from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .context import ContextManager
from .langchain_tools import adapt_tools
from .llm import LLM
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.todo import TodoWriteTool


class AgentState(TypedDict, total=False):
    messages: list[dict]
    rounds: int
    response: object
    permissions: list[str | None]
    cwd: str | None
    output: str


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        permission=None,
        hooks=None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.permission = permission
        self.hooks = hooks
        self._tool_by_name = {t.name: t for t in self.tools}
        self._langchain_tool_by_name = {t.name: t for t in adapt_tools(self.tools)}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.plan_mode = False  # toggled by /plan; while on, mutating tools are refused

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)
        self._pending_tool_calls = []
        self._session_cwd = None
        self.graph = self._build_graph()

    def _full_messages(self) -> list[dict]:
        system = self._system
        # re-injected every round, like the task list below, so a toggle made
        # between turns takes effect on the very next request
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        # the task list is re-injected every round, so the model always sees the
        # current state rather than a stale copy buried in old tool results
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        return [{"role": "system", "content": system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [convert_to_openai_tool(t) for t in self._langchain_tool_by_name.values()]

    def chat(self, user_input: str, on_token=None, on_tool=None, on_reasoning=None) -> str:
        """Run a LangGraph invocation, retaining the conversation between turns."""
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        self.messages.append({"role": "user", "content": user_input})
        self._pending_tool_calls = []
        self._session_cwd = get_tracked_cwd()
        try:
            result = self.graph.invoke(
                {"messages": self.messages, "rounds": 0, "cwd": self._session_cwd},
                config={
                    # max_rounds counts model calls, not graph transitions.
                    "recursion_limit": max(16, self.max_rounds * 4 + 8),
                    "configurable": {
                        "on_token": on_token,
                        "on_tool": on_tool,
                        "on_reasoning": on_reasoning,
                    },
                },
            )
            self.messages = result["messages"]
            return result["output"]
        except KeyboardInterrupt:
            self._answer_pending_tool_calls(self._pending_tool_calls)
            raise
        finally:
            # Graph nodes can run on worker threads; cwd must cross that edge.
            set_tracked_cwd(self._session_cwd)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("model", self._model_node)
        builder.add_node("authorize", self._authorize_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("compact", self._compact_node)
        builder.add_node("limit", self._limit_node)
        builder.add_edge(START, "prepare")
        builder.add_conditional_edges("prepare", self._next_round, {"model": "model", "limit": "limit"})
        builder.add_conditional_edges("model", self._after_model, {"tools": "authorize", "done": END})
        builder.add_edge("authorize", "tools")
        builder.add_edge("tools", "compact")
        builder.add_conditional_edges("compact", self._next_round, {"model": "model", "limit": "limit"})
        builder.add_edge("limit", END)
        return builder.compile()

    def _prepare_node(self, state: AgentState) -> dict:
        self.messages = state["messages"]
        self.context.maybe_compress(self.messages, self.llm)
        return {"messages": self.messages}

    def _model_node(self, state: AgentState, config: RunnableConfig) -> dict:
        self.messages = state["messages"]
        callbacks = config.get("configurable", {})
        response = self.llm.chat(
            messages=self._full_messages(),
            tools=self._tool_schemas(),
            on_token=callbacks.get("on_token"),
            on_reasoning=callbacks.get("on_reasoning"),
        )
        self.messages.append(response.message)
        self._pending_tool_calls = response.tool_calls
        return {
            "messages": self.messages,
            "rounds": state["rounds"] + 1,
            "response": response,
            "output": response.content,
        }

    def _authorize_node(self, state: AgentState, config: RunnableConfig) -> dict:
        calls = state["response"].tool_calls
        on_tool = config.get("configurable", {}).get("on_tool")
        for call in calls:
            if on_tool:
                on_tool(call.name, call.arguments)
        # All consent is settled before starting parallel tool workers.
        return {"permissions": [self._pre_hooks(tc) or self._permit(tc) for tc in calls]}

    def _tools_node(self, state: AgentState) -> dict:
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        self.messages = state["messages"]
        calls = state["response"].tool_calls
        set_tracked_cwd(state.get("cwd"))
        try:
            if len(calls) == 1:
                tc = calls[0]
                result = state["permissions"][0]
                if result is None:
                    result = self._exec_tool(tc)
                    self._post_hooks(tc, result)
                results = [result]
            else:
                results = self._exec_tools_parallel(calls, prechecked=state["permissions"])
            for tc, result in zip(calls, results):
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            self._pending_tool_calls = []
            return {"messages": self.messages, "cwd": get_tracked_cwd()}
        finally:
            self._session_cwd = get_tracked_cwd()

    def _compact_node(self, state: AgentState) -> dict:
        self.messages = state["messages"]
        self.context.maybe_compress(self.messages, self.llm)
        return {"messages": self.messages}

    def _next_round(self, state: AgentState) -> str:
        return "limit" if state["rounds"] >= self.max_rounds else "model"

    @staticmethod
    def _after_model(state: AgentState) -> str:
        return "tools" if state["response"].tool_calls else "done"

    @staticmethod
    def _limit_node(state: AgentState) -> dict:
        return {"output": "(reached maximum tool-call rounds)"}

    def _pre_hooks(self, tc) -> str | None:
        """PreToolUse hooks, fired before consent. A string return blocks the
        call and becomes the tool result the model sees; None lets it through."""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)

    def _post_hooks(self, tc, result: str):
        """PostToolUse hooks observe a finished call; they can never block."""
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)

    def _permit(self, tc) -> str | None:
        """Consent check for one call. None means go ahead; a string is the
        refusal, returned as the tool result instead of executing."""
        # plan mode outranks consent, even --yes: while it's on nothing mutates
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return (
                "Plan mode is on, so this call was refused: plan mode is "
                "read-only. Do not retry it. Keep investigating with the "
                "read-only tools, then present the plan and stop. The user can "
                'approve it by typing "approve", or exit plan mode with /plan.'
            )
        if self.permission is None:
            return None
        return self.permission.check(tc.name, tc.arguments)

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        # a tool that blows up gets reported back as text, never kills the loop
        try:
            return self._langchain_tool_by_name[tc.name].invoke(tc.arguments)
        except Exception as e:  # noqa: BLE001
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None, prechecked=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        if prechecked is None:
            for tc in tool_calls:
                if on_tool:
                    on_tool(tc.name, tc.arguments)
            results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
        else:
            results = list(prechecked)
        # the tracked cwd is thread-local, so pool workers would otherwise
        # start from the launch directory: hand the session cwd in, and merge
        # any cd back in call order so the batch behaves like a sequence
        session_cwd = get_tracked_cwd()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            def run(i, tc, base_cwd):
                is_bash = tc.name == "bash"
                # AgentTool can launch a child graph from this worker too.
                # Every worker inherits the session cwd; only Bash reports cd.
                set_tracked_cwd(base_cwd)
                out = self._exec_tool(tc)
                if not is_bash:
                    return i, out, None
                cwd_after = get_tracked_cwd()
                # only a worker that actually moved the dir reports back; a
                # sibling that merely inherited the base must not revert a cd
                return i, out, cwd_after if cwd_after != base_cwd else None

            futures = {
                i: pool.submit(run, i, tc, session_cwd)
                for i, tc in enumerate(tool_calls)
                if results[i] is None
            }
            for i, future in futures.items():
                i, results[i], worker_cwd = future.result()
                if worker_cwd is not None:
                    session_cwd = worker_cwd
                    set_tracked_cwd(worker_cwd)
        for i in futures:
            self._post_hooks(tool_calls[i], results[i])
        return results

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        # a user-visible reset must not leak the dead conversation's checklist
        # into the next system prompt
        todo = self._tool_by_name.get("todo_write")
        if isinstance(todo, TodoWriteTool):
            todo._tasks = []
