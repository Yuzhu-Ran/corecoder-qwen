"""Real StructuredTool calls around local and dynamic MCP implementations."""

from copy import deepcopy
from typing import ClassVar

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from corecoder.langchain_tools import adapt_tool, adapt_tools
from corecoder.mcp import MCPTool
from corecoder.tools.base import Tool


class EchoTool(Tool):
    name = "echo"
    description = "Return a value with an optional prefix."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "prefix": {"type": "string", "description": "Optional prefix"},
        },
        "required": ["value"],
    }

    def __init__(self):
        self.calls = []

    def execute(self, value: str, prefix: str = ">") -> str:
        self.calls.append((value, prefix))
        return prefix + value


def test_schema_and_defaults_survive_real_structured_tool_invoke():
    original = EchoTool()
    before = deepcopy(original.parameters)
    bridge = adapt_tool(original)

    assert isinstance(bridge, StructuredTool)
    assert bridge.name == original.name
    assert bridge.description == original.description
    assert bridge.args_schema == before
    assert bridge.invoke({"value": "hello"}) == ">hello"
    assert bridge.invoke({"value": "again", "prefix": "!"}) == "!again"
    assert original.calls == [("hello", ">"), ("again", "!")]
    assert original.parameters == before

    schema = convert_to_openai_tool(bridge)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == original.name
    assert schema["function"]["description"] == original.description
    assert schema["function"]["parameters"] == before


def test_adapted_schema_does_not_mutate_source_schema():
    original = EchoTool()
    bridge = adapt_tool(original)
    bridge.args_schema["properties"]["value"]["description"] = "Changed copy"
    assert "description" not in original.parameters["properties"]["value"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "missing a required argument"),
        ({"value": "hello", "unknown": 3}, "unexpected keyword argument"),
    ],
)
def test_python_signature_errors_are_not_silently_filtered(arguments, message):
    original = EchoTool()
    with pytest.raises(TypeError, match=message):
        adapt_tool(original).invoke(arguments)
    assert original.calls == []


def test_tool_implementation_type_error_is_not_relabelled():
    original = EchoTool()

    def broken(value, prefix=">"):
        raise TypeError("internal implementation failure")

    original.execute = broken
    with pytest.raises(TypeError, match="internal implementation failure"):
        adapt_tool(original).invoke({"value": "hello"})


def test_bridge_uses_current_tool_implementation():
    original = EchoTool()
    bridge = adapt_tool(original)
    original.execute = lambda value, prefix=">": "replacement:" + value
    assert bridge.invoke({"value": "hello"}) == "replacement:hello"


def test_dynamic_mcp_schema_and_all_arguments_reach_original_client():
    class FakeClient:
        name = "local_test"

        def __init__(self):
            self.calls = []

        def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return "remote tool result"

    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "filters": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["query"],
        "additionalProperties": True,
        "$defs": {"unused": {"type": "string"}},
    }
    client = FakeClient()
    original = MCPTool(client, {"name": "search", "description": "Search", "inputSchema": schema})
    bridge = adapt_tool(original)
    arguments = {"query": "hello", "filters": {"enabled": True, "tags": ["a", "b"]}, "limit": 3}

    assert bridge.args_schema == schema
    assert bridge.invoke(arguments) == "remote tool result"
    assert client.calls == [("search", arguments)]


def test_registry_order_and_independent_tool_state_are_preserved():
    first, second = EchoTool(), EchoTool()
    second.name = "second_echo"
    bridges = adapt_tools(iter([first, second]))

    assert [tool.name for tool in bridges] == ["echo", "second_echo"]
    bridges[1].invoke({"value": "second"})
    assert first.calls == []
    assert second.calls == [("second", ">")]
