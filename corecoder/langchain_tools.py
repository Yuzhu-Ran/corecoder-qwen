"""Adapt CoreCoder tools to LangChain without changing their execution policy.

The JSON Schema is retained as a dictionary rather than inferred from the
wrapper's ``**arguments`` signature. This also preserves dynamic MCP schemas.
Python argument binding remains strict, while semantic/type validation remains
the responsibility of the original tool or remote MCP server, as before.

Permission checks and hooks belong to Agent's dispatcher and must run before
invoking an adapted tool. A StructuredTool is an execution adapter, not an
authorization boundary.
"""

import inspect
from collections.abc import Iterable
from copy import deepcopy

from langchain_core.tools import StructuredTool

from .tools.base import Tool


def adapt_tool(tool: Tool) -> StructuredTool:
    """Wrap a tool while preserving its schema, defaults, state and errors.

    Using a dict ``args_schema`` deliberately avoids coercing JSON arguments
    into Python types or dropping fields from MCP's dynamic input objects.
    ``signature.bind`` rejects missing/extra Python parameters; tool-specific
    errors are left untouched for the Agent to report back to the model.
    """

    def execute(**arguments):
        # Resolve the method at call time: tools are stateful and callers may
        # replace their implementation (for example during an offline test).
        method = tool.execute
        inspect.signature(method).bind(**arguments)
        return method(**arguments)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=deepcopy(tool.parameters),
        func=execute,
    )


def adapt_tools(tools: Iterable[Tool]) -> list[StructuredTool]:
    """Adapt a registry while keeping its original order and tool instances."""
    return [adapt_tool(tool) for tool in tools]
