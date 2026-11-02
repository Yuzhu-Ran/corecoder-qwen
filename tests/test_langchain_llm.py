"""LangChain streaming integration tests without credentials or HTTP calls."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessageChunk
from openai import APIConnectionError, BadRequestError

from corecoder import langchain_llm as module
from corecoder.langchain_llm import LangChainLLM


def fake_model(monkeypatch, responses):
    """Return instances created by the adapter and scripts bound to them."""
    instances = []
    scripts = iter(responses)

    class Model:
        def __init__(self, **options):
            self.options = options
            self.tools = None
            self.messages = []
            instances.append(self)

        def bind_tools(self, tools):
            self.tools = tools
            return self

        def stream(self, messages):
            self.messages.append(messages)
            yield from next(scripts)

    monkeypatch.setattr(module, "ChatOpenAI", Model)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return instances


def test_stream_binds_tools_reassembles_arguments_and_counts_usage(monkeypatch):
    chunks = [
        AIMessageChunk(content="Checking ", tool_call_chunks=[{"name": "read_file", "args": '{"file_', "id": "c1", "index": 0}]),
        AIMessageChunk(content="now.", tool_call_chunks=[{"name": None, "args": 'path":"example.py"}', "id": None, "index": 0}]),
        AIMessageChunk(content="", usage_metadata={"input_tokens": 31, "output_tokens": 12, "total_tokens": 43}),
    ]
    models = fake_model(monkeypatch, [chunks])
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
    model = LangChainLLM("qwen-flash", "test-key", "https://example.invalid/v1", max_tokens=128, temperature=0)
    seen = []
    response = model.chat([{"role": "user", "content": "Check example.py"}], tools, on_token=seen.append)
    assert response.content == "Checking now."
    assert seen == ["Checking ", "now."]
    assert response.tool_calls[0].arguments == {"file_path": "example.py"}
    assert response.tool_calls[0].id == "c1"
    assert models[0].tools == tools
    assert models[0].options["use_responses_api"] is False
    assert models[0].options["max_retries"] == 0
    assert model.total_prompt_tokens == 31
    assert model.total_completion_tokens == 12


def test_switching_model_rebuilds_client_and_keeps_totals(monkeypatch):
    models = fake_model(monkeypatch, [[AIMessageChunk(content="one")], [AIMessageChunk(content="two")]])
    llm = LangChainLLM("first", "test-key")
    assert llm.chat([{"role": "user", "content": "hello"}]).content == "one"
    llm.model = "second"
    assert llm.chat([{"role": "user", "content": "hello"}]).content == "two"
    assert [m.options["model"] for m in models] == ["first", "second"]


def test_reasoning_is_not_put_in_answer_or_history(monkeypatch):
    chunks = [
        AIMessageChunk(content=[{"type": "reasoning", "reasoning": "private thought"}]),
        AIMessageChunk(content=[{"type": "text", "text": "answer"}]),
    ]
    fake_model(monkeypatch, [chunks])
    shown = []
    response = LangChainLLM("qwen-flash", "test-key").chat([], on_reasoning=shown.append)
    assert shown == ["private thought"]
    assert response.content == "answer"
    assert "private thought" not in str(response.message)


def test_empty_stream_and_missing_usage(monkeypatch):
    fake_model(monkeypatch, [[], [AIMessageChunk(content="OK")]])
    llm = LangChainLLM("qwen-flash", "test-key")
    assert llm.chat([]).content == ""
    assert llm.chat([]).content == "OK"
    assert llm.total_prompt_tokens == 0


def test_invalid_tool_arguments_become_empty_object(monkeypatch):
    fake_model(
        monkeypatch,
        [
            [
                AIMessageChunk(
                    content="",
                    invalid_tool_calls=[
                        {"name": "read_file", "args": "broken", "id": "c1", "error": "bad JSON", "type": "invalid_tool_call"}
                    ],
                )
            ]
        ],
    )
    result = LangChainLLM("qwen-flash", "test-key").chat([])
    assert result.tool_calls[0].arguments == {}
    assert result.tool_calls[0].id == "c1"


def test_null_usage_values_are_safe(monkeypatch):
    chunk = SimpleNamespace(
        content="OK", tool_calls=[], invalid_tool_calls=[], usage_metadata={"input_tokens": None, "output_tokens": None}
    )
    fake_model(monkeypatch, [[chunk]])
    response = LangChainLLM("qwen-flash", "test-key").chat([])
    assert response.prompt_tokens == response.completion_tokens == 0


def test_midstream_connection_failure_retries_without_double_counting(monkeypatch):
    def broken():
        yield AIMessageChunk(content="partial")
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))

    fake_model(
        monkeypatch,
        [
            broken(),
            [AIMessageChunk(content="complete", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})],
        ],
    )
    llm = LangChainLLM("qwen-flash", "test-key")
    response = llm.chat([])
    assert response.content == "complete"
    assert llm.total_prompt_tokens == 10
    assert llm.total_completion_tokens == 2


def test_transient_retries_are_bounded(monkeypatch):
    def broken():
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        yield

    models = fake_model(monkeypatch, [broken(), broken(), broken()])
    with pytest.raises(APIConnectionError):
        LangChainLLM("qwen-flash", "test-key").chat([])
    assert len(models[0].messages) == 3


def test_rejected_stream_options_are_disabled_and_cached(monkeypatch):
    def rejected():
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
        raise BadRequestError("unsupported stream_options", response=response, body=None)
        yield

    models = fake_model(monkeypatch, [rejected(), [AIMessageChunk(content="OK")], [AIMessageChunk(content="again")]])
    llm = LangChainLLM("qwen-flash", "test-key")
    assert llm.chat([]).content == "OK"
    assert llm.chat([]).content == "again"
    assert [m.options["stream_usage"] for m in models] == [True, False]


def test_other_client_errors_are_not_retried_indefinitely(monkeypatch):
    def rejected():
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
        raise BadRequestError("invalid request", response=response, body=None)
        yield

    models = fake_model(monkeypatch, [rejected(), rejected()])
    with pytest.raises(BadRequestError):
        LangChainLLM("qwen-flash", "test-key").chat([])
    # One compatibility fallback is allowed; an unrecognized 400 then raises.
    assert len(models) == 2


def event_stream(chunks):
    events = []
    for chunk in chunks:
        events.append("data: " + json.dumps(chunk) + "\n\n")
    events.append("data: [DONE]\n\n")
    return "".join(events).encode()


@pytest.mark.parametrize(
    ("usage", "expected_tokens"),
    [
        ({"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}, (11, 3)),
        ({"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}, (0, 0)),
    ],
)
def test_real_langchain_client_serializes_tools_and_streams_results(usage, expected_tokens):
    """Exercise ChatOpenAI and its SDK using an in-process HTTP transport."""
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "qwen-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"file_path":"example.py"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "qwen-flash",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "qwen-flash",
                "choices": [],
                "usage": usage,
            },
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=event_stream(chunks))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file.",
                "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            },
        }
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = LangChainLLM("qwen-flash", "test-key", "https://example.invalid/compatible-mode/v1", http_client=client)
        result = llm.chat([{"role": "user", "content": "Read example.py"}], tools=tools)
    url, body = requests[0]
    assert url.endswith("/compatible-mode/v1/chat/completions")
    assert body["model"] == "qwen-flash"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert result.tool_calls[0].arguments == {"file_path": "example.py"}
    assert (result.prompt_tokens, result.completion_tokens) == expected_tokens


def test_real_client_falls_back_to_legacy_token_limit(monkeypatch):
    requests = []
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "max_completion_tokens" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: 'max_completion_tokens'", "type": "invalid_request_error"}},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=event_stream(
                [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": "qwen-flash",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    },
                ]
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = LangChainLLM("qwen-flash", "test-key", "https://example.invalid/v1", max_tokens=128, http_client=client)
        assert llm.chat([{"role": "user", "content": "Reply OK"}]).content == "OK"
    assert len(requests) == 2
    assert requests[1]["max_tokens"] == 128
    assert "max_completion_tokens" not in requests[1]


def test_shared_model_supports_parallel_subagent_calls(monkeypatch):
    workers = 8
    rendezvous = threading.Barrier(workers, timeout=5)
    instances = []

    class Model:
        def __init__(self, **options):
            # Allow other callers to reach the cache while creation is in flight.
            threading.Event().wait(0.01)
            instances.append(self)

        def stream(self, messages):
            # If the adapter serialized the network stream, this would time out.
            rendezvous.wait()
            yield AIMessageChunk(content="OK", usage_metadata={"input_tokens": 11, "output_tokens": 2, "total_tokens": 13})

    monkeypatch.setattr(module, "ChatOpenAI", Model)
    llm = LangChainLLM("qwen-flash", "test-key")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(llm.chat, [{"role": "user", "content": "Reply OK"}]) for _ in range(workers)]
        assert [future.result(timeout=10).content for future in futures] == ["OK"] * workers
    assert len(instances) == 1
    assert llm.total_prompt_tokens == workers * 11
    assert llm.total_completion_tokens == workers * 2


def test_old_inflight_call_does_not_replace_new_model_configuration(monkeypatch):
    old_started = threading.Event()
    old_release = threading.Event()
    invoked = []

    class Model:
        def __init__(self, **options):
            self.options = options

        def stream(self, messages):
            model = self.options["model"]
            invoked.append(model)
            if model == "first" and self.options["stream_usage"]:
                response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
                raise BadRequestError("unsupported stream_options", response=response, body=None)
            if model == "first":
                old_started.set()
                assert old_release.wait(timeout=5)
            yield AIMessageChunk(content=model)

    monkeypatch.setattr(module, "ChatOpenAI", Model)
    llm = LangChainLLM("first", "test-key")
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(llm.chat, [])
        try:
            assert old_started.wait(timeout=5)
            llm.model = "second"
            assert llm.chat([]).content == "second"
        finally:
            old_release.set()
        assert pending.result(timeout=5).content == "first"
    assert llm.chat([]).content == "second"
    assert invoked == ["first", "first", "second", "second"]


@pytest.mark.parametrize("final_success", [True, False])
def test_parameter_adaptations_and_transient_retries_have_six_request_bound(monkeypatch, final_success):
    def reject(message):
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid"))
        raise BadRequestError(message, response=response, body=None)
        yield

    def unavailable():
        raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        yield

    scripts = [
        reject("unsupported temperature"),
        unavailable(),
        reject("unsupported max_completion_tokens"),
        unavailable(),
        reject("unsupported stream_options"),
        [AIMessageChunk(content="OK")] if final_success else unavailable(),
    ]
    instances = fake_model(monkeypatch, scripts)
    llm = LangChainLLM("qwen-flash", "test-key", temperature=0, max_tokens=128)
    if final_success:
        assert llm.chat([]).content == "OK"
    else:
        with pytest.raises(APIConnectionError):
            llm.chat([])
    assert sum(len(instance.messages) for instance in instances) == 6
