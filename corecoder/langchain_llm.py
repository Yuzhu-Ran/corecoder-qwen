"""LangChain chat-model adapter for CoreCoder's public model interface.

The graph and CLI keep the small ``LLMResponse`` contract so scripted models
and the optional legacy/LiteLLM providers remain usable. Real compatible API
requests here go through ChatOpenAI, its tool binding and message streams.
"""

import json
import threading
import time
from typing import Any
from uuid import uuid4

from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APITimeoutError, BadRequestError, RateLimitError

from .llm import LLM, LLMResponse, ToolCall


def _text_content(content: Any) -> str:
    """Keep answer text, excluding reasoning and other non-text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "text_delta", "output_text"}:
            value = block.get("text", "")
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _reasoning_content(chunk: Any) -> str:
    extra = getattr(chunk, "additional_kwargs", {}) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning
    parts = []
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"reasoning", "thinking"}:
                text = block.get("reasoning") or block.get("thinking") or block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


class LangChainLLM(LLM):
    """OpenAI-compatible model backed by LangChain's ChatOpenAI.

    ``model`` stays mutable for /model and saved-session restoration. Client
    instances are cached only while their effective configuration is unchanged.
    The SDK's inner retries are disabled. Each chat call allows at most two
    transient retries and three parameter adaptations (six requests total),
    including failures after a stream has already started.

    Provider-specific reasoning deltas are display-only when LangChain exposes
    them. ChatOpenAI may omit non-standard fields from a third-party provider;
    use the legacy provider if that provider's reasoning dialect is required.
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None, **kwargs):
        # Do not instantiate the legacy OpenAI client: this path uses LangChain.
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._requested_options: dict | None = None
        self._effective_options: dict | None = None
        self._cached_options: dict | None = None
        self._chat_model = None
        # Sub-agents reuse this instance and can run in parallel. Protect only
        # cache/counter bookkeeping, never the network stream itself.
        self._state_lock = threading.RLock()

    def _options(self) -> dict:
        options = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": 60,
            "stream_usage": True,
            **self.extra,
        }
        # Qwen and the other compatible providers use Chat Completions, not
        # OpenAI's Responses API. Bound retries at the outer streaming layer.
        options["use_responses_api"] = False
        options["max_retries"] = 0
        return options

    def _model_for(self, options: dict):
        with self._state_lock:
            if self._chat_model is None or options != self._cached_options:
                self._chat_model = ChatOpenAI(**options)
                self._cached_options = dict(options)
            return self._chat_model

    @staticmethod
    def _adapt_options(options: dict, error: BadRequestError) -> bool:
        """Apply one bounded, recognizable provider-dialect fallback."""
        message = str(error).lower()
        if "temperature" in message and "temperature" in options:
            options.pop("temperature")
            return True
        # ChatOpenAI normalizes max_tokens to max_completion_tokens. Older
        # compatible endpoints need the legacy field sent via extra_body.
        if "max_completion_tokens" in message and "max_tokens" in options:
            extra = dict(options.get("extra_body") or {})
            extra["max_tokens"] = options.pop("max_tokens")
            options["extra_body"] = extra
            return True
        if options.get("stream_usage"):
            options["stream_usage"] = False
            options.pop("stream_options", None)
            return True
        return False

    def chat(self, messages: list[dict], tools=None, on_token=None, on_reasoning=None) -> LLMResponse:
        with self._state_lock:
            requested = self._options()
            if requested != self._requested_options:
                self._requested_options = dict(requested)
                self._effective_options = dict(requested)
            options = dict(self._effective_options)
        transient_attempts = 0
        adaptations = 0

        while True:
            try:
                model = self._model_for(options)
                runnable = model.bind_tools(tools) if tools else model
                result = self._drain_langchain(runnable.stream(messages), on_token, on_reasoning)
                with self._state_lock:
                    # A slower in-flight call must not overwrite configuration
                    # established by /model or another caller in the meantime.
                    if requested == self._requested_options:
                        self._effective_options = dict(options)
                return result
            except BadRequestError as error:
                if adaptations >= 3 or not self._adapt_options(options, error):
                    raise
                adaptations += 1
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if transient_attempts >= 2:
                    raise
                time.sleep(2**transient_attempts)
                transient_attempts += 1
            except APIError as error:
                status = getattr(error, "status_code", None)
                if not status or status < 500 or transient_attempts >= 2:
                    raise
                time.sleep(2**transient_attempts)
                transient_attempts += 1

    def _drain_langchain(self, stream, on_token, on_reasoning) -> LLMResponse:
        full = None
        try:
            for chunk in stream:
                text = _text_content(getattr(chunk, "content", ""))
                if text and on_token:
                    on_token(text)
                reasoning = _reasoning_content(chunk)
                if reasoning and on_reasoning:
                    on_reasoning(reasoning)
                full = chunk if full is None else full + chunk
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

        if full is None:
            return LLMResponse()

        parsed = []
        calls = [*getattr(full, "tool_calls", []), *getattr(full, "invalid_tool_calls", [])]
        for call in calls:
            arguments = call.get("args")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            parsed.append(
                ToolCall(
                    id=call.get("id") or f"call_{uuid4().hex}",
                    name=call.get("name") or "",
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        usage = getattr(full, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        with self._state_lock:
            self.total_prompt_tokens += input_tokens
            self.total_completion_tokens += output_tokens
        return LLMResponse(
            content=_text_content(full.content),
            tool_calls=parsed,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
