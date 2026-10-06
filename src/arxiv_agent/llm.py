"""Groq LLM client wrapper.

Every LLM call goes through call_llm(). This module is deliberately the only
place that calls the Groq SDK. It owns:
  - the client (lazily instantiated so unit tests can patch env/client first),
  - the uniform response shape (LLMResponse) the loop reads,
  - detection of malformed/truncated tool calls, raised as typed exceptions
    the loop catches and dispatches to failure handlers in failures.py.

What it deliberately does NOT own:
  - retries / backoff — those live in failures.py, which is the ONE place
    that decides when to retry. The Groq SDK's own retries are disabled
    (max_retries=0) so handler logic reflects reality, not reality-plus-two-
    hidden-retries, and so latency_ms is honest.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from groq import Groq
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool invocation the LLM asked for.

    Groq (OpenAI-compatible) returns tool calls as a list on the assistant
    message. Each has an id we must echo back in the tool result message, or
    the next turn confuses the model.
    """

    id: str
    name: str
    args: dict[str, Any]


class LLMResponse(BaseModel):
    """Uniform shape returned by call_llm(). The loop reads content vs
    tool_calls to decide the next state transition.
    """

    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""                 # "stop", "tool_calls", "length", ...
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw_response: Optional[dict[str, Any]] = None   # only populated if include_raw=True


class MalformedToolArgsError(Exception):
    """LLM emitted a tool_call whose arguments aren't a parseable JSON object.

    The loop catches this and routes to the repair handler. tool_call_id is
    included so the repair prompt can point the model at the specific call.
    """

    def __init__(self, tool_name: str, tool_call_id: str, raw_args: str, parse_error: str):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.raw_args = raw_args
        self.parse_error = parse_error
        super().__init__(f"malformed args for {tool_name} (id={tool_call_id}): {parse_error}")


class TruncatedResponseError(Exception):
    """finish_reason=='length' — model hit max_tokens mid-response.

    Distinct from malformed JSON because the fix is different: a repair
    prompt won't help; need to raise max_tokens or shorten context. The
    loop's handler bumps max_tokens once, then gives up if it happens again.
    """

    def __init__(self, prompt_tokens: int, max_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        super().__init__(f"response truncated at max_tokens={max_tokens}")


_client: Optional[Groq] = None
_MODEL: Optional[str] = None


def _get_client() -> Groq:
    """Lazy client instantiation.

    Env lookup and client construction happen on first use, not import.
    Tests can set env vars or monkeypatch module-level _client before any
    call_llm() is made.

    max_retries=0 disables the SDK's hidden retry layer. All retry logic
    lives in failures.py so it's testable in one place and latency_ms is
    accurate.

    timeout=30s is explicit. The SDK default of 60s is too generous given
    an 8K TPM budget — a single stuck call wastes nearly a quarter of a
    minute's budget before failing.
    """
    global _client
    if _client is None:
        _client = Groq(max_retries=0, timeout=30.0)
    return _client


def _get_model() -> str:
    global _MODEL
    if _MODEL is None:
        _MODEL = os.environ["GROQ_MODEL"]
    return _MODEL


def call_llm(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    include_raw: bool = False,
    allow_truncation: bool = False,
) -> LLMResponse:
    """Call the LLM. See module docstring for scope.

    Raises:
      TruncatedResponseError: finish_reason=='length' before parsing tool_calls.
      MalformedToolArgsError: a tool_call's arguments aren't a JSON object.
      groq.* / httpx.*: everything else propagates for failures.py to handle.

    Temperature defaults to 0.0. Agent loops are hard enough to debug without
    stochastic tool calls; raise it deliberately for creative sub-tasks.

    Known limitation: when the model emits multiple tool_calls and one is
    malformed, this raises on the first bad call and discards the rest. The
    Weekend 1 loop uses sequential tool calls, so this is tolerable. Weekend 2
    revisits when parallel tool-call batching is added.
    """
    kwargs: dict[str, Any] = {
        "model": _get_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools

    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    choice = resp.choices[0]
    msg = choice.message
    finish_reason = choice.finish_reason or ""

    # Truncation check must come BEFORE parsing tool_calls — truncated JSON
    # would otherwise masquerade as a malformed-args error, sending the
    # repair handler into a loop it can't win against the same max_tokens.
    if finish_reason == "length" and not allow_truncation:
        raise TruncatedResponseError(
            prompt_tokens=(resp.usage.prompt_tokens if resp.usage else 0),
            max_tokens=max_tokens,
        )

    tool_calls: list[ToolCall] = []
    for tc in (msg.tool_calls or []):
        raw_args = tc.function.arguments or "{}"
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as e:
            raise MalformedToolArgsError(
                tool_name=tc.function.name,
                tool_call_id=tc.id,
                raw_args=raw_args,
                parse_error=str(e),
            ) from e

        # json.loads accepts strings, numbers, lists, nulls — none of which
        # are valid tool arguments. Open-weights models occasionally emit
        # these. Treat as malformed so the repair handler sees it.
        if not isinstance(parsed, dict):
            raise MalformedToolArgsError(
                tool_name=tc.function.name,
                tool_call_id=tc.id,
                raw_args=raw_args,
                parse_error=f"expected JSON object, got {type(parsed).__name__}",
            )

        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=parsed))

    usage = resp.usage
    return LLMResponse(
        content=msg.content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=latency_ms,
        raw_response=(resp.model_dump() if include_raw else None),
    )