"""summarize tool — focused summarization of a text block.

Wraps call_llm. Enforces an input size cap so a single summarize call can't
blow the 8K TPM budget on its own. Truncates oversized input and reports it
(not silent truncation — the agent needs to know the summary didn't see
everything).

The `focus` parameter is required. Generic summaries ("summarize this
paper") produce diluted output; focused summaries ("summarize this paper's
evaluation methodology") are useful. Making focus required pushes the agent
to be specific.
"""

from __future__ import annotations

import time
from typing import Any

from arxiv_agent.llm import (
    MalformedToolArgsError,
    TruncatedResponseError,
    call_llm,
)
from arxiv_agent.schema import (
    ToolError,
    ToolErrorKind,
    ToolResult,
    approx_token_count,
)


# Input budget. Chosen so one summarize call ~= one minute of TPM budget at
# most: 6000 input tokens + ~500 output ≈ 6500 tokens. Leaves headroom in
# the per-minute window for the planner/reflect calls around it.
_MAX_INPUT_TOKENS = 6000
_MAX_INPUT_CHARS = _MAX_INPUT_TOKENS * 4

# Output cap. Summaries longer than this stop being summaries.
_MAX_OUTPUT_TOKENS = 800

# Minimum useful input. Below this, there's nothing to summarize — the agent
# should just read the text itself. Fail fast rather than wasting an LLM call.
_MIN_INPUT_CHARS = 200


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "summarize",
        "description": (
            "Summarize a block of text with a specific focus. "
            "Use when you have more raw text than you can reason over directly "
            "(e.g. the full sections from fetch_paper, or 10 retrieved chunks). "
            "The 'focus' parameter must be specific — 'summarize this paper's "
            "evaluation methodology', not 'summarize this paper'. "
            "Input is truncated at ~6000 tokens; truncation is reported in the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize.",
                },
                "focus": {
                    "type": "string",
                    "description": (
                        "What to summarize for. Required and must be specific. "
                        "Good: 'the ablation study results'. Bad: 'everything'."
                    ),
                },
            },
            "required": ["text", "focus"],
        },
    },
}


_SYSTEM_PROMPT = (
    "You are a precise technical summarizer. Produce a focused summary of the "
    "provided text, targeting the user's stated focus. Rules:\n"
    "- Stay strictly within the provided text. Do not invent facts, numbers, "
    "or claims not present in the source.\n"
    "- If the text does not address the focus at all, say so explicitly rather "
    "than padding with tangential content.\n"
    "- Prefer concrete numbers, names, and specifics over general descriptions.\n"
    "- No preamble ('Here is a summary...'). Start with the content.\n"
    "- Target length: 100-300 words unless the content genuinely demands more."
)


def summarize(text: str, focus: str) -> ToolResult:
    """Pure function. Returns ToolResult. Never raises into the loop."""
    t0 = time.perf_counter()

    if not focus or not focus.strip():
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message="focus is required and must be non-empty",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    if len(text) < _MIN_INPUT_CHARS:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=f"text too short to summarize ({len(text)} chars); read it directly",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    truncated_input = False
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]
        truncated_input = True

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Focus: {focus}\n\nText:\n{text}"},
    ]

    try:
        resp = call_llm(
            messages=messages,
            temperature=0.0,
            max_tokens=_MAX_OUTPUT_TOKENS,
            allow_truncation=True,   # we accept partial summaries and flag them
        )
    except MalformedToolArgsError:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message="unexpected tool call from summarize prompt",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"{type(e).__name__}: {e}",
                retryable=True,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    summary = (resp.content or "").strip()
    output_truncated = resp.finish_reason == "length"
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if not summary:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.EMPTY_RESULT,
                message="LLM returned empty summary",
                retryable=True,
            ),
            latency_ms=latency_ms,
        )

    return ToolResult(
        ok=True,
        data={
            "summary": summary,
            "focus": focus,
            "input_truncated": truncated_input,
            "output_truncated": output_truncated,   # NEW: agent-visible flag
            "input_chars": len(text),
            "approx_input_tokens": approx_token_count(text),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        },
        latency_ms=latency_ms,
    )