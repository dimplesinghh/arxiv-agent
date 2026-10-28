"""citation_check tool — verify a claim against provided source chunks.

Returns per-source supported/not plus a one-line rationale, and an overall
verdict:
  - supported:            at least one source clearly supports the claim
  - partially_supported:  sources mention the topic but don't fully support
                          the claim as stated (common: numbers don't match)
  - unsupported:          no source addresses the specific claim

The agent calls this before finalizing an answer. If the verdict is
'unsupported', ANSWER state should refuse or re-retrieve. This is the
honesty gate — without it, the agent will happily cite sources that
don't actually support the claim.

Design: structured JSON output, not freeform text. Freeform verdicts
drift between runs; JSON forces the same shape every time.
"""

from __future__ import annotations

import json
import time
from typing import Any

from arxiv_agent.failures import call_llm_with_retry
from arxiv_agent.llm import MalformedToolArgsError
from arxiv_agent.schema import ToolError, ToolErrorKind, ToolResult


# Caps protect the per-minute token budget. 10 × 500 ≈ 5000 input tokens
# plus prompt + claim ≈ 5500. Fits 8K TPM with headroom.
_MAX_SOURCES = 10
_MAX_SOURCE_CHARS_EACH = 2000
_MAX_OUTPUT_TOKENS = 1200

_VALID_VERDICTS = {"supported", "partially_supported", "unsupported"}


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "citation_check",
        "description": (
            "Verify whether a specific claim is supported by a provided set of "
            "source chunks (e.g. from rag_retrieve). Returns a per-source "
            "verdict and an overall verdict (supported / partially_supported / "
            "unsupported). Call this before answering — if the overall verdict "
            "is 'unsupported' you must not make the claim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "A single, specific factual claim to verify. Not a question.",
                },
                "sources": {
                    "type": "array",
                    "description": "List of source chunks to check against. Max 10.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            # null allowed: fetch_paper sections have no chunk_idx,
                            # and Groq rejects the whole call on a type mismatch.
                            "arxiv_id": {"type": ["string", "null"]},
                            "chunk_idx": {"type": ["integer", "null"]},
                        },
                        "required": ["text"],
                    },
                    "maxItems": 10,
                },
            },
            "required": ["claim", "sources"],
        },
    },
}


_SYSTEM_PROMPT = """You are a strict citation verifier. Given a claim and numbered source chunks, decide for each source whether it supports the claim, and give an overall verdict.

Rules:
- A source "supports" a claim only if the claim's specific assertion is explicitly stated or directly implied by the source text. Topic overlap is not support.
- Numbers must match: if the claim says "50% improvement" and the source says "30% improvement", that source does NOT support the claim.
- Be strict. When in doubt, mark as not supported.

Return a single JSON object with this exact shape:
{
  "per_source": [
    {"index": 0, "supported": true, "rationale": "one short sentence"},
    ...
  ],
  "verdict": "supported" | "partially_supported" | "unsupported",
  "verdict_rationale": "one short sentence explaining the overall verdict"
}

Verdict rules:
- "supported":            at least one source clearly supports the claim
- "partially_supported":  sources discuss the topic but don't fully support the claim as stated (e.g. different numbers, missing qualifier)
- "unsupported":          no source addresses the specific claim

Return ONLY the JSON object. No preamble, no explanation outside the JSON."""


def citation_check(claim: str, sources: list[dict[str, Any]]) -> ToolResult:
    """Pure function. Returns ToolResult. Never raises into the loop."""
    t0 = time.perf_counter()

    if not claim or not claim.strip():
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message="claim is required and must be non-empty",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    if not isinstance(sources, list) or not sources:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message="sources must be a non-empty list",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    if len(sources) > _MAX_SOURCES:
        sources = sources[:_MAX_SOURCES]

    # Validate + truncate each source. Per-chunk silent truncation is fine
    # (unlike summarize) because verification is per-chunk — a truncated chunk
    # simply has less evidence to prove support with, which the model handles.
    normalized_sources: list[dict[str, Any]] = []
    for i, s in enumerate(sources):
        if not isinstance(s, dict) or "text" not in s:
            return ToolResult(
                ok=False,
                error=ToolError(
                    kind=ToolErrorKind.TOOL_ERROR_FATAL,
                    message=f"source {i} missing required 'text' field",
                    retryable=False,
                ),
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        text = str(s["text"])[:_MAX_SOURCE_CHARS_EACH]
        normalized_sources.append({
            "index": i,
            "text": text,
            "arxiv_id": s.get("arxiv_id"),
            "chunk_idx": s.get("chunk_idx"),
        })

    sources_block = "\n\n".join(
        f"[Source {s['index']}]\n{s['text']}"
        for s in normalized_sources
    )
    user_content = f"Claim: {claim}\n\n{sources_block}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        resp = call_llm_with_retry(
            messages=messages,
            temperature=0.0,
            max_tokens=_MAX_OUTPUT_TOKENS,
            allow_truncation=False,   # malformed JSON on truncation; fail cleanly
        )
    except MalformedToolArgsError:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message="unexpected tool call from citation_check prompt",
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

    latency_ms = int((time.perf_counter() - t0) * 1000)
    raw = (resp.content or "").strip()

    # Models often ignore "JSON only" and wrap in ```json ... ``` fences.
    # Strip defensively.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"citation_check returned non-JSON: {e}; got: {raw[:200]!r}",
                retryable=True,
            ),
            latency_ms=latency_ms,
        )

    if not isinstance(parsed, dict):
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"expected JSON object, got {type(parsed).__name__}",
                retryable=True,
            ),
            latency_ms=latency_ms,
        )

    verdict = parsed.get("verdict")
    if verdict not in _VALID_VERDICTS:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"invalid verdict: {verdict!r}; expected one of {_VALID_VERDICTS}",
                retryable=True,
            ),
            latency_ms=latency_ms,
        )

    per_source = parsed.get("per_source")
    if not isinstance(per_source, list):
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message="per_source missing or not a list",
                retryable=True,
            ),
            latency_ms=latency_ms,
        )

    # Attach arxiv_id/chunk_idx back to each per_source entry — the model
    # doesn't see those (to avoid name bias), so merge metadata in from the
    # original sources by index.
    index_to_source = {s["index"]: s for s in normalized_sources}
    enriched = []
    for entry in per_source:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        src = index_to_source.get(idx, {})
        enriched.append({
            "index": idx,
            "arxiv_id": src.get("arxiv_id"),
            "chunk_idx": src.get("chunk_idx"),
            "supported": bool(entry.get("supported", False)),
            "rationale": str(entry.get("rationale", "")),
        })

    supporting_sources = [e for e in enriched if e["supported"]]

    return ToolResult(
        ok=True,
        data={
            "verdict": verdict,
            "verdict_rationale": str(parsed.get("verdict_rationale", "")),
            "per_source": enriched,
            "num_sources_checked": len(normalized_sources),
            "num_supporting": len(supporting_sources),
            "supporting_arxiv_ids": [
                e["arxiv_id"] for e in supporting_sources if e["arxiv_id"]
            ],
            "claim": claim,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
        },
        latency_ms=latency_ms,
    )