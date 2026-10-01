"""arxiv_search tool — search the public arxiv API for papers matching a query.

Returns paper metadata (id, title, abstract, authors, date). Does NOT fetch
full paper text — that's fetch_paper's job. Separation of concerns: search
is cheap and query-driven; fetch is expensive and paper-id-driven.

The arxiv Python client enforces its own rate limit (~3s between calls) so
burst calls don't get IP-blocked. That delay is counted in our latency_ms —
the agent needs to know search is slow so it uses it deliberately.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import arxiv

from arxiv_agent.schema import ToolError, ToolErrorKind, ToolResult


# JSON schema the LLM sees. Matches OpenAI function-calling format.
# Kept narrow: three params, strong defaults. The more params, the more
# ways the LLM can emit malformed args.
SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "arxiv_search",
        "description": (
            "Search the public arXiv API for papers matching a query. "
            "Returns a list of papers (id, title, abstract, authors, published date). "
            "Use for discovering papers by topic or author. Does NOT fetch full text. "
            "Slow (~3s per call due to arXiv rate limits); use deliberately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Supports arXiv query syntax: 'cat:cs.LG AND ti:transformer', 'au:bengio', or free text.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max papers to return. Default 5, max 10.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "submitted_date"],
                    "description": "Sort order. 'relevance' is default; 'submitted_date' for 'recent' queries.",
                    "default": "relevance",
                },
            },
            "required": ["query"],
        },
    },
}


# Timeout for the whole call. arxiv client's ~3s internal delay means a
# normal call takes 3-8s; 20s absorbs transient slowness without hanging
# the agent indefinitely. TOOL_TIMEOUT fires beyond this.
_TIMEOUT_SECONDS = 20


def _do_search(query: str, max_results: int, sort_by: str) -> list[dict[str, Any]]:
    """Run the search synchronously. Called inside a ThreadPoolExecutor so we
    can enforce a timeout — arxiv client has no native timeout param.
    """
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submitted_date": arxiv.SortCriterion.SubmittedDate,
    }
    client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=0)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_map[sort_by],
    )
    results = []
    for r in client.results(search):
        # arxiv IDs come as "http://arxiv.org/abs/2401.12345v1" — strip to the bare ID.
        arxiv_id = r.entry_id.split("/")[-1]
        results.append({
            "arxiv_id": arxiv_id,
            "title": r.title.strip(),
            "abstract": r.summary.strip(),
            "authors": [a.name for a in r.authors],
            "published": r.published.isoformat() if r.published else None,
            "categories": r.categories,
        })
    return results


def arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> ToolResult:
    """Pure function. Returns ToolResult. Never raises into the loop."""
    t0 = time.perf_counter()

    # Clamp max_results defensively — the LLM sometimes ignores the schema's maximum.
    max_results = max(1, min(10, max_results))

    if sort_by not in ("relevance", "submitted_date"):
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=f"invalid sort_by: {sort_by!r}; expected 'relevance' or 'submitted_date'",
                retryable=False,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_search, query, max_results, sort_by)
            try:
                results = future.result(timeout=_TIMEOUT_SECONDS)
            except FuturesTimeout:
                return ToolResult(
                    ok=False,
                    error=ToolError(
                        kind=ToolErrorKind.TOOL_TIMEOUT,
                        message=f"arxiv_search exceeded {_TIMEOUT_SECONDS}s",
                        retryable=True,
                    ),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
    except arxiv.UnexpectedEmptyPageError as e:
        # arxiv's own error for malformed responses from upstream.
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_RETRYABLE,
                message=f"arxiv returned unexpected empty page: {e}",
                retryable=True,
            ),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as e:
        # Catch-all. Treat unknown errors as retryable once; failures.py
        # decides whether to actually retry. Fatal reclassification happens
        # if the retry also fails with the same error type.
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

    if not results:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.EMPTY_RESULT,
                message=f"no papers matched query: {query!r}",
                retryable=False,  # retrying the same query won't help; agent must reformulate
            ),
            latency_ms=latency_ms,
        )

    return ToolResult(
        ok=True,
        data={"results": results, "count": len(results)},
        latency_ms=latency_ms,
    )