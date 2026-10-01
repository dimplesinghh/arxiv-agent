"""rag_retrieve tool — semantic search over the arxiv_chunks Qdrant index.

Reuses Project 1's (arxiv-rag) index directly. Must use the SAME embedding model arxiv-rag
used to build the index, or query/chunk vectors live in different spaces and
retrieval returns garbage. The model name is hardcoded as a module constant,
not configurable — swapping it silently is the kind of bug that produces
plausible-looking but useless results.

Reranking (cross-encoder) is enabled by default. Project 1 measured +18%
Recall@1 and +16% MRR from reranking; for an agent whose answers are heavily
influenced by the top-1 result, that improvement compounds.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from arxiv_agent.schema import ToolError, ToolErrorKind, ToolResult


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Over-fetch when reranking: dense retrieval is approximate, so pull more
# candidates than the final top_k and let the cross-encoder re-score them.
# Project 1 used 20 candidates → top-10; same pattern here.
_RERANK_CANDIDATES = 20

_TIMEOUT_SECONDS = 15


SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "rag_retrieve",
        "description": (
            "Retrieve relevant chunks from the local arXiv paper index "
            "(cs.LG and cs.AI papers, ~300 papers, ~2500 chunks). "
            "Returns chunks with their source arxiv_id and the chunk text. "
            "Use for questions about the content of papers already in the index. "
            "Does NOT search the live arXiv API — use arxiv_search for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to find.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return after reranking. Default 5, max 10.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


# Loaded once per process. SentenceTransformer and CrossEncoder each take
# 1-2s to initialize (model load + warmup). Caching here means the agent
# loop pays the cost once at first call, not per step.
@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_qdrant() -> QdrantClient:
    return QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=int(os.environ["QDRANT_PORT"]),
        timeout=10,
    )


def _do_retrieve(query: str, top_k: int, use_rerank: bool) -> list[dict[str, Any]]:
    """Runs synchronously inside the thread-pool timeout wrapper."""
    collection = os.environ["QDRANT_COLLECTION"]
    embedder = _get_embedder()
    client = _get_qdrant()

    # Encode query. convert_to_numpy=True returns a plain array — Qdrant
    # client accepts list[float], so .tolist() is required.
    query_vec = embedder.encode(query, convert_to_numpy=True).tolist()

    fetch_k = _RERANK_CANDIDATES if use_rerank else top_k

    # query_points replaces the deprecated .search() in qdrant-client >= 1.11.
    # Response is a QueryResponse object with .points (list of ScoredPoint).
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=fetch_k,
        with_payload=True,
    )
    hits = response.points

    candidates = [
        {
            "arxiv_id": h.payload.get("paper_id"),
            "chunk_idx": h.payload.get("chunk_idx"),
            "text": h.payload.get("text", ""),
            "dense_score": float(h.score),
        }
        for h in hits
    ]

    if not use_rerank or not candidates:
        return candidates[:top_k]

    reranker = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def rag_retrieve(
    query: str,
    top_k: int = 5,
    use_rerank: bool = True,
) -> ToolResult:
    """Pure function. Returns ToolResult. Never raises into the loop."""
    t0 = time.perf_counter()

    top_k = max(1, min(10, top_k))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_retrieve, query, top_k, use_rerank)
            try:
                results = future.result(timeout=_TIMEOUT_SECONDS)
            except FuturesTimeout:
                return ToolResult(
                    ok=False,
                    error=ToolError(
                        kind=ToolErrorKind.TOOL_TIMEOUT,
                        message=f"rag_retrieve exceeded {_TIMEOUT_SECONDS}s",
                        retryable=True,
                    ),
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
    except KeyError as e:
        # Missing env var — not retryable; config problem.
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=f"missing env var: {e}",
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

    if not results:
        return ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.EMPTY_RESULT,
                message=f"no chunks matched query: {query!r}",
                retryable=False,
            ),
            latency_ms=latency_ms,
        )

    return ToolResult(
        ok=True,
        data={
            "results": results,
            "count": len(results),
            "reranked": use_rerank,
        },
        latency_ms=latency_ms,
    )