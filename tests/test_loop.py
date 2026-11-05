"""Tests for loop._maybe_reformulate.

All offline — the LLM call inside _maybe_reformulate is monkeypatched,
and the retrieval tool it re-invokes is replaced in the registry.

Covers the reformulation contract:
  - fires on EMPTY_RESULT from rag_retrieve
  - fires on EMPTY_RESULT from arxiv_search
  - does NOT fire for other tools (e.g. fetch_paper)
  - does NOT fire for non-EMPTY_RESULT errors (e.g. TOOL_TIMEOUT)
  - fires at most once per run (counter guard)
  - a reformulated empty result is accepted; no second reformulation
  - new step is recorded with 'reformulated' note, step_n incremented,
    reformulations_attempted counter incremented
  - identical/empty reformulation output is rejected without consuming budget
"""

from __future__ import annotations

from typing import Any

import pytest

from arxiv_agent import loop
from arxiv_agent.llm import LLMResponse
from arxiv_agent.schema import (
    AgentState,
    ToolError,
    ToolErrorKind,
    ToolResult,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _empty_result(kind: ToolErrorKind = ToolErrorKind.EMPTY_RESULT) -> ToolResult:
    return ToolResult(
        ok=False,
        error=ToolError(kind=kind, message="no results", retryable=False),
        latency_ms=1,
    )


def _ok_result(data: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(ok=True, data=data or {"results": [{"arxiv_id": "x"}]}, latency_ms=1)


def _patch_llm(monkeypatch, new_query: str):
    """Make loop.call_llm_with_retry return an LLMResponse whose content is
    the reformulated query string.
    """
    def fake(**kwargs):
        return LLMResponse(
            content=new_query,
            tool_calls=[],
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=5,
        )
    monkeypatch.setattr(loop, "call_llm_with_retry", fake)


def _patch_tool(monkeypatch, name: str, result: ToolResult, calls: list):
    """Replace get_function(name) so the reformulated call runs against a
    fake tool and records its arguments for assertion.
    """
    def fake_tool(**kwargs):
        calls.append(kwargs)
        return result
    def fake_get_function(n):
        return fake_tool if n == name else None
    monkeypatch.setattr(loop, "get_function", fake_get_function)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_fires_on_empty_rag_retrieve(monkeypatch):
    state = AgentState(question="q", step_budget=10, step_n=1)
    calls: list = []
    _patch_llm(monkeypatch, new_query="different phrasing for same intent")
    _patch_tool(monkeypatch, "rag_retrieve", _ok_result(), calls)

    fired = loop._maybe_reformulate(
        state,
        tool_name="rag_retrieve",
        tool_args={"query": "original question", "top_k": 5},
        tool_result=_empty_result(),
    )

    assert fired is True
    assert state.counters.reformulations_attempted == 1
    assert state.step_n == 2
    # Reformulated call kept non-query kwargs, swapped query
    assert calls == [{"query": "different phrasing for same intent", "top_k": 5}]
    # One new OBSERVE step appended, carrying the 'reformulated' marker
    assert len(state.history) == 1
    assert "reformulated" in state.history[-1].notes


def test_fires_on_empty_arxiv_search(monkeypatch):
    state = AgentState(question="q", step_budget=10, step_n=1)
    calls: list = []
    _patch_llm(monkeypatch, new_query="broader search terms")
    _patch_tool(monkeypatch, "arxiv_search", _ok_result(), calls)

    fired = loop._maybe_reformulate(
        state,
        tool_name="arxiv_search",
        tool_args={"query": "original", "max_results": 5, "sort_by": "relevance"},
        tool_result=_empty_result(),
    )

    assert fired is True
    assert calls == [{"query": "broader search terms", "max_results": 5, "sort_by": "relevance"}]


def test_does_not_fire_for_non_retrieval_tool(monkeypatch):
    """fetch_paper's empty result is a different failure mode; don't reformulate."""
    state = AgentState(question="q", step_budget=10, step_n=1)
    _patch_llm(monkeypatch, new_query="won't be used")

    fired = loop._maybe_reformulate(
        state,
        tool_name="fetch_paper",
        tool_args={"arxiv_id": "2401.12345"},
        tool_result=_empty_result(),
    )

    assert fired is False
    assert state.counters.reformulations_attempted == 0
    assert state.step_n == 1
    assert state.history == []


def test_does_not_fire_on_non_empty_error(monkeypatch):
    """A TOOL_TIMEOUT is not a semantic failure — don't reformulate it."""
    state = AgentState(question="q", step_budget=10, step_n=1)
    _patch_llm(monkeypatch, new_query="won't be used")

    fired = loop._maybe_reformulate(
        state,
        tool_name="rag_retrieve",
        tool_args={"query": "original"},
        tool_result=_empty_result(kind=ToolErrorKind.TOOL_TIMEOUT),
    )

    assert fired is False
    assert state.counters.reformulations_attempted == 0


def test_fires_only_once_per_run(monkeypatch):
    """Second EMPTY_RESULT in the same run must not trigger another reformulation."""
    state = AgentState(question="q", step_budget=10, step_n=1)
    state.counters.reformulations_attempted = 1   # simulate prior reformulation
    _patch_llm(monkeypatch, new_query="won't be used")

    fired = loop._maybe_reformulate(
        state,
        tool_name="rag_retrieve",
        tool_args={"query": "original"},
        tool_result=_empty_result(),
    )

    assert fired is False
    assert state.counters.reformulations_attempted == 1   # unchanged
    assert state.step_n == 1


def test_identical_reformulation_is_rejected(monkeypatch):
    """If the model returns the same query (case-insensitive), don't re-run the
    tool — that would burn a step on a guaranteed empty result. Just record a
    skip note.
    """
    state = AgentState(question="q", step_budget=10, step_n=1)
    calls: list = []
    _patch_llm(monkeypatch, new_query="ORIGINAL QUERY")    # same, just caps
    _patch_tool(monkeypatch, "rag_retrieve", _ok_result(), calls)

    fired = loop._maybe_reformulate(
        state,
        tool_name="rag_retrieve",
        tool_args={"query": "original query"},
        tool_result=_empty_result(),
    )

    assert fired is False
    assert state.counters.reformulations_attempted == 0
    assert state.step_n == 1
    assert calls == []   # tool was never re-invoked
    assert len(state.history) == 1
    assert "skipped" in state.history[-1].notes


def test_reformulated_empty_result_is_recorded(monkeypatch):
    """If the reformulation itself returns EMPTY_RESULT, we don't retry a third
    time — the result is just recorded and the counter stays at 1 so no further
    reformulations fire for the rest of the run.
    """
    state = AgentState(question="q", step_budget=10, step_n=1)
    calls: list = []
    _patch_llm(monkeypatch, new_query="different query")
    _patch_tool(monkeypatch, "rag_retrieve", _empty_result(), calls)

    fired = loop._maybe_reformulate(
        state,
        tool_name="rag_retrieve",
        tool_args={"query": "original"},
        tool_result=_empty_result(),
    )

    assert fired is True
    assert state.counters.reformulations_attempted == 1
    assert state.step_n == 2
    assert len(calls) == 1
    # Last appended step carries the empty-result kind in the notes
    last = state.history[-1]
    assert "reformulated" in last.notes
    assert "empty_result" in last.notes
