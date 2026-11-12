"""Runtime feature flags for ablation runs.

Read via environment variables — the eval harness sets these before any
agent call, so individual questions in one run all see the same config.
Default (all flags unset) is the baseline configuration.
"""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_VALUES


def no_failure_handling() -> bool:
    """When true: the loop uses raw llm.call_llm (no retries, no backoff),
    skips reformulation, and skips budget-exhausted partial answers. Isolates
    the effect of the failure-handling layer in ablation runs.
    """
    return _env_flag("ARXIV_AGENT_NO_FAILURE_HANDLING")


def no_rerank() -> bool:
    """When true: rag_retrieve forces use_rerank=False regardless of caller
    argument. Isolates the effect of the cross-encoder reranker.
    """
    return _env_flag("ARXIV_AGENT_NO_RERANK")


def active_summary() -> dict[str, bool]:
    """Snapshot of current flag state — attached to per-question eval JSON
    so each result records exactly which config it ran under.
    """
    return {
        "no_failure_handling": no_failure_handling(),
        "no_rerank": no_rerank(),
    }
