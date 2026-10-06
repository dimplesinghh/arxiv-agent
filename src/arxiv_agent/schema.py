"""Core data structures for the agent.

Every tool returns a ToolResult. The agent loop maintains an AgentState.
Every step of the loop appends a StepRecord to the history. All errors are
values (ToolError), not exceptions — tools never raise into the loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolErrorKind(str, Enum):
    """Failure taxonomy. Every tool failure maps to exactly one kind.
    """

    TOOL_TIMEOUT = "tool_timeout"                # tool exceeded its timeout budget
    RATE_LIMITED = "rate_limited"                # 429 from external API (Groq, arxiv)
    TOOL_ERROR_RETRYABLE = "tool_error_retryable"  # network blip, transient service error
    TOOL_ERROR_FATAL = "tool_error_fatal"        # bad input, no retry helps (e.g. arxiv ID doesn't exist)
    MALFORMED_TOOL_ARGS = "malformed_tool_args"  # LLM emitted invalid JSON or wrong param names
    TRUNCATED = "truncated" 
    EMPTY_RESULT = "empty_result"                # tool ran fine but found nothing relevant


class ToolError(BaseModel):
    """Structured error returned by a tool.

    `retryable` is the single source of truth for whether the failure handler
    should retry. Do not infer retryability from `kind` at call sites.
    """

    kind: ToolErrorKind
    message: str
    retryable: bool = False


class ToolResult(BaseModel):
    """Uniform return type for every tool.

    Exactly one of (data, error) is populated. `ok` is the fast-path check:
    if ok, use data; if not, hand error to the failure handler.
    """

    ok: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[ToolError] = None
    latency_ms: int = 0

    def model_post_init(self, _ctx) -> None:
        # Invariants — bugs here corrupt every downstream metric.
        if self.ok and self.error is not None:
            raise ValueError("ToolResult.ok=True must not have error set")
        if not self.ok and self.error is None:
            raise ValueError("ToolResult.ok=False must have error set")


class AgentStateStatus(str, Enum):
    RUNNING = "running"
    ANSWERED = "answered"        # produced a final answer (may or may not be correct)
    FAILED = "failed"            # gave up — usually budget exhausted or unrecoverable
    INSUFFICIENT = "insufficient"  # answered honestly: not enough evidence


class LoopState(str, Enum):
    """States of the agent state machine. Every transition is testable."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    ANSWER = "answer"
    DONE = "done"


class StepRecord(BaseModel):
    """One iteration of the loop. Appended to history on every OBSERVE.

    This is the primary artifact for tracing, debugging, and evals — the eval
    harness reads these to compute per-step and per-tool metrics.
    """

    step_n: int
    state: LoopState
    tool_name: Optional[str] = None       # None for pure LLM steps (PLAN, REFLECT, ANSWER)
    tool_args: Optional[dict[str, Any]] = None
    result: Optional[ToolResult] = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    notes: Optional[str] = None           # human-readable trace note, e.g. "retry 1/3"


class FailureCounters(BaseModel):
    """Per-failure-mode tallies for a single agent run.
    """

    tool_timeout: int = 0
    rate_limited: int = 0
    tool_error_retryable: int = 0
    tool_error_fatal: int = 0
    malformed_tool_args: int = 0
    empty_result: int = 0
    repairs_attempted: int = 0
    retries_attempted: int = 0
    budget_exhausted: bool = False
    truncated: int = 0

    def bump(self, kind: ToolErrorKind) -> None:
        setattr(self, kind.value, getattr(self, kind.value) + 1)


class AgentState(BaseModel):
    """The one object the loop mutates. Everything else is a pure function of it."""

    question: str
    plan: Optional[str] = None
    history: list[StepRecord] = Field(default_factory=list)
    step_budget: int = 10
    step_n: int = 0
    status: AgentStateStatus = AgentStateStatus.RUNNING
    final_answer: Optional[str] = None
    citations: list[str] = Field(default_factory=list)  # arxiv_ids or chunk refs
    counters: FailureCounters = Field(default_factory=FailureCounters)

    def budget_remaining(self) -> int:
        return max(0, self.step_budget - self.step_n)

    def budget_exhausted(self) -> bool:
        return self.step_n >= self.step_budget

# ~4 chars/token is the standard rough estimate for English prose. Close enough
# for budget enforcement; we don't need tokenizer-exact counts here.
CHARS_PER_TOKEN_ESTIMATE = 4

def approx_token_count(text: str) -> int:
    """Rough token estimate from character count. Within ~15% for English prose.
    Use for budget guardrails, not for anything that requires exact counts.
    """
    return len(text) // CHARS_PER_TOKEN_ESTIMATE