"""Agent loop — explicit state machine.

Five states (PLAN → ACT → OBSERVE → REFLECT → ANSWER). Each transition is a
pure function of AgentState. The loop runs until status is terminal or the
step budget is exhausted.

Initial scope: happy path plus LLM retries. Rate limits, timeouts and 5xx
are retried in failures.call_llm_with_retry; anything else sets status to
FAILED. No repair, no reflection beyond "call ACT again" yet.

Why no framework: every state transition is one function.
Every tool call is one dispatch. The entire control flow fits in this file
and is unit-testable without mocking a framework.
"""

from __future__ import annotations

import json
from typing import Any

import groq

from arxiv_agent.failures import RetryCallback, call_llm_with_retry
from arxiv_agent.prompts import (
    ACT_SYSTEM_PROMPT,
    ANSWER_PROMPT,
    BUDGET_EXHAUSTED_ANSWER_PROMPT,
    PLANNER_PROMPT,
    REFORMULATE_QUERY_PROMPT,
)
from arxiv_agent.schema import (
    AgentState,
    AgentStateStatus,
    LoopState,
    StepRecord,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from arxiv_agent.tools import TOOLS, get_function, get_schemas
from arxiv_agent.llm import TruncatedResponseError, MalformedToolArgsError


# Hard safety cap on top of AgentState.step_budget. Prevents the loop from
# running forever if a transition function is buggy and never terminates.
_HARD_STEP_CEILING = 25

# Per-field caps when rendering tool results for the model. Every ACT call
# re-sends every earlier tool result, so these set how fast the prompt grows
# per step against Groq's 8K tokens/minute.
_MAX_RESULT_CHARS = 4500
_MAX_ITEM_TEXT_CHARS = 400

# Only the most recent tool results are shown in full; older ones become a
# one-line note. Without this the ACT prompt grows every step until a single
# request exceeds the 8K tokens/minute limit (HTTP 413, not retryable).
_KEEP_FULL_RESULTS = 3

# Retrieval tools whose EMPTY_RESULT triggers a one-shot query reformulation.
_REFORMULATABLE_TOOLS = frozenset({"rag_retrieve", "arxiv_search"})
_MAX_REFORMULATIONS_PER_RUN = 1


def _record_retry(state: AgentState, loop_state: LoopState) -> RetryCallback:
    """on_retry callback for call_llm_with_retry: counts the retry and leaves
    a trace line, so waits show up in the trace instead of looking like hangs.
    """
    def on_retry(err: Exception, attempt: int, wait: float) -> None:
        state.counters.retries_attempted += 1
        if isinstance(err, groq.RateLimitError):
            state.counters.rate_limited += 1
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=loop_state,
            notes=f"{type(err).__name__} on attempt {attempt}; waiting {wait:.1f}s",
        ))
    return on_retry


def run_agent(question: str, step_budget: int = 10) -> AgentState:
    """Entry point. Takes a question, returns a final AgentState.

    AgentState.status tells you the outcome; AgentState.final_answer, .citations,
    and .history are the artifacts. Never raises — any failure sets status to
    FAILED with the reason in the last StepRecord's notes.
    """
    state = AgentState(question=question, step_budget=step_budget)

    # PLAN once at the start. The plan is advisory; it exists to prime the
    # model and show up in traces, not as a contract the loop enforces.
    _plan(state)

    # Main loop. Each iteration advances state.step_n and may transition
    # status to terminal.
    while state.status == AgentStateStatus.RUNNING:
        if state.budget_exhausted() or state.step_n >= _HARD_STEP_CEILING:
            state.counters.budget_exhausted = True
            state.history.append(StepRecord(
                step_n=state.step_n,
                state=LoopState.DONE,
                notes=f"step budget exhausted at step_n={state.step_n}",
            ))
            # If we have any evidence at all, emit an honest partial answer
            # with explicit uncertainty instead of silently failing.
            _emit_partial_answer_on_budget_exhaustion(state)
            if state.status == AgentStateStatus.RUNNING:
                state.status = AgentStateStatus.FAILED
            break

        _step(state)

    # If we fell out with status still RUNNING, mark it failed. Should not
    # happen with current code, but defensive.
    if state.status == AgentStateStatus.RUNNING:
        state.status = AgentStateStatus.FAILED

    return state


def _plan(state: AgentState) -> None:
    try:
        resp = call_llm_with_retry(
            on_retry=_record_retry(state, LoopState.PLAN),
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": state.question},
            ],
            temperature=0.0,
            max_tokens=600,
            allow_truncation=True,
        )
    except Exception as e:
        # Weekend 1: PLAN failure terminates the run cleanly. Weekend 2 adds
        # retry logic — a failed plan is recoverable (the question hasn't
        # changed, model state is independent).
        state.status = AgentStateStatus.FAILED
        state.history.append(StepRecord(
            step_n=0,
            state=LoopState.DONE,
            notes=f"PLAN raised {type(e).__name__}: {str(e)[:200]}",
        ))
        return
    state.plan = (resp.content or "").strip()
    plan_truncated = resp.finish_reason == "length"
    state.history.append(StepRecord(
        step_n=0,
        state=LoopState.PLAN,
        latency_ms=resp.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        notes="initial plan (truncated)" if plan_truncated else "initial plan",
    ))


def _step(state: AgentState) -> None:
    """One iteration: ACT → (OBSERVE → REFLECT) OR direct to ANSWER.

    The LLM either emits a tool_call (continue the loop) or emits content
    with no tool_call (treat as answer, terminate).
    """
    state.step_n += 1
    messages = _build_act_messages(state)

    try:
        resp = call_llm_with_retry(
            on_retry=_record_retry(state, LoopState.ACT),
            messages=messages,
            tools=get_schemas(),
            temperature=0.0,
            max_tokens=1024,
        )
    except TruncatedResponseError as e:
        # terminate cleanly on ACT truncation. 
        state.status = AgentStateStatus.FAILED
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.DONE,
            notes=f"ACT truncated at max_tokens={e.max_tokens}",
        ))
        return
    except MalformedToolArgsError as e:
        # terminate. 
        state.status = AgentStateStatus.FAILED
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.DONE,
            notes=f"ACT malformed tool args for {e.tool_name}: {e.parse_error}",
        ))
        return
    except groq.BadRequestError as e:
        if _record_rejected_tool_call(state, e):
            return
        state.status = AgentStateStatus.FAILED
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.DONE,
            notes=f"ACT raised {type(e).__name__}: {str(e)[:200]}",
        ))
        return
    except Exception as e:
        # Transient errors were already retried in call_llm_with_retry; what
        # reaches here is out of retries or not retryable. Terminate cleanly.
        state.status = AgentStateStatus.FAILED
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.DONE,
            notes=f"ACT raised {type(e).__name__}: {str(e)[:200]}",
        ))
        return

    # Short-circuit: no tool call means the model is ready to answer.
    if not resp.tool_calls:
        _finalize_answer(state, resp.content or "", act_latency_ms=resp.latency_ms,
                         prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens)
        return

    # Weekend 1 handles one tool call per step. The loop (and failures.py in
    # Weekend 2) will handle batched calls then. Document the discard.
    if len(resp.tool_calls) > 1:
        extras = len(resp.tool_calls) - 1
    else:
        extras = 0
    tc = resp.tool_calls[0]

    # OBSERVE — dispatch the tool.
    tool_fn = get_function(tc.name)
    if tool_fn is None:
        # Model hallucinated a tool name. Weekend 1: mark as fatal and let
        # it bubble into REFLECT, which will likely terminate.
        tool_result = ToolResult(
            ok=False,
            error=None,  # cannot construct a ToolError here without rewriting ToolResult invariants
            latency_ms=0,
        )
        # ToolResult invariants require error on ok=False. Build it properly:
        from arxiv_agent.schema import ToolError, ToolErrorKind
        tool_result = ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.TOOL_ERROR_FATAL,
                message=f"unknown tool: {tc.name!r}",
                retryable=False,
            ),
            latency_ms=0,
        )
    else:
        tool_result = tool_fn(**tc.args)

    note_parts = [f"tool={tc.name}"]
    if extras:
        note_parts.append(f"discarded_extra_tool_calls={extras}")
    if tool_result.ok is False and tool_result.error is not None:
        note_parts.append(f"error_kind={tool_result.error.kind.value}")

    state.history.append(StepRecord(
        step_n=state.step_n,
        state=LoopState.OBSERVE,
        tool_name=tc.name,
        tool_args=tc.args,
        result=tool_result,
        latency_ms=resp.latency_ms + tool_result.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        notes=" | ".join(note_parts),
    ))

    # One-shot query reformulation on EMPTY_RESULT from a retrieval tool.
    # Records its own OBSERVE step and consumes one step from the budget.
    _maybe_reformulate(state, tc.name, tc.args, tool_result)

    # Weekend 1 REFLECT is implicit: the next loop iteration will call ACT
    # again with the updated history, which lets the model see the tool result
    # and decide whether to call another tool or answer. A proper separate
    # REFLECT state lands in Weekend 2's ablation (`baseline+reflect`).


def _record_rejected_tool_call(state: AgentState, err: groq.BadRequestError) -> bool:
    """Groq validates tool-call arguments against the schema server-side and
    answers 400 'tool_use_failed' when they don't match. Turn that into a
    MALFORMED_TOOL_ARGS observation so the model sees the validation message
    on the next ACT and can correct the call, instead of ending the run.

    Returns False if the 400 was something else, so the caller fails as before.
    """
    body = err.body if isinstance(err.body, dict) else {}
    detail = body.get("error", body)
    if not isinstance(detail, dict) or detail.get("code") != "tool_use_failed":
        return False

    name, args = "unknown_tool", {}
    try:
        generated = json.loads(detail.get("failed_generation") or "")
        name = generated.get("name") or name
        if isinstance(generated.get("arguments"), dict):
            args = generated["arguments"]
    except (ValueError, AttributeError):
        pass

    state.counters.malformed_tool_args += 1
    state.history.append(StepRecord(
        step_n=state.step_n,
        state=LoopState.OBSERVE,
        tool_name=name,
        tool_args=args,
        result=ToolResult(
            ok=False,
            error=ToolError(
                kind=ToolErrorKind.MALFORMED_TOOL_ARGS,
                message=str(detail.get("message", "tool call rejected by the API"))[:500],
                retryable=True,
            ),
        ),
        notes=f"tool={name} | error_kind=malformed_tool_args (rejected by API)",
    ))
    return True


def _build_act_messages(state: AgentState) -> list[dict[str, Any]]:
    """Construct the message list for ACT.

    Shape:
      system: ACT system prompt
      user:   the original question
      assistant/tool pairs: one per tool call in history
      user (final): a tiny nudge if nothing has happened yet

    OpenAI-compatible tool-use requires that every assistant message with
    tool_calls is followed by one tool message per call. We synthesize those
    from StepRecords.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ACT_SYSTEM_PROMPT},
    ]

    # Fold the plan into the user message so it stays visible without burning
    # a separate turn.
    user_content = state.question
    if state.plan:
        user_content = f"{state.question}\n\nYour plan:\n{state.plan}"
    messages.append({"role": "user", "content": user_content})

    observed = [s for s in state.history if s.state == LoopState.OBSERVE and s.tool_name is not None]
    full_from = max(0, len(observed) - _KEEP_FULL_RESULTS)

    for i, step in enumerate(observed):
        tool_call_id = f"call_{step.step_n}"
        # Assistant message that "made" the call. We fabricate it from the
        # recorded tool_name/tool_args; the model never sees its own earlier
        # assistant turn because we don't persist the raw response.
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": step.tool_name,
                    "arguments": json.dumps(step.tool_args or {}),
                },
            }],
        })
        # Tool message with the result.
        if i < full_from:
            result_summary = "[older result omitted to save tokens; call the tool again if you need it]"
        else:
            result_summary = _summarize_tool_result_for_model(step.result)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_summary,
        })

    return messages


def _summarize_tool_result_for_model(result: ToolResult | None) -> str:
    """Render a ToolResult as a compact string for the tool message.

    The model needs enough signal to decide the next step, not the full
    raw tool output. For large results (fetch_paper's sections, 10 chunks
    from rag_retrieve), truncate here to protect the token budget — Weekend 4
    adds proper compaction.
    """
    if result is None:
        return "tool returned nothing"
    if not result.ok:
        err = result.error
        return f"TOOL ERROR [{err.kind.value}]: {err.message}" if err else "TOOL ERROR (unspecified)"
    data = _compact(result.data or {})
    try:
        rendered = json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = repr(data)
    if len(rendered) > _MAX_RESULT_CHARS:
        rendered = rendered[:_MAX_RESULT_CHARS] + "...[truncated]"
    return rendered


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    """Shorten long text fields per item before the overall cap applies.

    Cutting the whole JSON at a fixed length showed the model the first two
    of ten chunks and nothing of the rest. Capping each chunk, abstract or
    section keeps every item visible.
    """
    def short(text: Any) -> Any:
        if isinstance(text, str) and len(text) > _MAX_ITEM_TEXT_CHARS:
            return text[:_MAX_ITEM_TEXT_CHARS] + "...[truncated]"
        return text

    out = dict(data)
    if isinstance(out.get("results"), list):
        out["results"] = [
            {k: short(v) for k, v in item.items() if k not in ("dense_score",)}
            if isinstance(item, dict) else item
            for item in out["results"]
        ]
    if isinstance(out.get("sections"), list):
        out["sections"] = [
            {k: short(v) for k, v in sec.items()} if isinstance(sec, dict) else sec
            for sec in out["sections"]
        ]
    return out


def _finalize_answer(
    state: AgentState,
    assistant_content: str,
    act_latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """ANSWER state. The model's last response was a tool-free message, which
    we treat as the final answer. One more LLM call to produce a cleaned
    final answer informed by the full history, since the ACT-time content is
    often terse ("I'm ready to answer.") rather than the answer itself.
    """
    # If the ACT response already looks like a substantive answer (>80 chars),
    # use it directly. Otherwise do a dedicated ANSWER pass.
    if len(assistant_content.strip()) >= 80:
        final = assistant_content.strip()
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.ANSWER,
            latency_ms=act_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            notes="answer produced in-line during ACT",
        ))
    else:
        messages = _build_act_messages(state) + [
            {"role": "system", "content": ANSWER_PROMPT},
        ]
        try:
            resp = call_llm_with_retry(
                on_retry=_record_retry(state, LoopState.ANSWER),
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            state.status = AgentStateStatus.FAILED
            state.history.append(StepRecord(
                step_n=state.step_n,
                state=LoopState.DONE,
                notes=f"ANSWER raised {type(e).__name__}: {str(e)[:200]}",
            ))
            return
        final = (resp.content or "").strip()
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.ANSWER,
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            notes="answer produced by dedicated ANSWER call",
        ))

    state.final_answer = final
    state.citations = _extract_citations(final, state.history)
    # Status: ANSWERED unless the model explicitly admitted insufficient evidence.
    if _is_insufficient(final):
        state.status = AgentStateStatus.INSUFFICIENT
    else:
        state.status = AgentStateStatus.ANSWERED


_ARXIV_ID_IN_TEXT = __import__("re").compile(
    r"\[((?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?))\]"
)


def _extract_citations(answer: str, history: list[StepRecord]) -> list[str]:
    """Pull [arxiv_id] citations out of the answer text. Dedupe. Only include
    IDs that actually appeared in a tool result — guards against models that
    fabricate IDs.
    """
    seen_in_history: set[str] = set()
    for step in history:
        if step.result and step.result.ok and step.result.data:
            data = step.result.data
            # rag_retrieve → list of chunks with arxiv_id
            for c in data.get("results", []) or []:
                aid = c.get("arxiv_id")
                if aid:
                    seen_in_history.add(aid)
            # fetch_paper → arxiv_id at top level
            if data.get("arxiv_id"):
                seen_in_history.add(data["arxiv_id"])
            # citation_check → supporting_arxiv_ids
            for aid in data.get("supporting_arxiv_ids", []) or []:
                if aid:
                    seen_in_history.add(aid)

    cited = []
    seen = set()
    for m in _ARXIV_ID_IN_TEXT.finditer(answer):
        aid = m.group(1)
        if aid in seen or aid not in seen_in_history:
            continue
        cited.append(aid)
        seen.add(aid)
    return cited


def _is_insufficient(answer: str) -> bool:
    """Heuristic: did the model say the evidence is insufficient?

    Weekend 1 is a keyword check; Weekend 3 may replace with an LLM judge.
    """
    needles = (
        "insufficient evidence",
        "not enough information",
        "cannot answer",
        "unable to answer",
        "no sources support",
    )
    lowered = answer.lower()
    return any(n in lowered for n in needles)


def _maybe_reformulate(
    state: AgentState,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    tool_result: ToolResult,
) -> bool:
    """If a retrieval tool returned EMPTY_RESULT and we have not yet reformulated
    this run, ask the LLM for a different query and re-execute the same tool.
    Records a new OBSERVE step. Returns True if reformulation fired.

    Design (interview-answerable):
    - Fires at most _MAX_REFORMULATIONS_PER_RUN (currently 1). An agent that
      cannot answer on its original + one rephrase should fall through to
      "insufficient evidence" instead of thrashing on query variants.
    - Only applies to query-driven retrieval tools (rag_retrieve, arxiv_search).
      Reformulating fetch_paper's arxiv_id is a different failure mode.
    - Separate recorded step, not silent — the trace shows that reformulation
      happened, which makes failure handling visible in evals.
    - Consumes one step from the budget. Honest accounting.
    """
    if state.counters.reformulations_attempted >= _MAX_REFORMULATIONS_PER_RUN:
        return False
    if tool_name not in _REFORMULATABLE_TOOLS:
        return False
    if tool_result.ok:
        return False
    if tool_result.error is None or tool_result.error.kind != ToolErrorKind.EMPTY_RESULT:
        return False

    original_query = (tool_args or {}).get("query", "")
    if not isinstance(original_query, str) or not original_query.strip():
        return False

    prompt = REFORMULATE_QUERY_PROMPT.format(
        tool_name=tool_name,
        original_query=original_query,
    )
    try:
        resp = call_llm_with_retry(
            on_retry=_record_retry(state, LoopState.OBSERVE),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
    except Exception as e:
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.OBSERVE,
            notes=f"reformulate llm failed: {type(e).__name__}",
        ))
        return False

    new_query = (resp.content or "").strip().strip('"').strip("'")
    if not new_query or new_query.lower() == original_query.lower():
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.OBSERVE,
            notes="reformulate produced empty-or-identical query; skipped",
        ))
        return False

    state.counters.reformulations_attempted += 1

    new_args = dict(tool_args or {})
    new_args["query"] = new_query

    state.step_n += 1

    tool_fn = get_function(tool_name)
    if tool_fn is None:
        return False
    reformulated_result = tool_fn(**new_args)

    note_parts = [f"tool={tool_name}", "reformulated"]
    if reformulated_result.ok is False and reformulated_result.error is not None:
        note_parts.append(f"error_kind={reformulated_result.error.kind.value}")

    state.history.append(StepRecord(
        step_n=state.step_n,
        state=LoopState.OBSERVE,
        tool_name=tool_name,
        tool_args=new_args,
        result=reformulated_result,
        latency_ms=resp.latency_ms + reformulated_result.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        notes=" | ".join(note_parts),
    ))
    return True


def _has_successful_tool_evidence(state: AgentState) -> bool:
    """True if the trace contains at least one OBSERVE step whose tool returned ok."""
    for step in state.history:
        if step.state == LoopState.OBSERVE and step.result is not None and step.result.ok:
            return True
    return False


def _emit_partial_answer_on_budget_exhaustion(state: AgentState) -> None:
    """Called when the step budget is exhausted.

    If the trace contains at least one successful tool result, do one final
    LLM call to produce an honest partial answer flagged with uncertainty,
    and set status to INSUFFICIENT. Otherwise leave status RUNNING — the
    run_agent caller will convert it to FAILED (nothing to summarize honestly).

    Interview answer: a FAILED status with no final_answer is interchangeable
    in metrics with a fundamentally-wrong answer. INSUFFICIENT with a partial
    answer flagged by an UNCERTAIN prefix lets the eval judge tell "ran out of
    budget with some evidence" apart from "could not understand the question."
    Different failure mode, different handling downstream.
    """
    if not _has_successful_tool_evidence(state):
        # No evidence to summarize honestly. Stay RUNNING; caller sets FAILED.
        return

    messages = _build_act_messages(state) + [
        {"role": "system", "content": BUDGET_EXHAUSTED_ANSWER_PROMPT},
    ]
    try:
        resp = call_llm_with_retry(
            on_retry=_record_retry(state, LoopState.ANSWER),
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            allow_truncation=True,  # a truncated honest answer still beats no answer
        )
    except Exception as e:
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.ANSWER,
            notes=f"budget-exhausted answer llm failed: {type(e).__name__}",
        ))
        return

    final = (resp.content or "").strip()
    if not final:
        state.history.append(StepRecord(
            step_n=state.step_n,
            state=LoopState.ANSWER,
            notes="budget-exhausted answer was empty",
        ))
        return

    state.final_answer = final
    state.citations = _extract_citations(final, state.history)
    state.status = AgentStateStatus.INSUFFICIENT
    state.history.append(StepRecord(
        step_n=state.step_n,
        state=LoopState.ANSWER,
        latency_ms=resp.latency_ms,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        notes="partial answer on budget exhaustion",
    ))

