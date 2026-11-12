# decisions.md

Architecture decision records for `arxiv-agent`. Each entry: context, decision, alternatives considered, tradeoffs, honest limitations. Living document, updated as the design evolves.

---

## 1. No agent framework

**Context.** Most 2026 agent tutorials start by importing LangGraph, LangChain, or AutoGen. Each framework hides the control flow — state transitions, tool dispatch, failure routing — inside its own abstraction. For a project whose focus is explicit failure handling, that level of indirection removes the thing worth looking at.

**Decision.** The loop, tool registry, and failure handlers are written from scratch in ~900 lines across `loop.py`, `failures.py`, and `tools/`.

**Alternatives considered.**
- LangGraph — used elsewhere in the portfolio, so this project needs differentiation.
- LangChain — heavy abstractions over tool-call lifecycle.
- Instructor + hand-rolled loop — plausible but adds a dependency for little saved code.

**Tradeoff.** More code to maintain. No community glue. Manual message construction. In exchange: every state transition is a pure function with unit tests, every tool call is one dispatch, every design choice is readable in one file.

**Limitation.** At 10+ tools or parallel tool-call batching, framework routing primitives start paying off. Noted as the sunset condition for the from-scratch decision.

---

## 2. Explicit state machine, not free-form ReAct

**Context.** The common ReAct pattern loops "thought → action → observation" in free-form text and lets the model decide when to stop. That's difficult to test because every step is an unstructured LLM response.

**Decision.** Four explicit states — `PLAN → ACT → OBSERVE → ANSWER`. The loop controls transitions; the model signals "done" by returning a response with no `tool_calls`.

**Alternatives considered.**
- Free-form ReAct — flexible, untestable.
- Planner-executor split with separate models — doubles LLM calls for marginal gain at this scale.

**Tradeoff.** Less "autonomous" — the agent can't re-plan mid-execution without going through ACT again. In exchange: each state is a testable function, `StepRecord` captures state cleanly, traces are grep-able.

**Limitation.** No separate `REFLECT` state between OBSERVE and the next ACT. Implicit reflection happens on the next ACT call since full history is replayed. A dedicated REFLECT prompt is a natural extension and would make a clean ablation.

---

## 3. Errors as values at the tool boundary, typed exceptions at the LLM boundary

**Context.** Two distinct failure surfaces: tools (`rag_retrieve`, `fetch_paper`) can fail — retrieval empty, API timeout, bad PDF — and LLM calls can fail — truncation, malformed tool-call JSON, rate limit. These need different control flows.

**Decision.**
- Tools never raise. They return `ToolResult(ok, data, error)` with a typed `ToolError(kind, message, retryable)`. `ToolErrorKind` is an enum that drives counter buckets. Invariant: `ok=True` must have no error, `ok=False` must have one, enforced in `model_post_init`.
- LLM calls raise typed exceptions — `TruncatedResponseError`, `MalformedToolArgsError` — at specific sites the loop catches explicitly. Other Groq SDK exceptions propagate through `failures.call_llm_with_retry`.

**Alternatives considered.**
- Pydantic `Result` type throughout — adds dependency for little benefit.
- `try/except` everywhere — hides control flow.
- Union return types — same shape, weaker typing.

**Tradeoff.** The loop inspects `tool_result.ok` and `tool_result.error.kind` to route decisions. In exchange: a compile-time guarantee that no tool call blows up the loop; the agent can observe and reason about failures; every error has a counter for mechanical aggregation in evals.

**Limitation.** `ToolErrorKind` as an enum means new failure modes require edits to `schema.py` and every handler. Discoverable, less extensible. A `kind: str` would be looser but harder to count mechanically.

---

## 4. One source of retry logic in `failures.py`; SDK retries disabled

**Context.** The Groq Python SDK retries connection errors, 408s, 409s, 429s, and 5xx responses twice by default with exponential backoff and a 60s timeout. A retry wrapper layered on top would stack: SDK does 3, wrapper does 4, actual attempts up to 12. `latency_ms` would include the SDK's hidden waits.

**Decision.** `Groq(max_retries=0, timeout=30.0)`. All retry logic lives in `failures.call_llm_with_retry`. One place to read, one place to test.

**Alternatives considered.**
- Layer custom retries on top of SDK retries — opaque, untestable.
- Rely only on SDK retries — no `retry-after` header parsing, no visible counters.
- Retry inside `llm.py` — mixes the raw call surface with retry policy.

**Tradeoff.** Lose the SDK's "free" retry behavior; own every retry decision. In exchange: `failures.py` tests are deterministic (`time.sleep` and `llm.call_llm` are monkeypatched); handler logic parses `retry-after` and waits exactly what the server asked; `retries_attempted` in `AgentState.counters` is accurate.

**Limitation.** If Groq ships new retry semantics (DNS, TLS edge cases), they need to be added manually here.

---

## 5. Groq-hosted `openai/gpt-oss-120b` over a frontier model

**Context.** A reliable model like Claude or GPT-5 would hide the failure modes this project is built around — 429s, malformed JSON, inconsistent tool calls. The handlers in `failures.py` would be untestable in real conditions.

**Decision.** Groq's free tier: `openai/gpt-oss-120b`. 8K tokens/minute, 200K tokens/day. The TPM cap is tight enough that a multi-step agent hits it mid-run on most questions.

**Alternatives considered.**
- Anthropic / OpenAI frontier APIs — reliable, hides failure modes, costs money.
- Local model via Ollama — free, but inference latency and tool-call quality are weak on consumer hardware.
- Groq's higher-tier plans — same model, higher caps, defeats the purpose.

**Tradeoff.** Lower raw answer quality. Occasional malformed JSON in tool calls. Open-weights models sometimes ignore "return JSON only" and wrap output in markdown fences (handled explicitly in `citation_check`). In exchange: real 429s in every eval run, real backoff waits in traces, real rate-limit counters.

**Limitation.** Headline pass rates will be lower than a Claude- or GPT-5-based baseline would be. Documented. When results are written up, agent-architecture failures (handler misbehavior) and model-quality failures (hallucinated citations, malformed JSON) are reported separately.

---

## 6. Qdrant index reused from `arxiv-rag`, not duplicated

**Context.** The sibling project `arxiv-rag` built a Qdrant index with a specific chunking strategy and embedding model. Rebuilding it here means hours of re-embedding, duplicate Docker state, and silent drift risk if the two diverge.

**Decision.** This repo reads the `arxiv_chunks` collection from Project 1's Qdrant instance directly. Startup requires Project 1's Docker compose stack to be up.

**Alternatives considered.**
- Duplicate the index — wastes disk, forks the data.
- Shared Qdrant Cloud instance — paid dependency for a free portfolio project.
- Mount Project 1's volume read-only — fragile across repos.

**Tradeoff.** Someone cloning this repo must also clone and start Project 1. In exchange: embedding model, chunking strategy, and collection metadata are single-sourced; changes propagate without manual migration.

**Limitation.** The dependency is undocumented in Project 1's README (it doesn't know this project exists). Documented in this repo's README instead.

---

## 7. One-shot query reformulation on `EMPTY_RESULT`

**Context.** Retrieval tools occasionally return nothing because the query uses the wrong vocabulary — "transformer efficiency" might miss a paper that calls itself "accelerator-optimized inference." Giving up on empty results produces worse answers; reformulating aggressively thrashes and wastes budget.

**Decision.** On `EMPTY_RESULT` from `rag_retrieve` or `arxiv_search`, the loop calls the LLM once to produce a reformulated query, then re-runs the same tool. Hard-capped at one reformulation per agent run, tracked in `state.counters.reformulations_attempted`.

**Alternatives considered.**
- No reformulation — agent sees the empty result and reasons about it.
- Multi-reformulation up to budget — thrashes.
- Reformulation inside the tool itself — invisible to the trace.

**Tradeoff.** One retrieval attempt could still miss a paper because the second query is also wrong. In exchange: common vocabulary mismatches are handled; the behavior is bounded; the reformulation is a recorded step in the trace.

**Limitation.** The reformulation prompt is generic. A domain-aware reformulator (e.g. "arXiv cs.LG papers from July 2026") would do better. Deferred.

---

## 8. Budget-exhausted partial-answer disclosure with `UNCERTAIN:` prefix

**Context.** When step budget hits zero, the naive design just sets `status = FAILED`. In eval metrics, that's indistinguishable from a question the agent never understood. Operationally they're different: "ran out of budget with partial evidence" is recoverable (more budget, different approach); "misunderstood the question" is not.

**Decision.** If the trace has at least one successful tool result when budget hits zero, the agent makes one final LLM call to produce a partial answer prefixed with `UNCERTAIN:`. Status becomes `INSUFFICIENT`. If no evidence at all, status stays `FAILED`.

**Alternatives considered.**
- Always emit a partial answer — invites fabrication when no evidence exists.
- Always FAILED — loses the signal.
- Let the model decide — unreliable; models often confabulate rather than admit insufficiency.

**Tradeoff.** One extra LLM call at the end of budget-exhausted runs. In exchange: `INSUFFICIENT` is a semantically useful third status; the eval judge can distinguish partial from no-answer; the `UNCERTAIN:` prefix is a mechanical signal eval graders can trust.

**Limitation.** The judge has to grade `UNCERTAIN:`-prefixed answers on what they do say, not penalize for what they admit. The judge prompt handles this but judge-agreement on the distinction isn't measured yet.

---

## 9. LLM-as-judge for eval grading, same model as the agent

**Context.** 40 questions × 3 ablation configs = 120 answers to grade. Manual grading doesn't scale; keyword matching is too brittle for semantic `success_criteria`.

**Decision.** `evals/run_eval.py` calls the same Groq-hosted model as the agent, with a strict system prompt requiring a structured JSON verdict (pass/fail + reason + criteria_met/missed). Temperature 0.0.

**Alternatives considered.**
- Stronger external judge (Anthropic / OpenAI) — better quality, adds cost, shifts a methodology choice this project explicitly owns.
- Human grading — accurate, doesn't scale.
- Keyword/regex grading — too brittle.

**Tradeoff.** Judge may share systematic biases with the agent (same model, same training). A judge that passes too easily inflates results. In exchange: no new dependency, fast, and criteria are LLM-judgeable. For a comparative ablation, judge bias cancels out across configs as long as it's consistent.

**Limitation.** Judge-grader agreement unmeasured. Planning a 10-sample manual spot-check after baseline run.

---

## 10. 50-paper index instead of the planned 300

**Context.** Project 1 was designed to index 300 arXiv cs.LG/cs.AI papers. Local embedding runs over 300 papers caused laptop thermal issues.

**Decision.** Reduced to 50 papers from July 2026. The 40-question eval set is written against this specific corpus.

**Alternatives considered.**
- Push through 300 at cost of hardware wear.
- Use a cloud embedding runner (Modal, Beam) for a one-off job.
- Reduce chunk size to fit 300 more cheaply.

**Tradeoff.** Smaller corpus, fewer natural question targets, less retrieval variety. In exchange: the index exists, the eval runs, and the architecture is unchanged — scaling to 300 later is a one-command re-embed.

**Limitation.** Topic coverage of the 50 papers isn't uniform; some subfields are over-represented, others sparse. Adversarial out-of-corpus questions use this honestly, but multi-paper comparative questions sometimes have fewer good pairs to draw from.

---

## 11. `fetch_paper` downloads via `urllib`, not the `arxiv` package

**Context.** The `arxiv` package removed PDF downloads in version 4.0. It's now metadata-only and exposes `pdf_url` for the caller to fetch.

**Decision.** `fetch_paper` uses `arxiv.Client` for metadata (title, pdf_url, 3s rate limit) and downloads the PDF via `urllib.request.urlopen` with an explicit User-Agent header.

**Alternatives considered.**
- Pin an older `arxiv` version — brittle, misses security patches.
- Add `requests` or `httpx` — unnecessary dependency for one GET.

**Tradeoff.** ~15 extra lines vs a one-line SDK call. A small risk of upstream behavior changes in arXiv's PDF service. In exchange: no dependency pin, no surprise from future `arxiv` package bumps, no extra HTTP library.

**Limitation.** The `arxiv` package rate limits the API endpoint but not the PDF endpoint. Occasional PDF fetches are tolerated but batch fetches could get IP-rate-limited. Not seen in practice yet.

---

## 12. `citation_check` returns structured JSON, not freeform text

**Context.** The agent needs a mechanical signal about whether a specific claim is backed by retrieved chunks. "I think this is probably supported..." is unusable; the loop needs `supported` / `partially_supported` / `unsupported` to decide whether to commit to the claim or re-retrieve.

**Decision.** `citation_check` prompts for a JSON object with a three-value verdict plus per-source rationales. Output parser strips markdown fences defensively and validates the verdict against an allowlist. On parse failure, returns `TOOL_ERROR_RETRYABLE`.

**Alternatives considered.**
- Binary pass/fail — collapses the "topic matches but specifics don't" case into a fail.
- Free-form rationale with keyword-search parsing — brittle.
- Pydantic schema via `instructor` — adds dependency for one call site.

**Tradeoff.** More prompt engineering to get `gpt-oss-120b` to emit clean JSON. Occasional parse failures trigger `TOOL_ERROR_RETRYABLE`. In exchange: a three-value verdict that captures the useful partial case; per-source rationales usable for debugging; mechanical integration with the ANSWER state.

**Limitation.** Testing suggests `gpt-oss-120b` is conservative on this — it returns `unsupported` where a human might say `partially_supported`. Will revisit after baseline results.

---

## Known limitations

Separate from design decisions. Honest shortcomings, not deliberate choices.

- **Multi-tool-call batching unsupported.** When the model emits parallel tool calls in one response, the loop takes the first and discards the rest. Rare on `gpt-oss-120b` but a real gap.
- **Judge model == agent model.** Same-model bias isn't mitigated structurally; relies on manual spot-checks post-hoc.
- **50-paper index.** Hardware constraint, not architectural.
- **No streaming.** One full completion per LLM call.
- **Token estimation is `len(text) // 4`.** Rough — drifts on JSON-heavy content and non-English. Fine for budget guardrails, not exact enforcement.
- **No HTTP cache for `arxiv_search`.** Repeated queries re-fetch. Low impact because the agent prefers `rag_retrieve`.
- **Judge-grader agreement unmeasured.** Planned 10-question manual sample after baseline.
- **PDF heading detection is heuristic.** Font-size based; works for arXiv's consistent LaTeX, breaks on inconsistent papers.

---

## Bugs worth logging

Real bugs caught during development, kept as lessons.

### Groq SDK's hidden retries invalidated failure tests

Initial `failures.py` layered on top of the SDK's own retry logic. Every 429 was retried twice by the SDK before the handler saw anything. A "429 → retry → success" test passed by accident because the SDK handled it. Fix: `Groq(max_retries=0, timeout=30.0)`. Lesson: when wrapping a client that retries, disable the client's retries or the test target is wrong.

### Truncation surfaced as malformed JSON in the repair loop

`finish_reason=="length"` means the model hit `max_tokens` mid-response. If the response was a tool call, the JSON is cut off and `json.loads` raises. Initial design routed this to the malformed-args repair handler, which re-prompted with the same `max_tokens` and got the same truncation. The repair loop couldn't converge. Fix: detect `finish_reason=="length"` before parsing and raise `TruncatedResponseError` as a distinct exception.

### `qdrant-client` 1.19.1 removed `.search()`

Deprecated in 1.11, removed later. Migrated to `.query_points()`. Different param name (`query_vector` → `query`), different response shape (list → `.points` on response object). Caught by smoke test, one-line fix.

### `arxiv` package 4.0 removed PDF downloads

Multiple attempts before diagnosis: `Result.download_pdf` (worked in 1.x), `Client.download_pdf` (brief in 2.x), neither in 4.0. Required introspecting `dir(arxiv)` to confirm PDF download was removed entirely. Fix: urllib against `pdf_url`.

### `summarize` output cap of 500 tokens was too tight for multi-aspect foci

First real question hit the cap on a 4-section paper summarized "for contribution and experimental setup." Hard-failing on `TRUNCATED` threw away a usable partial. Fix: raised cap to 800 and reclassified truncation as a soft warning with `output_truncated: True` on an `ok: True` result.

### First eval run revealed agent strategy weakness, not a bug

q001 burned 6/10 steps on `arxiv_search` to the live arXiv API when the target paper was already in the local index. Not a bug — a prompt tuning gap. Fix: added a bullet to the ACT system prompt directing the agent to try `rag_retrieve` first; raised `arxiv_search` timeout from 20s to 40s to reflect real arXiv latency. Caught by the eval harness on question 1 of 2.

### Groq free-tier TPM is 8K; a multi-step question uses 6-10K

A single `rag_retrieve + fetch_paper + citation_check + answer` run averages 6,500-7,500 tokens. The TPM window fires on roughly one in three questions. Not a bug — the entire project is built around this. Documented in the README.