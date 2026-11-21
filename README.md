# arxiv-agent

![CI](https://github.com/dimplesinghh/arxiv-agent/actions/workflows/ci.yml/badge.svg)

Scratch-built research assistant agent over arXiv papers. A multi-step tool-calling loop with an explicit failure taxonomy, a 17-test fault-injection suite, and a 40-question quantitative eval harness. No agent framework — the loop, state machine, and failure handlers are the point.

## Why this exists

Companion to [`dimplesinghh/arxiv-rag`](https://github.com/dimplesinghh/arxiv-rag). The RAG project answered "can retrieval be evaluated rigorously?" This one answers "can an agent built on top of it fail rigorously?" — i.e. can every failure mode (tool timeout, rate limit, malformed tool args, truncation, empty results, context overflow, step-budget exhaustion) be detected, handled, counted, and measured?

## Design decisions

- **From scratch, no LangGraph or LangChain.** The control flow, tool interface, and error handling are all readable in one repo. Tradeoff: more code, full ownership.
- **Explicit state machine** (`PLAN → ACT → OBSERVE → ANSWER`), not free-form ReAct. Each transition is a pure function of `AgentState` and unit-testable.
- **Errors are values at the tool boundary, typed exceptions at the LLM boundary.** Every tool returns `ToolResult(ok, data, error)` and never raises. The Groq client wrapper raises `TruncatedResponseError` and `MalformedToolArgsError` at specific sites the loop handles explicitly.
- **One source of retry logic — `failures.py`.** The Groq SDK's own retries are disabled (`max_retries=0`) so handler behavior reflects reality and `latency_ms` is honest. Rate-limit waits respect the server's `retry-after` header; transient errors use exponential backoff with jitter; daily-limit waits (>65s) raise immediately instead of blocking for hours.
- **One-shot query reformulation.** When `rag_retrieve` or `arxiv_search` returns `EMPTY_RESULT`, the loop asks the LLM for a different query and re-runs the tool once. Never twice — an agent that can't answer in two retrieval variants should fall through to "insufficient evidence."
- **Budget-exhausted partial-answer disclosure.** When step budget hits zero and the trace has at least one successful tool result, the agent emits an honest partial answer with an `UNCERTAIN:` prefix and `INSUFFICIENT` status, not silent `FAILED`. Lets the eval judge distinguish "ran out of budget with some evidence" from "did not understand the question."
- **LLM: Groq-hosted `openai/gpt-oss-120b`.** Chosen deliberately: an open-weights model with a rate-limited free tier. The eval harness exercises real 429s, real backoff, real degradation. Frontier models would hide these failure modes.
- **Qdrant index reused from `arxiv-rag`, not duplicated.** This repo reads the `arxiv_chunks` collection (50 papers, ~2,350 chunks) from Project 1's instance rather than re-embedding. Tradeoff: Project 1's Qdrant must be running before this repo is useful.

Full rationale in `docs/decisions.md`.

## Architecture

```
question
   │
   ▼
┌──────┐    ┌─────┐    ┌─────────┐    ┌────────┐
│ PLAN │───▶│ ACT │───▶│ OBSERVE │───▶│ ANSWER │───▶ final answer + citations
└──────┘    └──┬──┘    └────┬────┘    └────────┘
               │            │
               └────────────┘
         loop until model emits answer,
         budget exhausts, or handler
         gives up (status = FAILED |
         ANSWERED | INSUFFICIENT)
```

- **Five tools** wired through one registry (`src/arxiv_agent/tools/__init__.py`).
- **Every state transition** is a pure function of `AgentState`.
- **`run_agent(question)`** is the only public entry point; CLI at `python -m arxiv_agent "..."`.

## Failure handling

| Kind | Trigger | Handler |
|---|---|---|
| `RATE_LIMITED` | Groq 429 | Wait `retry-after` header (+ 0.5s margin), retry up to 4 attempts. Daily-limit-size waits (>65s) raise immediately. |
| `TOOL_ERROR_RETRYABLE` | Groq connection error or 5xx | Exponential backoff with jitter: `2^attempt × U(0.5, 1.5)`, up to 4 attempts. |
| `TOOL_TIMEOUT` | Tool exceeded its own budget | Captured in the trace as a failed tool call; agent decides whether to retry via a different tool. |
| `MALFORMED_TOOL_ARGS` | LLM emitted invalid JSON for a tool call | Captured as a typed exception; loop terminates cleanly rather than crashing. |
| `TRUNCATED` | `finish_reason == "length"` | `summarize` accepts partial output with a flag; elsewhere terminates cleanly. |
| `EMPTY_RESULT` | Retrieval tool returned nothing relevant | One-shot query reformulation via LLM; runs the tool once more with the new query. Never twice per run. |
| `BUDGET_EXHAUSTED` | Step budget reached | Emit `UNCERTAIN:`-prefixed partial answer if any evidence exists; status becomes `INSUFFICIENT`. Otherwise `FAILED`. |

Every handler increments a counter in `AgentState.counters`. Counters are aggregated per eval run.

## Tools

| Tool | Purpose |
|---|---|
| `arxiv_search` | Search the public arXiv API (slow; falls back when the local index is empty). |
| `rag_retrieve` | Dense retrieval + cross-encoder rerank over Project 1's `arxiv_chunks` collection. Preferred first step. |
| `fetch_paper` | Download, parse, and return one section of an arXiv paper by ID. PDF-cached to disk. |
| `summarize` | Focused LLM summarization with input-size cap and partial-output flag. |
| `citation_check` | Verify a claim against source chunks. JSON verdict: `supported` / `partially_supported` / `unsupported`. |

## Eval harness

40-question stratified set in `evals/questions.jsonl`:

| Count | Category | Shape |
|---|---|---|
| 10 | `single` | One tool call, one paper, one direct fact |
| 15 | `two` | `rag_retrieve` + `citation_check`, one paper, inline citations required |
| 10 | `multi` | 2+ tool calls across 2 papers, compare/contrast |
| 3 | `adversarial_missing_id` | Nonexistent arxiv_id — agent must refuse honestly |
| 2 | `adversarial_out_of_corpus` | Topic absent from the 50-paper index — agent must say insufficient evidence |

Each question has `expected_evidence_arxiv_ids` and an LLM-judge `success_criteria`. The harness in `evals/run_eval.py`:

- Runs `run_agent()` on each question
- Judges the answer via an LLM call against `success_criteria`
- Writes a JSON trace per question + `summary.json` + `summary.md`
- Is **resumable** — pass the same `--output-dir` across sessions; existing results are skipped

Run:

```bash
PYTHONPATH=src python -m evals.run_eval --output-dir evals/results/baseline-$(date -u +%Y-%m-%d) --limit 10
```

Full 40-question run is paced across multiple days due to the Groq 200K tokens/day free-tier cap.


## Results

Three configs, same 20-question stratified set (5 single, 8 two, 5 multi, 1 adversarial-missing-id, 1 adversarial-out-of-corpus), run against the Groq free-tier `openai/gpt-oss-120b` under real rate-limit conditions.

| Config | Overall | Single | Two | Multi | Adv-missing | Adv-out | Rate-limit events | Tokens |
|---|---|---|---|---|---|---|---|---|
| Baseline (all features) | **55%** | 100% | 50% | 20% | 100% | 0% | 175 handled | 551K |
| No failure handling | **0%** | 0% | 0% | 0% | 0% | 0% | 0 handled, 1 cascade | 15K |
| No rerank | **25%** | 80% | 12.5% | 0%\* | 0%\* | 0%\* | 86 handled | 261K |

`*` multi and adversarial categories in the no-rerank run hit the TPD cascade after q015; those cells reflect too few real runs to assess the reranker's isolated effect on those categories.

### Ablation 1 — No failure handling

A single question (q001) completed 6 steps before hitting the 8K TPM window. With retries disabled, the agent crashed. The next question's PLAN call immediately hit 429 because the per-minute window had not cleared. Every subsequent question crashed on its first LLM call.

The **55 → 0 delta is not an artifact of harsh test conditions** — it is the honest answer to "what happens if you build an agent for a rate-limited free tier without failure handling?" In production, the handler is not optional; it is structural. Baseline handled **175 rate-limit events** across 20 questions and still finished at 55%; the no-fh config handled zero and finished at zero.

### Ablation 2 — No reranking

With the cross-encoder reranker off, the retrieval tool falls back to dense-only. Fair-comparison categories (single, two):

- **single: 100% → 80%** (−20 points). Reranking promotes the right chunk on one of five single-tool questions that dense retrieval alone surfaces lower.
- **two: 50% → 12.5%** (−37.5 points). Two-tool questions call `citation_check` on the top retrieved chunks; when dense retrieval returns the wrong chunk as #1, the whole verification chain fails.

The reranker is not catastrophic to remove (unlike failure handling), but it costs real pass rate — especially on questions that need high retrieval precision for downstream verification.

### Headline interpretation

| Component | Measured contribution |
|---|---|
| Failure handling | **+55 points** (makes the agent possible at all under rate limits) |
| Reranking | **+20 to +37 points** on retrieval-dependent categories |

Both decisions were made before the ablation; the numbers above are post-hoc measurement of their effect.

### Known gaps in the N=20 study

- Ablation-2 was partially TPD-contaminated (q016 onward): multi and adversarial categories need a re-run to assess reranker's effect there.
- All results are at N=20. The full N=40 question set exists in `evals/questions.jsonl`; running it would take ~2x the token budget and was descoped due to the Groq free-tier 200K TPD cap.
- Judge is the same model as the agent (same-model bias documented in `docs/decisions.md`). Judge-grader agreement spot-check is a planned next step.

## Prerequisites

- Python 3.12
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient — see rate limits below)
- A running Qdrant instance with the `arxiv_chunks` collection from [`dimplesinghh/arxiv-rag`](https://github.com/dimplesinghh/arxiv-rag). Start it with `docker compose up -d` from your local `arxiv-rag/` checkout before working here.

### Groq free-tier rate limits — `openai/gpt-oss-120b`

| Limit | Value | Implication |
|---|---|---|
| Requests / minute | 30 | Not the bottleneck |
| Requests / day | 1,000 | Not the bottleneck |
| Tokens / minute | 8,000 | `RATE_LIMITED` handler fires on most multi-step runs; backoff waits are visible in traces |
| Tokens / day | 200,000 | ~10 eval questions/day; full 40-question runs spread across multiple days |

These limits directly shape the failure taxonomy — the `RATE_LIMITED` handler exists because it will fire, not because it might.

## Quickstart

```bash
# 1. Start the shared Qdrant from Project 1
cd ~/path/to/arxiv-rag && docker compose up -d
cd -

# 2. Set up this repo
git clone git@github.com:dimplesinghh/arxiv-agent.git
cd arxiv-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GROQ_API_KEY

# 3. Run the agent on a question
PYTHONPATH=src python -m arxiv_agent "What architecture does the RAG paper use for the retriever?"

# 4. Run the test suite (offline, no API key needed for tests)
PYTHONPATH=src pytest tests/ -v

# 5. Run a batch of eval questions
PYTHONPATH=src python -m evals.run_eval --output-dir evals/results/baseline-$(date -u +%Y-%m-%d) --limit 10
```

## Repo tour

```
src/arxiv_agent/
  schema.py              # pydantic models: ToolResult, ToolError, StepRecord, AgentState, FailureCounters
  llm.py                 # Groq client wrapper, tool-use parsing, truncation/malformed-args detection
  failures.py            # single source of retry logic — retry-after parsing, backoff, max-attempts cap
  loop.py                # agent state machine, query reformulation, budget-exhausted partial answer
  prompts.py             # PLANNER, ACT, ANSWER, REFORMULATE_QUERY, BUDGET_EXHAUSTED_ANSWER
  __main__.py            # CLI: `python -m arxiv_agent "question"`
  tools/
    __init__.py          # tool registry (name → function, schema)
    arxiv_search.py      # search the public arXiv API
    rag_retrieve.py      # dense retrieval + cross-encoder rerank
    fetch_paper.py       # download + parse full arXiv paper by ID (cached)
    summarize.py         # focused LLM summarization with input-size cap
    citation_check.py    # verify a claim against source chunks; JSON verdict
tests/
  test_failures.py       # 11 fault-injection tests for call_llm_with_retry
  test_loop.py           # 6 reformulation + 3 budget-exhausted tests
evals/
  questions.jsonl        # 40 stratified eval questions
  run_eval.py            # eval harness: agent + LLM judge + per-metric aggregation
  verify_questions.py    # one-shot tool to sanity-check criteria against actual chunks
  results/               # per-question traces + summary.json/summary.md (gitignored except summaries)
docs/
  decisions.md           # design decisions, known limitations, bugs log
```
