# arxiv-agent

Scratch-built research assistant agent over arXiv papers. A multi-step tool-calling loop with an explicit failure taxonomy and a quantitative eval harness. No agent framework — the loop, state machine, and failure handlers are the point.

## Why this exists

Companion to [`dimplesinghh/arxiv-rag`](https://github.com/dimplesinghh/arxiv-rag). The RAG project answered "can retrieval be evaluated rigorously?" This one answers "can an agent built on top of it fail rigorously?" — i.e. can every failure mode (tool timeout, rate limit, malformed tool args, empty results, context overflow, step-budget exhaustion) be detected, handled, counted, and measured?

## Design decisions

- **From scratch, no LangGraph or LangChain.** The control flow, tool interface, and error handling are all readable in one repo. Tradeoff: more code, full ownership.
- **Explicit state machine** (`PLAN → ACT → OBSERVE → REFLECT → ANSWER`), not free-form ReAct. Each transition is unit-testable.
- **Errors are values, not exceptions.** Every tool returns `ToolResult(ok, data, error)`; nothing raises into the loop.
- **LLM: Groq-hosted `openai/gpt-oss-120b`.** Chosen deliberately for an open-weights model with a rate-limited free tier: the eval harness exercises real 429s, real backoff, real degradation. Frontier models would hide these failure modes. Full rationale in `docs/decisions.md`.
- A running Qdrant instance with the `arxiv_chunks` collection from [`dimplesinghh/arxiv-rag`](https://github.com/dimplesinghh/arxiv-rag). This repo intentionally does not spin up its own Qdrant — it reuses Project 1's index rather than duplicating 300 papers of embeddings. Before working here, run `docker compose up -d` from your local `arxiv-rag/` checkout.

## Prerequisites

- Python 3.12
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient — see rate limits below)
- A running Qdrant instance with the `arxiv-rag` index available at `QDRANT_URL`. See [`dimplesinghh/arxiv-rag`](https://github.com/dimplesinghh/arxiv-rag) for how to build it.

### Groq free-tier rate limits (as of Sep 2026)

Using `openai/gpt-oss-120b`:

| Limit | Value | Implication |
|---|---|---|
| Requests / minute | 30 | Not the bottleneck |
| Requests / day | 1,000 | Not the bottleneck |
| Tokens / minute | 8,000 | Agent paces itself; `RATE_LIMITED` handler fires during eval runs |
| Tokens / day | 200,000 | ~13 eval questions/day; full 40-question runs are spread across days |

These limits directly shape the failure taxonomy — the `RATE_LIMITED` handler exists because it will fire, not because it might.

## Quickstart

```bash
git clone git@github.com:dimplesinghh/arxiv-agent.git
cd arxiv-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GROQ_API_KEY and QDRANT_COLLECTION
```

## Repo tour
