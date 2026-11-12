"""Eval harness for arxiv-agent.

Runs the agent on every question in evals/questions.jsonl, grades each
answer with an LLM judge against the per-question success_criteria, and
writes:
  - one JSON file per question with the full trace + judge verdict
  - summary.json with aggregated metrics
  - summary.md human-readable table

Resumable: a per-question JSON already on disk is reused unless --force.
Agent and judge calls are expensive; don't throw results away on a crash
or a daily-limit hit — just rerun and it picks up where it left off.

Usage:
    PYTHONPATH=src python -m evals.run_eval                # full 40
    PYTHONPATH=src python -m evals.run_eval --limit 2      # sanity check
    PYTHONPATH=src python -m evals.run_eval --limit 10     # batch
    PYTHONPATH=src python -m evals.run_eval --force        # re-run all
    PYTHONPATH=src python -m evals.run_eval --config baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from arxiv_agent.failures import call_llm_with_retry
from arxiv_agent.loop import run_agent
from arxiv_agent.schema import StepRecord


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = REPO_ROOT / "evals" / "questions.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evals" / "results"


JUDGE_SYSTEM_PROMPT = """You are a strict grader of research-assistant answers. You will be given:
- a QUESTION the agent was asked
- SUCCESS CRITERIA that define a passing answer
- the AGENT ANSWER (with its agent status)

Decide pass/fail based ONLY on the success criteria. Do not reward answers that are well-written but miss the criteria. Do not penalize answers that are terse but satisfy the criteria.

For adversarial questions (missing or malformed arxiv IDs, out-of-corpus topics), an honest refusal or explicit 'insufficient evidence' answer PASSES; a fabricated answer FAILS.

Return a single JSON object with this exact shape:
{
  "pass": true | false,
  "reason": "one short sentence explaining the verdict",
  "criteria_met": ["..."],
  "criteria_missed": ["..."]
}

Return ONLY the JSON object. No preamble."""


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def judge_answer(question: str, success_criteria: str, answer: str, status: str) -> dict[str, Any]:
    if not answer:
        return {
            "pass": False,
            "reason": f"agent returned no answer (status={status})",
            "criteria_met": [],
            "criteria_missed": [success_criteria],
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        }

    user_content = (
        f"QUESTION: {question}\n\n"
        f"SUCCESS CRITERIA: {success_criteria}\n\n"
        f"AGENT ANSWER (status={status}):\n{answer}"
    )
    try:
        resp = call_llm_with_retry(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        return {
            "pass": False,
            "reason": f"judge call failed: {type(e).__name__}: {e}",
            "criteria_met": [],
            "criteria_missed": [success_criteria],
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        }

    raw = (resp.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "pass": False,
            "reason": f"judge returned non-JSON: {e}; raw: {raw[:200]!r}",
            "criteria_met": [],
            "criteria_missed": [success_criteria],
            "judge_prompt_tokens": resp.prompt_tokens,
            "judge_completion_tokens": resp.completion_tokens,
        }

    parsed.setdefault("pass", False)
    parsed.setdefault("reason", "")
    parsed.setdefault("criteria_met", [])
    parsed.setdefault("criteria_missed", [])
    parsed["pass"] = bool(parsed["pass"])
    parsed["judge_prompt_tokens"] = resp.prompt_tokens
    parsed["judge_completion_tokens"] = resp.completion_tokens
    return parsed


def _tool_calls_summary(history: list[StepRecord]) -> dict[str, Any]:
    tool_calls = 0
    tool_ok = 0
    per_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0})
    for s in history:
        if s.tool_name:
            tool_calls += 1
            per_tool[s.tool_name]["calls"] += 1
            if s.result and s.result.ok:
                tool_ok += 1
                per_tool[s.tool_name]["ok"] += 1
    return {
        "total_calls": tool_calls,
        "ok_calls": tool_ok,
        "per_tool": dict(per_tool),
    }


def _history_snapshot(history: list[StepRecord]) -> list[dict[str, Any]]:
    snap = []
    for s in history:
        entry = {
            "step_n": s.step_n,
            "state": s.state.value,
            "tool_name": s.tool_name,
            "tool_args": s.tool_args,
            "latency_ms": s.latency_ms,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "notes": s.notes,
        }
        if s.result:
            entry["result_ok"] = s.result.ok
            if s.result.error:
                entry["result_error"] = {
                    "kind": s.result.error.kind.value,
                    "message": s.result.error.message,
                }
            else:
                data = s.result.data or {}
                entry["result_summary"] = {
                    k: (str(v)[:200] if isinstance(v, (str, int, float)) else f"<{type(v).__name__}>")
                    for k, v in list(data.items())[:5]
                }
        snap.append(entry)
    return snap


def run_one(q: dict[str, Any], result_path: Path, step_budget: int, force: bool) -> dict[str, Any]:
    if result_path.exists() and not force:
        with result_path.open() as f:
            existing = json.load(f)
        existing["_resumed"] = True
        return existing

    t0 = time.perf_counter()
    state = run_agent(q["question"], step_budget=step_budget)
    agent_seconds = time.perf_counter() - t0

    verdict = judge_answer(
        question=q["question"],
        success_criteria=q["success_criteria"],
        answer=state.final_answer or "",
        status=state.status.value,
    )

    tool_summary = _tool_calls_summary(state.history)
    row = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "success_criteria": q["success_criteria"],
        "expected_evidence_arxiv_ids": q.get("expected_evidence_arxiv_ids", []),
        "status": state.status.value,
        "final_answer": state.final_answer,
        "citations": state.citations,
        "steps_used": state.step_n,
        "step_budget": state.step_budget,
        "agent_seconds": round(agent_seconds, 2),
        "tool_calls": tool_summary,
        "counters": state.counters.model_dump(),
        "trace": _history_snapshot(state.history),
        "judge": verdict,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(row, f, indent=2)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0})
    counters_total: Counter = Counter()
    tool_calls_total = 0
    tool_ok_total = 0
    per_tool_total: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "ok": 0})
    steps_total = 0
    judge_tokens = 0
    agent_prompt_tokens = 0
    agent_completion_tokens = 0

    for r in rows:
        cat = r["category"]
        by_cat[cat]["n"] += 1
        if r["judge"]["pass"]:
            by_cat[cat]["pass"] += 1
        for k, v in r["counters"].items():
            if isinstance(v, (int, float)):
                counters_total[k] += v
        tool_calls_total += r["tool_calls"]["total_calls"]
        tool_ok_total += r["tool_calls"]["ok_calls"]
        for tn, d in r["tool_calls"]["per_tool"].items():
            per_tool_total[tn]["calls"] += d["calls"]
            per_tool_total[tn]["ok"] += d["ok"]
        steps_total += r["steps_used"]
        judge_tokens += r["judge"].get("judge_prompt_tokens", 0) + r["judge"].get("judge_completion_tokens", 0)
        for step in r["trace"]:
            agent_prompt_tokens += step.get("prompt_tokens", 0) or 0
            agent_completion_tokens += step.get("completion_tokens", 0) or 0

    n = max(1, len(rows))
    return {
        "n_questions": len(rows),
        "overall_pass_rate": round(sum(d["pass"] for d in by_cat.values()) / n, 3),
        "by_category": {
            cat: {
                "n": d["n"],
                "pass": d["pass"],
                "pass_rate": round(d["pass"] / max(1, d["n"]), 3),
            }
            for cat, d in by_cat.items()
        },
        "tool_calls": {
            "total": tool_calls_total,
            "ok": tool_ok_total,
            "ok_rate": round(tool_ok_total / max(1, tool_calls_total), 3),
            "per_tool": {
                tn: {"calls": d["calls"], "ok": d["ok"], "ok_rate": round(d["ok"] / max(1, d["calls"]), 3)}
                for tn, d in per_tool_total.items()
            },
        },
        "avg_steps_per_question": round(steps_total / n, 2),
        "token_totals": {
            "agent_prompt": agent_prompt_tokens,
            "agent_completion": agent_completion_tokens,
            "judge": judge_tokens,
            "total": agent_prompt_tokens + agent_completion_tokens + judge_tokens,
        },
        "failure_counters_total": dict(counters_total),
    }


def write_markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [f"# Eval summary — n={summary['n_questions']}", ""]
    lines.append(f"**Overall pass rate: {summary['overall_pass_rate']*100:.1f}%**")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | n | pass | pass rate |")
    lines.append("|---|---|---|---|")
    for cat, d in summary["by_category"].items():
        lines.append(f"| {cat} | {d['n']} | {d['pass']} | {d['pass_rate']*100:.1f}% |")
    lines.append("")
    lines.append("## Tool calls")
    lines.append("")
    tc = summary["tool_calls"]
    lines.append(f"Total: {tc['total']} | Ok: {tc['ok']} | Ok rate: {tc['ok_rate']*100:.1f}%")
    lines.append("")
    lines.append("| Tool | Calls | Ok | Ok rate |")
    lines.append("|---|---|---|---|")
    for tn, d in tc["per_tool"].items():
        lines.append(f"| {tn} | {d['calls']} | {d['ok']} | {d['ok_rate']*100:.1f}% |")
    lines.append("")
    lines.append(f"**Avg steps per question:** {summary['avg_steps_per_question']}")
    lines.append("")
    lines.append("## Failure counters (totals across all questions)")
    lines.append("")
    for k, v in sorted(summary["failure_counters_total"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Token totals")
    lines.append("")
    tt = summary["token_totals"]
    lines.append(f"- agent prompt tokens: {tt['agent_prompt']:,}")
    lines.append(f"- agent completion tokens: {tt['agent_completion']:,}")
    lines.append(f"- judge tokens: {tt['judge']:,}")
    lines.append(f"- **total**: {tt['total']:,}")
    lines.append("")
    lines.append("## Per-question verdicts")
    lines.append("")
    lines.append("| ID | Cat | Status | Steps | Pass | Judge reason |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        p = "PASS" if r["judge"]["pass"] else "fail"
        reason = r["judge"]["reason"].replace("|", "\\|")[:120]
        lines.append(f"| {r['id']} | {r['category']} | {r['status']} | {r['steps_used']} | {p} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the arxiv-agent eval harness.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: evals/results/<utc-timestamp>-<config>")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step-budget", type=int, default=10)
    parser.add_argument("--config", type=str, default="baseline")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = DEFAULT_RESULTS_DIR / f"{stamp}-{args.config}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"questions: {len(questions)}")
    print(f"output:    {args.output_dir}")
    print(f"config:    {args.config}")
    print(f"budget:    {args.step_budget}")
    print()

    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        result_path = args.output_dir / f"{q['id']}.json"
        t_start = time.perf_counter()
        print(f"[{i}/{len(questions)}] {q['id']} ({q['category']}) ... ", flush=True, end="")
        try:
            row = run_one(q, result_path, args.step_budget, args.force)
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            continue
        elapsed = time.perf_counter() - t_start
        p = "PASS" if row["judge"]["pass"] else "fail"
        resumed = " (resumed)" if row.get("_resumed") else ""
        print(f"{p} | status={row['status']} | steps={row['steps_used']} | {elapsed:.1f}s{resumed}")
        rows.append(row)

    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, rows, args.output_dir / "summary.md")

    print()
    print(f"overall pass rate: {summary['overall_pass_rate']*100:.1f}%")
    print(f"total tokens:      {summary['token_totals']['total']:,}")
    print(f"summary written:   {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())