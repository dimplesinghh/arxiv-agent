"""CLI entrypoint. Run with: python -m arxiv_agent 'question'
"""

from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

from arxiv_agent.loop import run_agent


def main() -> int:
    load_dotenv()
    if len(sys.argv) < 2:
        print("usage: python -m arxiv_agent 'your question here'", file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    state = run_agent(question, step_budget=10)
    print("=" * 60)
    print(f"QUESTION: {state.question}")
    print(f"STATUS:   {state.status.value}")
    print(f"STEPS:    {state.step_n}")
    print(f"CITATIONS: {state.citations}")
    print("=" * 60)
    print(state.final_answer or "(no answer)")
    print("=" * 60)
    print("TRACE:")
    for step in state.history:
        tn = f" tool={step.tool_name}" if step.tool_name else ""
        notes = f" notes={step.notes}" if step.notes else ""
        print(f"  step {step.step_n:>2} [{step.state.value:<8}]{tn} ({step.latency_ms}ms){notes}")
    return 0 if state.status.value in ("answered", "insufficient") else 1


if __name__ == "__main__":
    sys.exit(main())