"""One-shot verification of [VERIFY]-flagged questions.

For each (question, expected_arxiv_id) pair below, runs rag_retrieve and prints
the top-3 chunks. You read the output and tighten success_criteria for the
matching q-id in evals/questions.jsonl before committing.

Not part of the eval harness — this is a one-time tuning tool. Safe to delete
after questions.jsonl is finalized.
"""

from dotenv import load_dotenv
load_dotenv()

from arxiv_agent.tools.rag_retrieve import rag_retrieve

VERIFY_QUESTIONS = [
    ("q005", "2607.18100v1", "In 'Breaking LLMs Out of Self-Loops', what fine-grained control mechanism do the authors use?"),
    ("q007", "2607.18147v1", "What are the main application use cases covered in the Smart Grids LLM/agentic AI tutorial?"),
    ("q009", "2607.18069v1", "What hardware-level approach does the AI performance throttling paper propose?"),
    ("q010", "2607.18082v1", "What does the Rubric-based RL paper propose to improve?"),
    ("q012", "2607.18042v1", "What does 'Anticipate Before Acting' propose for vision-language navigation?"),
    ("q013", "2607.18046v1", "How does SEE support long-horizon GUI agent tasks?"),
    ("q015", "2607.18130v1", "What is the Manifold-Constrained Hyper-Connections paper about?"),
    ("q016", "2607.18142v1", "What makes O-VAD 'object-centric' for industrial video anomaly detection?"),
    ("q018", "2607.18228v1", "What are 'learned soft prefixes' used for in the syllogistic stability paper?"),
    ("q019", "2607.18114v1", "In 'How Does Alignment Tuning Shape Representations of Sycophancy', what representations do the authors probe?"),
    ("q020", "2607.18237v1", "What perceptual metric does 'The Many Senses of Visual Similarity' propose?"),
    ("q021", "2607.18119v1", "What fairness concept does the subgroup-fair clustering paper enforce?"),
    ("q022", "2607.18200v1", "What robot setting does 'Learning Adaptive Safety Margins' target?"),
    ("q023", "2607.18195v1", "What perturbations does 'Certified Training for Convolutional Perturbations' defend against?"),
    ("q024", "2607.18218v1", "What domain does GigaPath-Flash target, and what makes it 'Flash'?"),
    ("q025", "2607.18199v1", "What does PPL-Factory select and under what budget?"),
    ("q030", "2607.18082v1 + 2607.18110v1", "Compare the LLM learning paradigms in 'Enhancing Rubric-based RL' and 'LLM-as-a-Coach'."),
    ("q033", "2607.18064v1 + 2607.18235v1", "Both 'Autoresearch with Coding Agents' and 'Automated Discovery Has No Universally Superior Harness' address automated research. Where do their claims disagree?"),
    ("q034", "2607.18069v1 + 2607.18101v1", "Compare the AI-hardware approaches in the throttling paper and the edge AI accelerator paper."),
]


def verify_one(qid: str, expected: str, question: str) -> None:
    print("=" * 80)
    print(f"{qid}  |  expected: {expected}")
    print(f"QUESTION: {question}")
    print("-" * 80)
    r = rag_retrieve(question, top_k=3)
    if not r.ok:
        print(f"[FAIL] rag_retrieve error: {r.error.kind.value} — {r.error.message}")
        return
    for c in r.data["results"]:
        aid = c.get("arxiv_id")
        match = "  <-- MATCH" if aid in expected else ""
        rerank = c.get("rerank_score", 0.0)
        print(f"  {aid} chunk {c.get('chunk_idx')} (rerank={rerank:.2f}){match}")
        print(f"    {c['text'][:300].replace(chr(10), ' ')}...")
        print()


if __name__ == "__main__":
    for qid, expected, question in VERIFY_QUESTIONS:
        verify_one(qid, expected, question)
