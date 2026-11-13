# Eval summary — n=10

**Overall pass rate: 70.0%**

## By category

| Category | n | pass | pass rate |
|---|---|---|---|
| single | 10 | 7 | 70.0% |

## Tool calls

Total: 49 | Ok: 45 | Ok rate: 91.8%

| Tool | Calls | Ok | Ok rate |
|---|---|---|---|
| rag_retrieve | 6 | 5 | 83.3% |
| arxiv_search | 14 | 12 | 85.7% |
| fetch_paper | 19 | 18 | 94.7% |
| citation_check | 10 | 10 | 100.0% |

**Avg steps per question:** 5.9

## Failure counters (totals across all questions)

- budget_exhausted: 0
- empty_result: 0
- malformed_tool_args: 0
- rate_limited: 74
- reformulations_attempted: 1
- repairs_attempted: 0
- retries_attempted: 74
- tool_error_fatal: 0
- tool_error_retryable: 0
- tool_timeout: 0
- truncated: 0

## Token totals

- agent prompt tokens: 165,638
- agent completion tokens: 14,022
- judge tokens: 5,529
- **total**: 185,189

## Per-question verdicts

| ID | Cat | Status | Steps | Pass | Judge reason |
|---|---|---|---|---|---|
| q001 | single | answered | 8 | fail | The answer lists the one-step and two-step methods but does not mention causal inference as the target domain. |
| q002 | single | answered | 7 | PASS | The answer correctly identifies selective neuron loading and selective computation as the core mechanism and notes on-de |
| q003 | single | answered | 6 | PASS | The answer correctly identifies anomaly detection in large-scale cloud monitoring and mentions operational context aware |
| q004 | single | answered | 8 | PASS | Correctly identifies codeslop and mentions trajectory minimization |
| q005 | single | answered | 5 | PASS | Correctly identifies activation steering and links it to reasoning control/self‑loops. |
| q006 | single | answered | 4 | PASS | The answer correctly identifies the one‑answer rule as the degrading mechanism and notes that better individual informat |
| q007 | single | answered | 8 | PASS | The answer correctly identifies the tutorial as implementation‑grounded for a power‑systems audience and lists prompting |
| q008 | single | answered | 5 | PASS | The answer states that no single harness is universally superior and references a comparison across harnesses. |
| q009 | single | failed | 5 | fail | agent returned no answer (status=failed) |
| q010 | single | failed | 3 | fail | agent returned no answer (status=failed) |
