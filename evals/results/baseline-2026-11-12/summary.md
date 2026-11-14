# Eval summary — n=20

**Overall pass rate: 35.0%**

## By category

| Category | n | pass | pass rate |
|---|---|---|---|
| single | 5 | 5 | 100.0% |
| two | 8 | 2 | 25.0% |
| multi | 5 | 0 | 0.0% |
| adversarial_missing_id | 1 | 0 | 0.0% |
| adversarial_out_of_corpus | 1 | 0 | 0.0% |

## Tool calls

Total: 49 | Ok: 47 | Ok rate: 95.9%

| Tool | Calls | Ok | Ok rate |
|---|---|---|---|
| rag_retrieve | 6 | 6 | 100.0% |
| fetch_paper | 20 | 18 | 90.0% |
| citation_check | 10 | 10 | 100.0% |
| arxiv_search | 13 | 13 | 100.0% |

**Avg steps per question:** 2.95

## Failure counters (totals across all questions)

- budget_exhausted: 1
- empty_result: 0
- malformed_tool_args: 0
- rate_limited: 76
- reformulations_attempted: 0
- repairs_attempted: 0
- retries_attempted: 76
- tool_error_fatal: 0
- tool_error_retryable: 0
- tool_timeout: 0
- truncated: 0

## Token totals

- agent prompt tokens: 190,923
- agent completion tokens: 14,554
- judge tokens: 4,770
- **total**: 210,247

## Per-question verdicts

| ID | Cat | Status | Steps | Pass | Judge reason |
|---|---|---|---|---|---|
| q001 | single | answered | 4 | PASS | The answer lists both a one-step and a two-step method and notes that they are for causal inference policy learning. |
| q002 | single | answered | 7 | PASS | The answer correctly identifies selective neuron loading and selective computation as the core mechanism and notes on-de |
| q003 | single | answered | 6 | PASS | The answer correctly identifies anomaly detection in large-scale cloud monitoring and mentions operational context aware |
| q004 | single | answered | 8 | PASS | Correctly identifies codeslop and mentions trajectory minimization |
| q005 | single | answered | 5 | PASS | Correctly identifies activation steering and links it to reasoning control/self‑loops. |
| q011 | two | answered | 7 | PASS | All criteria are satisfied: threat identified, multi-turn explained, citation included. |
| q012 | two | answered | 6 | PASS | All required elements are present and the citation matches the required paper ID. |
| q013 | two | failed | 10 | fail | agent returned no answer (status=failed) |
| q014 | two | failed | 4 | fail | agent returned no answer (status=failed) |
| q015 | two | failed | 1 | fail | agent returned no answer (status=failed) |
| q016 | two | failed | 1 | fail | agent returned no answer (status=failed) |
| q017 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q018 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q026 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q027 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q028 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q029 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q030 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q036 | adversarial_missing_id | failed | 0 | fail | agent returned no answer (status=failed) |
| q039 | adversarial_out_of_corpus | failed | 0 | fail | agent returned no answer (status=failed) |
