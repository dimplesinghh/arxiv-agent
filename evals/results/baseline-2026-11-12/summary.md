# Eval summary — n=20

**Overall pass rate: 55.0%**

## By category

| Category | n | pass | pass rate |
|---|---|---|---|
| single | 5 | 5 | 100.0% |
| two | 8 | 4 | 50.0% |
| multi | 5 | 1 | 20.0% |
| adversarial_missing_id | 1 | 1 | 100.0% |
| adversarial_out_of_corpus | 1 | 0 | 0.0% |

## Tool calls

Total: 129 | Ok: 111 | Ok rate: 86.0%

| Tool | Calls | Ok | Ok rate |
|---|---|---|---|
| rag_retrieve | 26 | 25 | 96.2% |
| fetch_paper | 45 | 41 | 91.1% |
| citation_check | 20 | 20 | 100.0% |
| arxiv_search | 38 | 25 | 65.8% |

**Avg steps per question:** 7.15

## Failure counters (totals across all questions)

- budget_exhausted: 6
- empty_result: 0
- malformed_tool_args: 0
- rate_limited: 175
- reformulations_attempted: 0
- repairs_attempted: 0
- retries_attempted: 175
- tool_error_fatal: 0
- tool_error_retryable: 0
- tool_timeout: 0
- truncated: 0

## Token totals

- agent prompt tokens: 505,410
- agent completion tokens: 34,701
- judge tokens: 10,610
- **total**: 550,721

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
| q014 | two | answered | 7 | fail | The answer does not state that safety gains are judge-dependent. |
| q015 | two | answered | 5 | fail | Fails to credit KromHC or mention Kronecker‑product residual matrices |
| q016 | two | failed | 4 | fail | agent returned no answer (status=failed) |
| q017 | two | answered | 6 | PASS | All required points are addressed and the citation is present. |
| q018 | two | answered | 5 | PASS | All criteria satisfied: correct identification, explanation, and required citation present. |
| q026 | multi | answered | 9 | PASS | All criteria satisfied: correct description of SelectInfer, correct description of the adaptation work, and both arXiv I |
| q027 | multi | failed | 10 | fail | agent returned no answer (status=failed) |
| q028 | multi | failed | 10 | fail | agent returned no answer (status=failed) |
| q029 | multi | failed | 10 | fail | agent returned no answer (status=failed) |
| q030 | multi | insufficient | 10 | fail | judge call failed: TruncatedResponseError: response truncated at max_tokens=400 |
| q036 | adversarial_missing_id | answered | 4 | PASS | The answer correctly states the paper cannot be found and does not fabricate any content. |
| q039 | adversarial_out_of_corpus | insufficient | 10 | fail | The answer invents a claim about a paper addressing quantum error correction thresholds, violating the no‑invented‑claim |
