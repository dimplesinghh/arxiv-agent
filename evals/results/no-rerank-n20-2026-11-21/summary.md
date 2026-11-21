# Eval summary — n=20

**Overall pass rate: 25.0%**

## By category

| Category | n | pass | pass rate |
|---|---|---|---|
| single | 5 | 4 | 80.0% |
| two | 8 | 1 | 12.5% |
| multi | 5 | 0 | 0.0% |
| adversarial_missing_id | 1 | 0 | 0.0% |
| adversarial_out_of_corpus | 1 | 0 | 0.0% |

## Tool calls

Total: 64 | Ok: 53 | Ok rate: 82.8%

| Tool | Calls | Ok | Ok rate |
|---|---|---|---|
| rag_retrieve | 8 | 8 | 100.0% |
| fetch_paper | 21 | 20 | 95.2% |
| citation_check | 13 | 13 | 100.0% |
| arxiv_search | 22 | 12 | 54.5% |

**Avg steps per question:** 3.85

## Failure counters (totals across all questions)

- budget_exhausted: 1
- empty_result: 0
- malformed_tool_args: 0
- rate_limited: 86
- reformulations_attempted: 0
- repairs_attempted: 0
- retries_attempted: 86
- tool_error_fatal: 0
- tool_error_retryable: 0
- tool_timeout: 0
- truncated: 0

## Token totals

- agent prompt tokens: 236,678
- agent completion tokens: 18,599
- judge tokens: 6,193
- **total**: 261,470

## Per-question verdicts

| ID | Cat | Status | Steps | Pass | Judge reason |
|---|---|---|---|---|---|
| q001 | single | answered | 4 | fail | The answer lists the one-step and two-step methods but does not credit causal inference as the target domain. |
| q002 | single | answered | 10 | PASS | The answer correctly identifies selective neuron loading and selective computation as the core mechanisms and notes the  |
| q003 | single | answered | 6 | PASS | Answer correctly identifies anomaly detection in large-scale cloud monitoring and mentions context awareness. |
| q004 | single | answered | 7 | PASS | The answer correctly identifies codeslop as the target and cites agent trajectory minimization as the mechanism. |
| q005 | single | answered | 8 | PASS | Correctly identifies activation steering and mentions self-loops as the target problem |
| q011 | two | answered | 9 | fail | The answer does not identify prompt injection or external-content attacks as the evaluated threat. |
| q012 | two | failed | 10 | fail | agent returned no answer (status=failed) |
| q013 | two | answered | 5 | PASS | All required points are addressed and the citation is present (any version accepted). |
| q014 | two | answered | 6 | fail | The answer does not state that safety gains are judge-dependent nor that helpfulness costs are model-specific. |
| q015 | two | answered | 6 | fail | judge call failed: TruncatedResponseError: response truncated at max_tokens=400 |
| q016 | two | failed | 3 | fail | agent returned no answer (status=failed) |
| q017 | two | failed | 1 | fail | agent returned no answer (status=failed) |
| q018 | two | failed | 1 | fail | agent returned no answer (status=failed) |
| q026 | multi | failed | 1 | fail | agent returned no answer (status=failed) |
| q027 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q028 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q029 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q030 | multi | failed | 0 | fail | agent returned no answer (status=failed) |
| q036 | adversarial_missing_id | failed | 0 | fail | agent returned no answer (status=failed) |
| q039 | adversarial_out_of_corpus | failed | 0 | fail | agent returned no answer (status=failed) |
