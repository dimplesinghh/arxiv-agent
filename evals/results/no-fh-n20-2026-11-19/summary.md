# Eval summary — n=20

**Overall pass rate: 0.0%**

## By category

| Category | n | pass | pass rate |
|---|---|---|---|
| single | 5 | 0 | 0.0% |
| two | 8 | 0 | 0.0% |
| multi | 5 | 0 | 0.0% |
| adversarial_missing_id | 1 | 0 | 0.0% |
| adversarial_out_of_corpus | 1 | 0 | 0.0% |

## Tool calls

Total: 5 | Ok: 3 | Ok rate: 60.0%

| Tool | Calls | Ok | Ok rate |
|---|---|---|---|
| rag_retrieve | 3 | 2 | 66.7% |
| arxiv_search | 2 | 1 | 50.0% |

**Avg steps per question:** 0.35

## Failure counters (totals across all questions)

- budget_exhausted: 0
- empty_result: 0
- malformed_tool_args: 0
- rate_limited: 0
- reformulations_attempted: 0
- repairs_attempted: 0
- retries_attempted: 0
- tool_error_fatal: 0
- tool_error_retryable: 0
- tool_timeout: 0
- truncated: 0

## Token totals

- agent prompt tokens: 13,805
- agent completion tokens: 1,233
- judge tokens: 0
- **total**: 15,038

## Per-question verdicts

| ID | Cat | Status | Steps | Pass | Judge reason |
|---|---|---|---|---|---|
| q001 | single | failed | 6 | fail | agent returned no answer (status=failed) |
| q002 | single | failed | 0 | fail | agent returned no answer (status=failed) |
| q003 | single | failed | 0 | fail | agent returned no answer (status=failed) |
| q004 | single | failed | 0 | fail | agent returned no answer (status=failed) |
| q005 | single | failed | 0 | fail | agent returned no answer (status=failed) |
| q011 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q012 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q013 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q014 | two | failed | 0 | fail | agent returned no answer (status=failed) |
| q015 | two | failed | 0 | fail | agent returned no answer (status=failed) |
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
