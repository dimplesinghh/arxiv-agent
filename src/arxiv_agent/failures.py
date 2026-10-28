"""Failure handlers for LLM calls.

The one place that decides when an LLM call is retried. llm.py stays the raw
call surface (SDK retries disabled); everything that wants retries calls
call_llm_with_retry() instead of call_llm().

Retried:
  - 429 rate limits: wait for the server's retry-after, then try again.
    Groq's free tier allows 8K tokens/minute and the agent's ACT prompt grows
    by ~1K tokens per step, so this fires on most multi-step runs.
  - connection errors, timeouts, 5xx: exponential backoff with jitter.

Not retried (raised immediately): 400s, auth errors, truncation, malformed
tool args. Re-sending the same request cannot fix those.

A 429 whose reset is further away than MAX_WAIT_SECONDS (the daily token
limit, for example) is raised immediately rather than blocking for hours.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Callable, Optional

import groq

from arxiv_agent import llm


MAX_ATTEMPTS = 4

# The TPM window is one minute; a longer wait means a daily limit or worse.
MAX_WAIT_SECONDS = 65.0

# Small margin past the advertised reset so we don't land a few ms early.
_RESET_MARGIN_SECONDS = 0.5

_TRY_AGAIN_RE = re.compile(r"try again in (?:(\d+)m)?([0-9.]+)s")

_RETRYABLE = (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError)

# Called before each sleep with (error, attempt_number, wait_seconds).
RetryCallback = Callable[[Exception, int, float], None]


def retry_after_seconds(err: groq.RateLimitError) -> Optional[float]:
    """Seconds until the rate limit resets, from the retry-after header or,
    failing that, the 'Please try again in 13.35s' text in the error body.
    """
    response = getattr(err, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = _TRY_AGAIN_RE.search(str(err))
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        return minutes * 60 + float(m.group(2))
    return None


def _wait_for(err: Exception, attempt: int) -> float:
    if isinstance(err, groq.RateLimitError):
        server_wait = retry_after_seconds(err)
        if server_wait is not None:
            return server_wait + _RESET_MARGIN_SECONDS
    return min(30.0, 2.0 ** attempt) * random.uniform(0.5, 1.5)


def call_llm_with_retry(
    *,
    on_retry: Optional[RetryCallback] = None,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> llm.LLMResponse:
    """call_llm() with retries for transient failures. Same kwargs as call_llm.

    Raises the last error once attempts run out, or immediately for errors
    that are not transient. `sleep` is injectable so tests run instantly.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.call_llm(**kwargs)
        except _RETRYABLE as e:
            if attempt == max_attempts:
                raise
            wait = _wait_for(e, attempt)
            if wait > MAX_WAIT_SECONDS:
                raise
            if on_retry is not None:
                on_retry(e, attempt, wait)
            sleep(wait)
    raise AssertionError("unreachable")
