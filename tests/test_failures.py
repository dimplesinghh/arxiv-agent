"""Fault-injection tests for failures.call_llm_with_retry.

All offline — no network, no API key, no real Groq SDK calls. Tests use:
  - monkeypatch on llm.call_llm, to make the "LLM" fail or succeed on demand
  - a fake sleep via the sleep= kwarg, so tests run instantly
  - lightweight fake groq exception classes that pass isinstance() checks
    without needing a real httpx response.

Covers the handler contract documented in failures.py:
  - retries RateLimitError, waiting the server-advertised retry-after (+ margin)
  - retries APIConnectionError and InternalServerError with exponential backoff
  - gives up after MAX_ATTEMPTS and re-raises the last error
  - raises immediately if the server asks to wait longer than MAX_WAIT_SECONDS
    (daily-limit case — must not block the agent for hours)
  - does NOT retry non-transient errors (400s, auth, truncation, malformed args)
  - on_retry callback fires with (err, attempt, wait) per retry
  - retry_after_seconds parses both header and body text, minutes and seconds
"""

from __future__ import annotations

from types import SimpleNamespace

import groq
import pytest

from arxiv_agent import failures, llm


# ---------------------------------------------------------------------------
# Fake exceptions
#
# Real groq errors want an httpx.Response. We subclass and bypass the parent
# __init__ via Exception.__init__, giving us instances that:
#   (a) pass isinstance() against failures._RETRYABLE, and
#   (b) expose exactly the attributes the handler reads: .response.headers
#       (for RateLimitError) and str(err) for the body-text fallback.
# ---------------------------------------------------------------------------


class _FakeRateLimit(groq.RateLimitError):
    def __init__(self, retry_after_header: str | None = None,
                 body_text: str = "rate limit"):
        Exception.__init__(self, body_text)
        self._body_text = body_text
        headers = {"retry-after": retry_after_header} if retry_after_header else {}
        self.response = SimpleNamespace(headers=headers)

    def __str__(self) -> str:
        return self._body_text


class _FakeConnection(groq.APIConnectionError):
    def __init__(self):
        Exception.__init__(self, "connection reset")


class _FakeServerError(groq.InternalServerError):
    def __init__(self):
        Exception.__init__(self, "502 bad gateway")


def _ok_response() -> llm.LLMResponse:
    return llm.LLMResponse(
        content="ok",
        tool_calls=[],
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=10,
    )


def _patch_call_llm(monkeypatch, outcomes):
    """Replace llm.call_llm with a scripted sequence.

    Each entry is either an Exception (raised) or an LLMResponse (returned).
    Consumed in order; tests supply exactly as many as they expect to use.
    """
    def fake(**kwargs):
        x = outcomes.pop(0)
        if isinstance(x, BaseException):
            raise x
        return x
    monkeypatch.setattr(failures.llm, "call_llm", fake)


class _SleepRecorder:
    """Drop-in for time.sleep that records what was waited on."""
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


# ===========================================================================
# Group A — retry_after_seconds parsing
# ===========================================================================


def test_retry_after_from_header():
    """Header is authoritative when present."""
    err = _FakeRateLimit(retry_after_header="13.5")
    assert failures.retry_after_seconds(err) == 13.5


def test_retry_after_from_body_seconds_only():
    """Fallback to parsing 'try again in Ns' from the error body."""
    err = _FakeRateLimit(body_text=(
        "Rate limit reached for model openai/gpt-oss-120b. "
        "Please try again in 7.25s. Need more tokens?"
    ))
    assert failures.retry_after_seconds(err) == 7.25


def test_retry_after_from_body_with_minutes():
    """Body parser handles 'Nm M.MMs' format too."""
    err = _FakeRateLimit(body_text="Please try again in 2m18.5s.")
    assert failures.retry_after_seconds(err) == 2 * 60 + 18.5


def test_retry_after_none_when_absent():
    """No header, no 'try again in' text → None (handler falls back to backoff)."""
    err = _FakeRateLimit(body_text="no hint anywhere")
    assert failures.retry_after_seconds(err) is None


# ===========================================================================
# Group B — call_llm_with_retry behavior
# ===========================================================================


def test_happy_path_no_retry(monkeypatch):
    """First call succeeds → no sleep, no on_retry callback."""
    _patch_call_llm(monkeypatch, [_ok_response()])
    sleep = _SleepRecorder()
    retries: list = []

    resp = failures.call_llm_with_retry(
        sleep=sleep,
        on_retry=lambda e, a, w: retries.append((a, w)),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert resp.content == "ok"
    assert sleep.calls == []
    assert retries == []


def test_retries_rate_limit_then_succeeds(monkeypatch):
    """One 429 → wait server-asked time + margin → retry succeeds."""
    err = _FakeRateLimit(retry_after_header="2.0")
    _patch_call_llm(monkeypatch, [err, _ok_response()])
    sleep = _SleepRecorder()
    retries: list = []

    resp = failures.call_llm_with_retry(
        sleep=sleep,
        on_retry=lambda e, a, w: retries.append((a, w)),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert resp.content == "ok"
    # Server asked 2.0s; handler adds the 0.5s reset margin.
    assert sleep.calls == [pytest.approx(2.5, abs=0.01)]
    assert len(retries) == 1
    attempt_num, wait_seconds = retries[0]
    assert attempt_num == 1
    assert wait_seconds == pytest.approx(2.5, abs=0.01)


def test_retries_connection_error_with_expo_backoff(monkeypatch):
    """Connection errors have no retry-after; handler falls back to
    exponential backoff with jitter: 2^attempt * uniform(0.5, 1.5).
    Attempt 1 → 2^1 * [0.5, 1.5] = [1.0, 3.0] seconds.
    """
    _patch_call_llm(monkeypatch, [_FakeConnection(), _ok_response()])
    sleep = _SleepRecorder()

    resp = failures.call_llm_with_retry(
        sleep=sleep,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert resp.content == "ok"
    assert len(sleep.calls) == 1
    assert 1.0 <= sleep.calls[0] <= 3.0


def test_retries_server_error(monkeypatch):
    """5xx is treated the same as a connection blip."""
    _patch_call_llm(monkeypatch, [_FakeServerError(), _ok_response()])
    sleep = _SleepRecorder()

    resp = failures.call_llm_with_retry(
        sleep=sleep,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert resp.content == "ok"
    assert len(sleep.calls) == 1


def test_exhausts_max_attempts_and_raises(monkeypatch):
    """Persistent 429 → MAX_ATTEMPTS tries, (MAX_ATTEMPTS-1) sleeps, then raise."""
    err = _FakeRateLimit(retry_after_header="1.0")
    _patch_call_llm(monkeypatch, [err] * failures.MAX_ATTEMPTS)
    sleep = _SleepRecorder()

    with pytest.raises(groq.RateLimitError):
        failures.call_llm_with_retry(
            sleep=sleep,
            messages=[{"role": "user", "content": "hi"}],
        )
    # Last attempt raises without sleeping after.
    assert len(sleep.calls) == failures.MAX_ATTEMPTS - 1


def test_raises_immediately_when_wait_exceeds_cap(monkeypatch):
    """Daily token limit: server asks for a wait longer than MAX_WAIT_SECONDS.
    The handler must not block the agent for hours — raise immediately so the
    caller can report the daily limit instead of silently hanging.
    """
    oversized = str(failures.MAX_WAIT_SECONDS + 10)
    err = _FakeRateLimit(retry_after_header=oversized)
    # Second entry would succeed if the handler incorrectly kept trying;
    # confirming it's never consumed is part of the test.
    _patch_call_llm(monkeypatch, [err, _ok_response()])
    sleep = _SleepRecorder()

    with pytest.raises(groq.RateLimitError):
        failures.call_llm_with_retry(
            sleep=sleep,
            messages=[{"role": "user", "content": "hi"}],
        )
    # Zero sleeps — the handler decides to raise before sleeping even once.
    assert sleep.calls == []


def test_does_not_retry_fatal_errors(monkeypatch):
    """Non-transient errors (BadRequest, auth, truncation, malformed args)
    must propagate on the first attempt with no retries.
    """
    _patch_call_llm(monkeypatch, [ValueError("bad request shape")])
    sleep = _SleepRecorder()

    with pytest.raises(ValueError):
        failures.call_llm_with_retry(
            sleep=sleep,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert sleep.calls == []
