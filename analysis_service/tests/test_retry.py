"""Retry-with-backoff tests for the Anthropic API call.

``analysis_core.anthropic_client._create_message`` wraps
``client.messages.create`` in a tenacity retry that fires on
``anthropic.RateLimitError`` with exponential backoff, capped at 3 attempts.
These tests stand in a fake Anthropic client so no network calls happen, and
patch out tenacity's sleep so the backoff doesn't slow the suite.
"""

import anthropic
import httpx2
import pytest

from analysis_core import anthropic_client as ac


CLEAN_VERDICT = "RISK: 3/10\nVERDICT: Draft\nREASON: Priced a round light relative to his ADP."


def _rate_limit_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request, headers={"retry-after": "1"})
    return anthropic.RateLimitError("rate limited", response=response, body=None)


class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FlakyClient:
    """Raises RateLimitError on the first create() call, then succeeds."""

    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.calls = 0
        self.messages = self  # so ``client.messages.create`` resolves here

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _rate_limit_error()
        return _Response([_Block("text", text=CLEAN_VERDICT)])


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _seconds: None)


@pytest.fixture
def restore_client():
    original = ac.client
    yield
    ac.client = original


def test_retry_recovers_after_one_rate_limit(restore_client):
    fake = _FlakyClient(fail_times=1)
    ac.client = fake

    result = ac.analyze_player("Bijan Robinson")

    assert fake.calls == 2, "expected one failed attempt then one successful retry"
    assert result == CLEAN_VERDICT


def test_retry_gives_up_after_three_attempts(restore_client):
    fake = _FlakyClient(fail_times=99)
    ac.client = fake

    with pytest.raises(anthropic.RateLimitError):
        ac.analyze_player("Bijan Robinson")

    assert fake.calls == 3, "retry should cap at 3 attempts"
