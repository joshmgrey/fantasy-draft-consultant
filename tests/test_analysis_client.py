"""Unit tests for the in-process analysis client.

These pin the translation from ``analysis_core`` / ``anthropic`` failures into
the client error taxonomy that the ``/analyze`` route maps to HTTP responses.
No network calls: ``analysis_core.anthropic_client`` is monkeypatched.
"""

import anthropic
import httpx2
import pytest

from analysis_core import anthropic_client as core
from analysis_core.models import PlayerVerdict
from app.analysis.client.base import (
    Actor,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisUnavailable,
    build_analysis_client,
)
from app.analysis.client.inprocess import InProcessAnalysisClient

ACTOR = Actor(user_id="1", plan="free")
CLEAN = "RISK: 3/10\nVERDICT: Draft\nREASON: Priced a round light relative to his ADP."


def _request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def configured(monkeypatch):
    """Pretend an ANTHROPIC_API_KEY is set (a truthy client object)."""
    monkeypatch.setattr(core, "client", object())


def _analyze(name="Bijan Robinson"):
    return InProcessAnalysisClient().analyze(player_name=name, actor=ACTOR)


def test_happy_path_returns_parsed_verdict(configured, monkeypatch):
    monkeypatch.setattr(core, "analyze_player", lambda name: CLEAN)
    result = _analyze()
    assert isinstance(result, PlayerVerdict)
    assert result.verdict == "Draft"
    assert result.risk_score == 3


def test_missing_api_key_raises_not_configured(monkeypatch):
    monkeypatch.setattr(core, "client", None)
    with pytest.raises(AnalysisNotConfigured):
        _analyze()


def test_invalid_name_raises_bad_request(configured):
    with pytest.raises(AnalysisBadRequest):
        _analyze("Bijan <script>")


def test_rate_limit_error_becomes_rate_limited(configured, monkeypatch):
    def boom(name):
        raise anthropic.RateLimitError(
            "rate limited", response=httpx2.Response(429, request=_request()), body=None
        )

    monkeypatch.setattr(core, "analyze_player", boom)
    with pytest.raises(AnalysisRateLimited):
        _analyze()


def test_api_status_error_becomes_unavailable(configured, monkeypatch):
    def boom(name):
        raise anthropic.APIStatusError(
            "overloaded", response=httpx2.Response(529, request=_request()), body=None
        )

    monkeypatch.setattr(core, "analyze_player", boom)
    with pytest.raises(AnalysisUnavailable) as exc:
        _analyze()
    assert "529" in str(exc.value)


def test_api_connection_error_becomes_unavailable(configured, monkeypatch):
    def boom(name):
        raise anthropic.APIConnectionError(message="no route to host", request=_request())

    monkeypatch.setattr(core, "analyze_player", boom)
    with pytest.raises(AnalysisUnavailable):
        _analyze()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_defaults_to_inprocess():
    assert isinstance(build_analysis_client({}), InProcessAnalysisClient)


def test_factory_rejects_unknown_mode():
    with pytest.raises(RuntimeError):
        build_analysis_client({"ANALYSIS_MODE": "carrier-pigeon"})
