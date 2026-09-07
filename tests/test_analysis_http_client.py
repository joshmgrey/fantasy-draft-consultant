"""HttpAnalysisClient <-> analysis service contract.

The client runs against the REAL analysis_service app (via an in-process
session shim), so a drift between the two sides fails here. anthropic is
stubbed — no network, no spend.
"""

import anthropic
import httpx2
import pytest
import requests

from analysis_core import anthropic_client
from analysis_core.contract import sign_actor
from app.analysis.client.base import (
    Actor,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisServiceDown,
    AnalysisUnavailable,
)
from app.analysis.client.http import HttpAnalysisClient
from analysis_service.service import create_app as create_service
from tests.http_shim import AppSession, DeadSession, StubResponse

SECRET = "shared-contract-secret"
ACTOR = Actor(user_id="42", plan="free")
CLEAN = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."


def _req():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(anthropic_client, "client", object())
    return create_service({
        "TESTING": True,
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
    })


@pytest.fixture
def http_client(service):
    return HttpAnalysisClient(
        base_url="http://analysis.test", token_secret=SECRET,
        session=AppSession(service),
    )


def analyze(http_client, name="Justin Jefferson"):
    return http_client.analyze(player_name=name, actor=ACTOR)


# --------------------------------------------------------------------------- #
# happy path — the field-for-field contract check
# --------------------------------------------------------------------------- #
def test_success_round_trips_a_verdict(http_client, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    verdict = analyze(http_client)
    assert verdict.player == "Justin Jefferson"
    assert verdict.verdict == "Draft"
    assert verdict.risk_score == 2


def test_withheld_message_comes_back_as_null_verdict(http_client, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player",
                        lambda n: anthropic_client.WITHHELD_MESSAGE)
    verdict = analyze(http_client)
    assert verdict.verdict is None and verdict.risk_score is None


def test_sends_bearer_token_the_service_accepts(http_client, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    analyze(http_client)
    sent = http_client._session.calls[-1]["headers"]["Authorization"]
    assert sent.startswith("Bearer ")


# --------------------------------------------------------------------------- #
# status -> exception mapping
# --------------------------------------------------------------------------- #
def test_invalid_name_maps_to_bad_request(http_client):
    with pytest.raises(AnalysisBadRequest):
        analyze(http_client, "Bad <script>")


def test_service_429_maps_to_rate_limited(http_client, monkeypatch):
    def boom(n):
        raise anthropic.RateLimitError(
            "rl", response=httpx2.Response(429, request=_req()), body=None)
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    with pytest.raises(AnalysisRateLimited):
        analyze(http_client)


def test_service_502_maps_to_unavailable(http_client, monkeypatch):
    def boom(n):
        raise anthropic.APIConnectionError(message="down", request=_req())
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    with pytest.raises(AnalysisUnavailable):
        analyze(http_client)


def test_service_503_maps_to_service_down(service, monkeypatch):
    # anthropic not configured on the service -> it returns 503 unavailable
    monkeypatch.setattr(anthropic_client, "client", None)
    client = HttpAnalysisClient(base_url="http://analysis.test", token_secret=SECRET,
                                session=AppSession(service))
    with pytest.raises(AnalysisServiceDown):
        analyze(client)


def test_rejected_token_maps_to_not_configured(service):
    client = HttpAnalysisClient(base_url="http://analysis.test",
                                token_secret="wrong-secret", session=AppSession(service))
    with pytest.raises(AnalysisNotConfigured):
        analyze(client)


# --------------------------------------------------------------------------- #
# transport failures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc", [
    requests.ConnectionError("refused"),
    requests.Timeout("read timed out"),
])
def test_transport_failure_maps_to_service_down(exc):
    client = HttpAnalysisClient(base_url="http://analysis.test", token_secret=SECRET,
                                session=DeadSession(exc))
    with pytest.raises(AnalysisServiceDown):
        client.analyze(player_name="Justin Jefferson", actor=ACTOR)


def test_unreadable_200_body_maps_to_unavailable():
    class _Garbage:
        def post(self, *a, **k):
            return StubResponse(200, {"not": "a verdict"})

    client = HttpAnalysisClient(base_url="http://analysis.test", token_secret=SECRET,
                                session=_Garbage())
    with pytest.raises(AnalysisUnavailable):
        client.analyze(player_name="Justin Jefferson", actor=ACTOR)


# --------------------------------------------------------------------------- #
# misconfiguration is caught before any request
# --------------------------------------------------------------------------- #
def test_missing_base_url_raises_not_configured():
    client = HttpAnalysisClient(base_url="", token_secret=SECRET, session=DeadSession())
    with pytest.raises(AnalysisNotConfigured):
        client.analyze(player_name="x", actor=ACTOR)


def test_missing_secret_raises_not_configured():
    client = HttpAnalysisClient(base_url="http://x", token_secret="", session=DeadSession())
    with pytest.raises(AnalysisNotConfigured):
        client.analyze(player_name="x", actor=ACTOR)
