"""The core /analyze route running in ANALYSIS_MODE=http.

Covers the end-to-end happy path and — the part that needed a real answer —
how the route behaves when the analysis service is unreachable or failing:
a clean 503, no traceback, and the user's quota is NOT spent.
"""

import json

import pytest

from analysis_core import anthropic_client
from analysis_service.service import create_app as create_service
from app.analysis.client.http import HttpAnalysisClient
from tests.conftest import login
from tests.http_shim import AppSession, DeadSession

SECRET = "http-mode-secret"
CLEAN = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."


def _boom(_name):
    raise RuntimeError("kaboom internal detail")


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(anthropic_client, "client", object())
    return create_service({
        "TESTING": True,
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
    })


def wire(app, session):
    app.extensions["analysis_client"] = HttpAnalysisClient(
        base_url="http://analysis.test", token_secret=SECRET, session=session,
    )


def post_analyze(client, name="Justin Jefferson"):
    return client.post("/analyze", data=json.dumps({"player_name": name}),
                       content_type="application/json")


def queries_used(app, email="free@test.com"):
    with app.app_context():
        from app.identity.domain.models import User
        return User.query.filter_by(email=email).first().queries_this_month


def test_end_to_end_success(client, free_user, app, service, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    wire(app, AppSession(service))
    login(client, "free@test.com")

    res = post_analyze(client)
    assert res.status_code == 200
    body = res.get_json()
    assert body["verdict"] == "Draft" and body["risk_score"] == 2
    assert queries_used(app) == 1


def test_invalid_name_still_400(client, free_user, app, service):
    wire(app, AppSession(service))
    login(client, "free@test.com")
    res = post_analyze(client, "Player<>")
    assert res.status_code == 400
    assert queries_used(app) == 0


def test_rate_limited_service_returns_503(client, free_user, app, service, monkeypatch):
    import anthropic
    import httpx2

    def rl(_n):
        raise anthropic.RateLimitError(
            "rl",
            response=httpx2.Response(
                429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
            body=None,
        )

    monkeypatch.setattr(anthropic_client, "analyze_player", rl)
    wire(app, AppSession(service))
    login(client, "free@test.com")

    res = post_analyze(client)
    assert res.status_code == 503
    assert "busy" in res.get_json()["error"].lower()
    assert queries_used(app) == 0


def test_service_down_returns_clean_503_and_does_not_charge(client, free_user, app):
    wire(app, DeadSession())
    login(client, "free@test.com")

    res = post_analyze(client)
    assert res.status_code == 503
    body = res.get_json()
    assert "busy" in body["error"].lower()
    assert "traceback" not in json.dumps(body).lower()
    assert queries_used(app) == 0


def test_service_5xx_returns_clean_503_and_does_not_charge(client, free_user, app, monkeypatch):
    monkeypatch.setattr(anthropic_client, "client", object())
    monkeypatch.setattr(anthropic_client, "analyze_player", _boom)
    # PROPAGATE_EXCEPTIONS=False so the service returns its contract-shaped 500
    # rather than re-raising into the shim.
    svc = create_service({
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
        "PROPAGATE_EXCEPTIONS": False,
    })
    wire(app, AppSession(svc))
    login(client, "free@test.com")

    res = post_analyze(client)
    assert res.status_code == 503
    body = res.get_json()
    assert "kaboom" not in json.dumps(body)
    assert queries_used(app) == 0
