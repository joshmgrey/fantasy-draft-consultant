"""Endpoint behavior for the analysis service.

analyze_player is monkeypatched throughout — no network, no API spend.
"""

import anthropic
import httpx2
import pytest

from analysis_core import anthropic_client
from analysis_core.contract import (
    ANALYZE_PATH,
    HEALTH_PATH,
    NOTICE_WITHHELD,
    bearer_header,
    sign_actor,
)
from analysis_service.tests.conftest import SECRET

CLEAN = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."


def _request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def post(client, headers, **body):
    return client.post(ANALYZE_PATH, json=body, headers=headers)


# --------------------------------------------------------------------------- #
# healthz
# --------------------------------------------------------------------------- #
def test_healthz_ok(client):
    res = client.get(HEALTH_PATH)
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"
    assert res.get_json()["anthropic_configured"] is False


def test_healthz_reports_configured(client, configured):
    assert client.get(HEALTH_PATH).get_json()["anthropic_configured"] is True


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def test_missing_token_is_401(client):
    res = client.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"})
    assert res.status_code == 401
    assert res.get_json()["error"] == "unauthorized"


def test_bad_signature_is_401(client, configured):
    bad = {"Authorization": bearer_header(
        sign_actor(user_id="42", plan="free", secret="not-the-secret"))}
    res = post(client, bad, player_name="Bijan Robinson")
    assert res.status_code == 401


def test_missing_service_secret_is_503(configured):
    from analysis_service.service import create_app
    app = create_app({"TESTING": True, "ANALYSIS_TOKEN_SECRET": ""})
    token = sign_actor(user_id="42", plan="free", secret="whatever")
    res = app.test_client().post(
        ANALYZE_PATH, json={"player_name": "Bijan Robinson"},
        headers={"Authorization": bearer_header(token)})
    assert res.status_code == 503


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #
def test_non_json_body_is_422(client, auth_headers, configured):
    res = client.post(ANALYZE_PATH, data="not json", content_type="text/plain",
                      headers=auth_headers)
    assert res.status_code == 422
    assert res.get_json()["error"] == "malformed_request"


def test_missing_player_name_is_422(client, auth_headers, configured):
    res = post(client, auth_headers)
    assert res.status_code == 422


def test_invalid_player_name_is_400(client, auth_headers, configured):
    res = post(client, auth_headers, player_name="Bijan <script>")
    assert res.status_code == 400
    assert res.get_json()["error"] == "invalid_player_name"


def test_not_configured_is_503(client, auth_headers):
    res = post(client, auth_headers, player_name="Bijan Robinson")
    assert res.status_code == 503
    assert res.get_json()["error"] == "unavailable"


# --------------------------------------------------------------------------- #
# happy path + upstream failures
# --------------------------------------------------------------------------- #
def test_success_returns_verdict(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name: CLEAN)
    res = post(client, auth_headers, player_name="Justin Jefferson", request_id="req_1")
    assert res.status_code == 200
    body = res.get_json()
    assert body["player"] == "Justin Jefferson"
    assert body["verdict"] == "Draft"
    assert body["risk_score"] == 2
    assert body["model"] == "claude-opus-4-7"
    assert body["notice"] is None


def test_withheld_message_sets_notice_and_null_verdict(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(
        anthropic_client, "analyze_player",
        lambda name: anthropic_client.WITHHELD_MESSAGE,
    )
    res = post(client, auth_headers, player_name="Bijan Robinson")
    assert res.status_code == 200
    body = res.get_json()
    assert body["notice"] == NOTICE_WITHHELD
    assert body["verdict"] is None and body["risk_score"] is None


def test_rate_limit_is_429(client, auth_headers, configured, monkeypatch):
    def boom(name):
        raise anthropic.RateLimitError(
            "rate limited", response=httpx2.Response(429, request=_request()), body=None)
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    res = post(client, auth_headers, player_name="Bijan Robinson")
    assert res.status_code == 429
    assert res.get_json()["error"] == "upstream_rate_limited"


def test_api_connection_error_is_502(client, auth_headers, configured, monkeypatch):
    def boom(name):
        raise anthropic.APIConnectionError(message="no route", request=_request())
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    res = post(client, auth_headers, player_name="Bijan Robinson")
    assert res.status_code == 502
    assert res.get_json()["error"] == "upstream_error"


def test_unexpected_error_is_contract_shaped_500(auth_headers, monkeypatch):
    from analysis_service.service import create_app
    app = create_app({
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
        "PROPAGATE_EXCEPTIONS": False,
    })
    monkeypatch.setattr(anthropic_client, "client", object())

    def boom(name):
        raise RuntimeError("kaboom internal detail")

    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    res = app.test_client().post(
        ANALYZE_PATH, json={"player_name": "Bijan Robinson"}, headers=auth_headers)
    assert res.status_code == 500
    body = res.get_json()
    assert body["error"] == "unavailable"
    assert "kaboom" not in (body["detail"] or "")
