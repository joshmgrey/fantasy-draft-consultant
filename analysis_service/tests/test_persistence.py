"""The analysis service's audit log (analysis_record).

The service owns this table outright — the core app never touches it.
"""

import anthropic
import httpx2
import pytest

from analysis_core import anthropic_client
from analysis_core.contract import ANALYZE_PATH, bearer_header, sign_actor
from analysis_service.service.persistence import AnalysisRecord, db
from analysis_service.tests.conftest import SECRET

CLEAN = "RISK: 3/10\nVERDICT: Draft\nREASON: Priced a round light relative to his ADP."


def _req():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def post(client, headers, **body):
    return client.post(ANALYZE_PATH, json=body, headers=headers)


def rows(app):
    with app.app_context():
        return AnalysisRecord.query.order_by(AnalysisRecord.id).all()


def test_success_writes_a_record(client, app, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    res = post(client, auth_headers, player_name="Bijan Robinson", request_id="req_7")

    assert res.status_code == 200
    analysis_id = res.get_json()["analysis_id"]
    assert analysis_id and analysis_id.startswith("an_")

    (row,) = rows(app)
    assert row.public_id == analysis_id
    assert row.player_name == "Bijan Robinson"
    assert row.requested_by == "42"          # actor 'sub' from the test token
    assert row.request_id == "req_7"
    assert row.verdict == "Draft" and row.risk_score == 3
    assert row.model == "claude-opus-4-7"
    assert row.latency_ms is not None
    assert row.injection_flagged is False


def test_withheld_analysis_is_flagged(client, app, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player",
                        lambda n: anthropic_client.WITHHELD_MESSAGE)
    post(client, auth_headers, player_name="Bijan Robinson")

    (row,) = rows(app)
    assert row.injection_flagged is True
    assert row.verdict is None and row.risk_score is None


def test_failed_analyses_are_not_recorded(client, app, auth_headers, configured, monkeypatch):
    # invalid name -> 400, no record
    post(client, auth_headers, player_name="Bad <x>")
    # upstream error -> 502, no record
    def boom(n):
        raise anthropic.APIConnectionError(message="down", request=_req())
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    post(client, auth_headers, player_name="Bijan Robinson")

    assert rows(app) == []


def test_persistence_failure_does_not_break_analysis(client, app, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)

    def explode(*args, **kwargs):
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(db.session, "add", explode)
    res = post(client, auth_headers, player_name="Bijan Robinson")

    assert res.status_code == 200
    assert res.get_json()["analysis_id"] is None
    assert res.get_json()["verdict"] == "Draft"


def test_persistence_can_be_disabled(auth_headers, configured, monkeypatch):
    from analysis_service.service import create_app
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    app = create_app({
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
        "PERSIST_ANALYSES": False,
    })
    c = app.test_client()
    res = c.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"}, headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()["analysis_id"] is None
    assert c.get(ANALYZE_PATH, headers=auth_headers).get_json() == {"items": []}


# --------------------------------------------------------------------------- #
# GET /v1/analyses  (history)
# --------------------------------------------------------------------------- #
def test_history_returns_user_rows_newest_first(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    for name in ("Bijan Robinson", "Justin Jefferson", "Ja'Marr Chase"):
        post(client, auth_headers, player_name=name)

    items = client.get(ANALYZE_PATH, headers=auth_headers).get_json()["items"]
    assert [i["player"] for i in items] == ["Ja'Marr Chase", "Justin Jefferson", "Bijan Robinson"]
    assert items[0]["analysis_id"].startswith("an_")


def test_history_is_scoped_by_user(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    post(client, auth_headers, player_name="Bijan Robinson")  # user 42

    other = {"Authorization": bearer_header(
        sign_actor(user_id="99", plan="free", secret=SECRET))}
    assert client.get(ANALYZE_PATH, headers=other).get_json()["items"] == []
    assert client.get(f"{ANALYZE_PATH}?user_id=42", headers=other).get_json()["items"]


def test_history_respects_limit(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    for _ in range(4):
        post(client, auth_headers, player_name="Bijan Robinson")
    items = client.get(f"{ANALYZE_PATH}?limit=2", headers=auth_headers).get_json()["items"]
    assert len(items) == 2


@pytest.mark.parametrize("limit", ["-1", "0", "-999"])
def test_history_clamps_non_positive_limit(client, auth_headers, configured, monkeypatch, limit):
    """A negative limit must not become SQL 'LIMIT -1' (all rows / Postgres 500)."""
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda n: CLEAN)
    for _ in range(3):
        post(client, auth_headers, player_name="Bijan Robinson")
    res = client.get(f"{ANALYZE_PATH}?limit={limit}", headers=auth_headers)
    assert res.status_code == 200
    assert len(res.get_json()["items"]) == 1


def test_history_rejects_non_integer_limit(client, auth_headers, configured):
    res = client.get(f"{ANALYZE_PATH}?limit=abc", headers=auth_headers)
    assert res.status_code == 422


def test_history_requires_auth(client):
    assert client.get(ANALYZE_PATH).status_code == 401
