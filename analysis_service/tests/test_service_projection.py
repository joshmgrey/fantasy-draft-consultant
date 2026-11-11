"""The analysis service passes the model projection to the LLM and returns it."""

from analysis_core import anthropic_client
from analysis_core import projection as projection_model
from analysis_core.contract import ANALYZE_PATH
from analysis_core.projection import Projection

CLEAN = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."

SAMPLE = Projection(
    player="Ja'Marr Chase", position="WR", team="CIN", season=2026,
    projected_ppg=18.4, last_ppg=19.6, last_games=16, typical_error=2.4,
)


def post(client, headers, **body):
    return client.post(ANALYZE_PATH, json=body, headers=headers)


def test_projection_reaches_prompt_and_response(client, auth_headers, configured, monkeypatch):
    seen = {}
    monkeypatch.setattr(projection_model, "project_player", lambda name: SAMPLE)

    def fake(name, projection=None, **_):
        seen["projection"] = projection
        return CLEAN

    monkeypatch.setattr(anthropic_client, "analyze_player", fake)
    res = post(client, auth_headers, player_name="Ja'Marr Chase")
    assert res.status_code == 200
    assert seen["projection"] is SAMPLE
    assert res.get_json()["projected_ppg"] == 18.4


def test_no_projection_returns_null(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name, **_: CLEAN)
    res = post(client, auth_headers, player_name="Some Rookie")
    assert res.status_code == 200
    assert res.get_json()["projected_ppg"] is None
