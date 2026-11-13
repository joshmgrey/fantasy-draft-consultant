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


def test_projection_scoring_is_returned(client, auth_headers, configured, monkeypatch):
    dst = Projection(
        player="Pittsburgh Steelers D/ST", position="D/ST", team="PIT", season=2026,
        projected_ppg=6.4, last_ppg=6.9, last_games=17, typical_error=1.6,
    )
    monkeypatch.setattr(projection_model, "project_player", lambda name: dst)
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name, **_: CLEAN)
    body = post(client, auth_headers, player_name="Steelers D/ST").get_json()
    assert (body["projected_ppg"], body["projection_scoring"]) == (6.4, "D/ST")

    monkeypatch.setattr(projection_model, "project_player", lambda name: SAMPLE)
    body = post(client, auth_headers, player_name="Ja'Marr Chase").get_json()
    assert body["projection_scoring"] == "PPR"
