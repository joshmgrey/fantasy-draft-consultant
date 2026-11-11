"""The analysis service hands retrieved player history to the LLM."""

from analysis_core import anthropic_client, retrieval
from analysis_core.contract import ANALYZE_PATH

CLEAN = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."


def test_retrieved_context_reaches_the_prompt(client, auth_headers, configured, monkeypatch):
    seen = {}
    monkeypatch.setattr(retrieval, "retrieve_context", lambda name: f"history for {name}")

    def fake(name, retrieved_context=None, projection=None):
        seen["context"] = retrieved_context
        return CLEAN

    monkeypatch.setattr(anthropic_client, "analyze_player", fake)
    res = client.post(ANALYZE_PATH, json={"player_name": "Puka Nacua"}, headers=auth_headers)
    assert res.status_code == 200
    assert seen["context"] == "history for Puka Nacua"


def test_no_store_means_no_context(client, auth_headers, configured, monkeypatch):
    seen = {}

    def fake(name, retrieved_context=None, projection=None):
        seen["context"] = retrieved_context
        return CLEAN

    monkeypatch.setattr(anthropic_client, "analyze_player", fake)
    res = client.post(ANALYZE_PATH, json={"player_name": "Puka Nacua"}, headers=auth_headers)
    assert res.status_code == 200
    assert seen["context"] is None
