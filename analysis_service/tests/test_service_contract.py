"""Contract tests: the service's real HTTP responses must round-trip through
the shared analysis_core.contract dataclasses, and error statuses must match
ERROR_STATUS.
"""

import anthropic
import httpx2
import pytest

from analysis_core import anthropic_client
from analysis_core.contract import (
    ANALYZE_PATH,
    ERROR_STATUS,
    AnalyzeResponse,
    ErrorResponse,
)

CLEAN = "RISK: 4/10\nVERDICT: Draft\nREASON: Priced a round light relative to his ADP."


def _request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_success_body_parses_as_AnalyzeResponse(client, auth_headers, configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name: CLEAN)
    res = client.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"}, headers=auth_headers)

    parsed = AnalyzeResponse.from_wire(res.get_json())
    assert parsed.to_verdict().verdict == "Draft"
    assert parsed.model == "claude-opus-4-7"


@pytest.mark.parametrize(
    "make_request",
    [
        pytest.param(
            lambda c, h: c.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"}),
            id="unauthorized",
        ),
        pytest.param(
            lambda c, h: c.post(ANALYZE_PATH, data="x", content_type="text/plain", headers=h),
            id="malformed_request",
        ),
        pytest.param(
            lambda c, h: c.post(ANALYZE_PATH, json={"player_name": "Bad <x>"}, headers=h),
            id="invalid_player_name",
        ),
        pytest.param(
            lambda c, h: c.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"}, headers=h),
            id="unavailable",  # anthropic not configured
        ),
    ],
)
def test_error_bodies_parse_as_ErrorResponse(client, auth_headers, make_request):
    res = make_request(client, auth_headers)
    parsed = ErrorResponse.from_wire(res.get_json())
    assert ERROR_STATUS[parsed.error] == res.status_code


def test_rate_limit_error_body_matches_contract(client, auth_headers, configured, monkeypatch):
    def boom(name):
        raise anthropic.RateLimitError(
            "rl", response=httpx2.Response(429, request=_request()), body=None)
    monkeypatch.setattr(anthropic_client, "analyze_player", boom)
    res = client.post(ANALYZE_PATH, json={"player_name": "Bijan Robinson"}, headers=auth_headers)
    parsed = ErrorResponse.from_wire(res.get_json())
    assert res.status_code == ERROR_STATUS[parsed.error] == 429


def test_request_id_is_echoed_in_errors(client, auth_headers, configured):
    res = client.post(
        ANALYZE_PATH,
        json={"player_name": "Bad <x>", "request_id": "req_echo"},
        headers=auth_headers,
    )
    assert ErrorResponse.from_wire(res.get_json()).request_id == "req_echo"
