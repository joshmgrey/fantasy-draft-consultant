"""Unit tests for the analysis wire contract (analysis_core.contract).

Pure serialization / signing helpers — no Flask, no network.
"""

import json

import pytest

from analysis_core.contract import (
    ANALYZE_PATH,
    ERROR_STATUS,
    NOTICE_WITHHELD,
    ActorToken,
    AnalyzeRequest,
    AnalyzeResponse,
    ContractError,
    ErrorCode,
    ErrorResponse,
    TokenExpired,
    TokenInvalid,
    bearer_header,
    parse_bearer,
    sign_actor,
    verify_actor,
)
from analysis_core.models import PlayerVerdict

SECRET = "shared-test-secret"


# --------------------------------------------------------------------------- #
# AnalyzeRequest
# --------------------------------------------------------------------------- #
def test_analyze_request_round_trips():
    req = AnalyzeRequest(player_name="Justin Jefferson", request_id="req_1")
    assert AnalyzeRequest.from_wire(req.to_wire()) == req


def test_analyze_request_defaults_request_id_to_none():
    assert AnalyzeRequest.from_wire({"player_name": "Bijan Robinson"}).request_id is None


def test_analyze_request_survives_json_boundary():
    wire = json.loads(json.dumps(AnalyzeRequest("Ja'Marr Chase").to_wire()))
    assert AnalyzeRequest.from_wire(wire).player_name == "Ja'Marr Chase"


@pytest.mark.parametrize("body", [{}, {"player_name": 123}, {"player_name": None}, []])
def test_analyze_request_rejects_bad_body(body):
    with pytest.raises(ContractError):
        AnalyzeRequest.from_wire(body)


def test_analyze_request_does_not_validate_name_format():
    # Format enforcement is validate_player_name's job, not the codec's.
    assert AnalyzeRequest.from_wire({"player_name": "<script>"}).player_name == "<script>"


# --------------------------------------------------------------------------- #
# AnalyzeResponse
# --------------------------------------------------------------------------- #
def test_analyze_response_round_trips_with_values():
    resp = AnalyzeResponse(
        player="Justin Jefferson", risk_score=2, verdict="Draft",
        reason="Top WR value at his ADP.", model="claude-opus-4-7",
    )
    assert AnalyzeResponse.from_wire(resp.to_wire()) == resp


def test_analyze_response_round_trips_all_null_verdict():
    resp = AnalyzeResponse(player="Bijan Robinson", notice=NOTICE_WITHHELD)
    back = AnalyzeResponse.from_wire(json.loads(json.dumps(resp.to_wire())))
    assert back == resp
    assert back.risk_score is None and back.verdict is None


def test_analyze_response_to_wire_keeps_all_keys():
    keys = set(AnalyzeResponse(player="x").to_wire())
    assert keys == {"player", "risk_score", "verdict", "reason", "model", "notice", "analysis_id"}


def test_from_verdict_and_back():
    verdict = PlayerVerdict(player="Bijan Robinson", risk_score=3, verdict="Draft", reason="Worth it.")
    resp = AnalyzeResponse.from_verdict(verdict, model="claude-opus-4-7", analysis_id="an_1")
    assert resp.to_verdict() == verdict
    assert resp.model == "claude-opus-4-7"
    assert resp.analysis_id == "an_1"


def test_analyze_response_rejects_bool_risk_score():
    with pytest.raises(ContractError):
        AnalyzeResponse.from_wire({"player": "x", "risk_score": True})


def test_analyze_response_rejects_non_string_player():
    with pytest.raises(ContractError):
        AnalyzeResponse.from_wire({"player": None})


# --------------------------------------------------------------------------- #
# ErrorResponse
# --------------------------------------------------------------------------- #
def test_error_response_round_trips():
    err = ErrorResponse(error=ErrorCode.INVALID_PLAYER_NAME, detail="bad name", request_id="req_9")
    assert ErrorResponse.from_wire(err.to_wire()) == err


@pytest.mark.parametrize("code,status", list(ERROR_STATUS.items()))
def test_error_response_status_code_mapping(code, status):
    assert ErrorResponse(error=code).status_code == status


def test_error_response_unknown_code_is_500():
    assert ErrorResponse(error="something_new").status_code == 500


def test_error_response_rejects_empty_code():
    with pytest.raises(ContractError):
        ErrorResponse.from_wire({"error": ""})


# --------------------------------------------------------------------------- #
# Actor token
# --------------------------------------------------------------------------- #
def test_sign_then_verify_round_trips():
    token = sign_actor(user_id="42", plan="seasonal", secret=SECRET, now=1_000)
    actor = verify_actor(token, secret=SECRET, now=1_030)
    assert actor == ActorToken(sub="42", plan="seasonal", iat=1_000, exp=1_120)


def test_verify_rejects_wrong_secret():
    token = sign_actor(user_id="42", plan="free", secret=SECRET, now=1_000)
    with pytest.raises(TokenInvalid):
        verify_actor(token, secret="different-secret", now=1_030)


def test_verify_rejects_tampered_payload():
    token = sign_actor(user_id="42", plan="free", secret=SECRET, now=1_000)
    body, sig = token.split(".")
    flipped = body[:-1] + ("A" if body[-1] != "A" else "B")
    with pytest.raises(TokenInvalid):
        verify_actor(f"{flipped}.{sig}", secret=SECRET, now=1_030)


def test_verify_rejects_tampered_signature():
    token = sign_actor(user_id="42", plan="free", secret=SECRET, now=1_000)
    body, sig = token.split(".")
    with pytest.raises(TokenInvalid):
        verify_actor(f"{body}.{sig[:-1]}{'A' if sig[-1] != 'A' else 'B'}", secret=SECRET, now=1_030)


@pytest.mark.parametrize("bad", ["", "no-dot", "a.b.c", "onlybody."])
def test_verify_rejects_malformed_token(bad):
    with pytest.raises(TokenInvalid):
        verify_actor(bad, secret=SECRET, now=1_030)


def test_verify_rejects_expired_token():
    token = sign_actor(user_id="42", plan="free", secret=SECRET, ttl_seconds=120, now=1_000)
    with pytest.raises(TokenExpired):
        verify_actor(token, secret=SECRET, now=1_200)


def test_verify_allows_expiry_within_leeway():
    token = sign_actor(user_id="42", plan="free", secret=SECRET, ttl_seconds=120, now=1_000)
    actor = verify_actor(token, secret=SECRET, now=1_125, leeway_seconds=10)
    assert actor.sub == "42"


def test_sign_requires_secret():
    with pytest.raises(TokenInvalid):
        sign_actor(user_id="42", plan="free", secret="")


# --------------------------------------------------------------------------- #
# Bearer header helpers
# --------------------------------------------------------------------------- #
def test_bearer_header_round_trip():
    assert parse_bearer(bearer_header("abc.def")) == "abc.def"


@pytest.mark.parametrize("header", [None, "", "Token abc", "abc.def"])
def test_parse_bearer_rejects_non_bearer(header):
    with pytest.raises(TokenInvalid):
        parse_bearer(header)


def test_analyze_path_is_versioned():
    assert ANALYZE_PATH == "/v1/analyses"
