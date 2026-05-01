import json
import pytest
from unittest.mock import patch
from tests.conftest import login
from app.analysis.domain.services import validate_player_name, parse_verdict
from app.analysis.infrastructure import anthropic_client as ac_module

MOCK_VERDICT_DICT = {"player": "Justin Jefferson", "risk_score": 2, "verdict": "Draft", "reason": "Top WR value at his ADP."}


def _mock_verdict():
    from app.analysis.domain.models import PlayerVerdict
    return PlayerVerdict(player="Justin Jefferson", risk_score=2, verdict="Draft", reason="Top WR value at his ADP.")


# ---------------------------------------------------------------------------
# Unit: validate_player_name
# ---------------------------------------------------------------------------

def test_validate_valid_name():
    assert validate_player_name("  Justin Jefferson  ") == "Justin Jefferson"


def test_validate_apostrophe():
    assert validate_player_name("Ja'Marr Chase") == "Ja'Marr Chase"


def test_validate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_player_name("")


def test_validate_too_long_raises():
    with pytest.raises(ValueError, match="too long"):
        validate_player_name("A" * 51)


def test_validate_invalid_chars_raises():
    with pytest.raises(ValueError, match="invalid characters"):
        validate_player_name("Player<script>")


# ---------------------------------------------------------------------------
# Unit: parse_verdict
# ---------------------------------------------------------------------------

def test_parse_verdict_draft():
    result = parse_verdict("Test Player", "RISK: 3/10\nVERDICT: Draft\nREASON: Solid value at ADP.")
    assert result.risk_score == 3
    assert result.verdict == "Draft"
    assert result.reason == "Solid value at ADP."


def test_parse_verdict_pass():
    result = parse_verdict("Test Player", "RISK: 8/10\nVERDICT: Pass\nREASON: Overpriced relative to projection.")
    assert result.risk_score == 8
    assert result.verdict == "Pass"


def test_parse_verdict_strips_cite_tags():
    result = parse_verdict("Test Player", 'RISK: 4/10\nVERDICT: Draft\nREASON: <cite index="1-2">Good value</cite> at his ADP.')
    assert "<cite" not in result.reason
    assert "Good value" in result.reason


def test_parse_verdict_case_insensitive():
    result = parse_verdict("Test Player", "risk: 5/10\nverdict: draft\nreason: Worth the pick.")
    assert result.risk_score == 5
    assert result.verdict == "Draft"


def test_parse_verdict_missing_fields():
    result = parse_verdict("Test Player", "No structured output here.")
    assert result.risk_score is None
    assert result.verdict is None
    assert result.reason is None


# ---------------------------------------------------------------------------
# Integration: /analyze endpoint
# ---------------------------------------------------------------------------

def post_analyze(client, player_name):
    return client.post(
        "/analyze",
        data=json.dumps({"player_name": player_name}),
        content_type="application/json",
    )


def test_analyze_requires_auth(client):
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code in (401, 302)


MOCK_RAW = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."
ROUTES = "app.analysis.presentation.routes"


def test_analyze_success(client, free_user):
    login(client, "free@test.com")
    with patch(f"{ROUTES}.client", new=object()), \
         patch(f"{ROUTES}.analyze_player", return_value=MOCK_RAW):
        res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 200
    data = res.get_json()
    assert data["verdict"] == "Draft"
    assert data["risk_score"] == 2


def test_analyze_empty_name(client, free_user):
    login(client, "free@test.com")
    with patch(f"{ROUTES}.client", new=object()):
        res = post_analyze(client, "")
    assert res.status_code == 400
    assert b"empty" in res.data


def test_analyze_invalid_name(client, free_user):
    login(client, "free@test.com")
    with patch(f"{ROUTES}.client", new=object()):
        res = post_analyze(client, "Player<>")
    assert res.status_code == 400


def test_analyze_free_limit_reached(client, maxed_user):
    login(client, "maxed@test.com")
    with patch(f"{ROUTES}.client", new=object()):
        res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 403
    assert res.get_json()["error"] == "free_limit_reached"


def test_analyze_seasonal_bypasses_limit(client, seasonal_user):
    login(client, "seasonal@test.com")
    with patch(f"{ROUTES}.client", new=object()), \
         patch(f"{ROUTES}.analyze_player", return_value=MOCK_RAW):
        res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 200


def test_analyze_increments_query_count(client, free_user, app):
    login(client, "free@test.com")
    with patch(f"{ROUTES}.client", new=object()), \
         patch(f"{ROUTES}.analyze_player", return_value=MOCK_RAW):
        post_analyze(client, "Justin Jefferson")
    with app.app_context():
        from app.identity.domain.models import User
        user = User.query.filter_by(email="free@test.com").first()
        assert user.queries_this_month == 1
