import json
import pytest
from tests.conftest import login, FakeAnalysisClient
from analysis_core.models import PlayerVerdict
from analysis_core.services import validate_player_name, parse_verdict
from app.analysis.client.base import AnalysisRateLimited, AnalysisUnavailable


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
    result = parse_verdict("Test Player", 'RISK: 4/10\nVERDICT: Draft\nREASON: (cite index="1-2">Good value</cite> at his ADP.')
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
#
# The route reaches the analysis context through
# ``app.extensions["analysis_client"]``. Tests swap in a FakeAnalysisClient to
# drive the endpoint without touching analysis_core or the network. Exception
# -> response mapping for the in-process client is covered in
# test_analysis_client.py.
# ---------------------------------------------------------------------------

MOCK_VERDICT = PlayerVerdict(
    player="Justin Jefferson", risk_score=2, verdict="Draft",
    reason="Top WR value at his ADP.",
)


def post_analyze(client, player_name):
    return client.post(
        "/analyze",
        data=json.dumps({"player_name": player_name}),
        content_type="application/json",
    )


def use_fake(app, **kwargs):
    fake = FakeAnalysisClient(**kwargs)
    app.extensions["analysis_client"] = fake
    return fake


def test_analyze_requires_auth(client):
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code in (401, 302)


def test_analyze_success(client, free_user, app):
    login(client, "free@test.com")
    use_fake(app, verdict=MOCK_VERDICT)
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 200
    data = res.get_json()
    assert data["verdict"] == "Draft"
    assert data["risk_score"] == 2


def test_analyze_passes_player_and_actor_to_client(client, free_user, app):
    login(client, "free@test.com")
    fake = use_fake(app, verdict=MOCK_VERDICT)
    post_analyze(client, "Justin Jefferson")
    assert len(fake.calls) == 1
    player_name, actor = fake.calls[0]
    assert player_name == "Justin Jefferson"
    with app.app_context():
        from app.identity.domain.models import User
        expected_id = str(User.query.filter_by(email="free@test.com").first().id)
    assert actor.user_id == expected_id


def test_analyze_empty_name(client, free_user, app):
    login(client, "free@test.com")
    use_fake(app)
    res = post_analyze(client, "")
    assert res.status_code == 400
    assert b"empty" in res.data


def test_analyze_invalid_name(client, free_user, app):
    login(client, "free@test.com")
    use_fake(app)
    res = post_analyze(client, "Player<>")
    assert res.status_code == 400


def test_analyze_free_limit_reached(client, maxed_user, app):
    login(client, "maxed@test.com")
    use_fake(app)
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 403
    assert res.get_json()["error"] == "free_limit_reached"


def test_analyze_seasonal_bypasses_limit(client, seasonal_user, app):
    login(client, "seasonal@test.com")
    use_fake(app, verdict=MOCK_VERDICT)
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 200


def test_analyze_rate_limited_returns_clean_error(client, free_user, app):
    """A rate-limited backend surfaces as a clean 503 message, not a traceback."""
    login(client, "free@test.com")
    use_fake(app, error=AnalysisRateLimited(
        "The analysis service is busy right now. Please try again in a minute."))
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 503
    body = res.get_json()
    assert "busy" in body["error"].lower()
    assert "traceback" not in body["error"].lower()


def test_analyze_does_not_increment_query_on_failure(client, free_user, app):
    login(client, "free@test.com")
    use_fake(app, error=AnalysisUnavailable("API error (502): bad gateway"))
    res = post_analyze(client, "Justin Jefferson")
    assert res.status_code == 502
    with app.app_context():
        from app.identity.domain.models import User
        user = User.query.filter_by(email="free@test.com").first()
        assert user.queries_this_month == 0


def test_analyze_increments_query_count(client, free_user, app):
    login(client, "free@test.com")
    use_fake(app, verdict=MOCK_VERDICT)
    post_analyze(client, "Justin Jefferson")
    with app.app_context():
        from app.identity.domain.models import User
        user = User.query.filter_by(email="free@test.com").first()
        assert user.queries_this_month == 1
