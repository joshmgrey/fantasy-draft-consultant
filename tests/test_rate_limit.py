"""Rate-limit keying tests for the /analyze endpoint.

Flask-Limiter is disabled under ``TESTING`` (see the app factory), so the
integration test re-enables it explicitly and resets it afterwards.
"""

import json

import pytest
from unittest.mock import patch

from app.extensions import limiter, user_or_ip_key
from app.identity.domain.models import User
from app.extensions import db
from tests.conftest import login

ROUTES = "app.analysis.presentation.routes"
MOCK_RAW = "RISK: 2/10\nVERDICT: Draft\nREASON: Top WR value at his ADP."


def post_analyze(client, name="Justin Jefferson"):
    return client.post("/analyze", data=json.dumps({"player_name": name}),
                       content_type="application/json")


# --------------------------------------------------------------------------- #
# Unit: the key function identifies the user, not the connection
# --------------------------------------------------------------------------- #
def test_key_is_ip_when_anonymous(app):
    with app.test_request_context("/analyze"):
        assert user_or_ip_key().startswith("ip:")


def test_key_is_user_when_authenticated(app, free_user):
    from flask_login import login_user
    with app.test_request_context("/analyze"):
        user = User.query.filter_by(email="free@test.com").first()
        login_user(user)
        assert user_or_ip_key() == f"user:{user.get_id()}"


# --------------------------------------------------------------------------- #
# Integration: one user's limit does not spill onto another user
# --------------------------------------------------------------------------- #
@pytest.fixture
def limiter_enabled():
    limiter.enabled = True
    try:
        limiter.reset()
    except Exception:
        pass
    yield
    limiter.enabled = False


def test_per_user_buckets_are_isolated(app, limiter_enabled):
    for email in ("a@test.com", "b@test.com"):
        u = User(email=email)
        u.set_password("password123")
        db.session.add(u)
    db.session.commit()

    patches = patch.multiple(
        ROUTES, client=object(), analyze_player=lambda name: MOCK_RAW,
    )

    # Each user gets its own app context so Flask-Login's per-context user
    # cache (g._login_user) doesn't bleed between the two test clients — the
    # conftest fixture keeps one app context open for the whole test.
    with patches, app.app_context():
        client_a = app.test_client()
        login(client_a, "a@test.com")
        codes_a = [post_analyze(client_a).status_code for _ in range(11)]

    # user A: first 10 ok, 11th rate-limited on their own bucket
    assert codes_a[:10] == [200] * 10
    assert codes_a[10] == 429

    with patches, app.app_context():
        client_b = app.test_client()
        login(client_b, "b@test.com")
        # user B starts fresh — not affected by user A hitting the limit
        assert post_analyze(client_b).status_code == 200
