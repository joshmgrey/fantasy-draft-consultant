import pytest

from analysis_core.contract import bearer_header, sign_actor
from analysis_service.service import create_app

SECRET = "test-actor-secret"


@pytest.fixture
def app():
    return create_app({
        "TESTING": True,
        "ANALYSIS_TOKEN_SECRET": SECRET,
        "TOKEN_LEEWAY_SECONDS": 0,
    })


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_headers():
    token = sign_actor(user_id="42", plan="free", secret=SECRET)
    return {"Authorization": bearer_header(token)}


@pytest.fixture
def configured(monkeypatch):
    """Pretend ANTHROPIC_API_KEY is set (truthy client)."""
    from analysis_core import anthropic_client
    monkeypatch.setattr(anthropic_client, "client", object())
