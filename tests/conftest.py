import pytest
from app import create_app
from app.extensions import db
from app.identity.domain.models import User
from app.subscription.domain.services import FREE_QUERY_LIMIT, season_expiry
from analysis_core.models import PlayerVerdict


class FakeAnalysisClient:
    """Test double for ``app.analysis.client.base.AnalysisClient``.

    Assign it to ``app.extensions["analysis_client"]`` to control what the
    ``/analyze`` route sees without touching ``analysis_core`` or the network::

        app.extensions["analysis_client"] = FakeAnalysisClient(
            error=AnalysisRateLimited("busy"))
    """

    def __init__(self, *, verdict=None, error=None):
        self._verdict = verdict
        self._error = error
        self.calls = []

    def analyze(self, *, player_name, actor):
        self.calls.append((player_name, actor))
        if self._error is not None:
            raise self._error
        if self._verdict is not None:
            return self._verdict
        return PlayerVerdict(
            player=player_name,
            risk_score=2,
            verdict="Draft",
            reason="Top WR value at his ADP.",
        )


@pytest.fixture
def app():
    flask_app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "SECRET_KEY": "test-secret",
    })
    with flask_app.app_context():
        db.create_all()
        yield flask_app
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def free_user(app):
    with app.app_context():
        user = User(email="free@test.com")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        return user


@pytest.fixture
def seasonal_user(app):
    with app.app_context():
        user = User(email="seasonal@test.com", plan="seasonal", plan_expires=season_expiry(), stripe_customer_id="cus_test")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        return user


@pytest.fixture
def maxed_user(app):
    from datetime import datetime
    with app.app_context():
        user = User(
            email="maxed@test.com",
            queries_this_month=FREE_QUERY_LIMIT,
            query_month=datetime.utcnow().strftime("%Y-%m"),
        )
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
        return user


def login(client, email, password="password123"):
    return client.post("/login", data={"email": email, "password": password}, follow_redirects=True)
