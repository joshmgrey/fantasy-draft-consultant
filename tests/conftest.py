import pytest
from app import app as flask_app, db, User


@pytest.fixture
def app():
    flask_app.config.update({
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
    user = User(email="free@test.com")
    user.set_password("password123")
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def seasonal_user(app):
    from app import _season_expiry
    user = User(email="seasonal@test.com", plan="seasonal", plan_expires=_season_expiry(), stripe_customer_id="cus_test")
    user.set_password("password123")
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def maxed_user(app):
    from app import FREE_QUERY_LIMIT
    from datetime import datetime
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
