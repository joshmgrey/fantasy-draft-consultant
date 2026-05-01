from tests.conftest import login


def test_signup_loads(client):
    res = client.get("/signup")
    assert res.status_code == 200


def test_signup_success(client):
    res = client.post("/signup", data={"email": "new@test.com", "password": "password123"}, follow_redirects=True)
    assert res.status_code == 200
    assert b"Draft Consultant" in res.data


def test_signup_duplicate_email(client, free_user):
    res = client.post("/signup", data={"email": "free@test.com", "password": "password123"}, follow_redirects=True)
    assert b"already exists" in res.data


def test_signup_short_password(client):
    res = client.post("/signup", data={"email": "new@test.com", "password": "short"}, follow_redirects=True)
    assert b"8 characters" in res.data


def test_signup_missing_fields(client):
    res = client.post("/signup", data={"email": "", "password": ""}, follow_redirects=True)
    assert b"required" in res.data


def test_login_loads(client):
    res = client.get("/login")
    assert res.status_code == 200


def test_login_success(client, free_user):
    res = login(client, "free@test.com")
    assert res.status_code == 200
    assert b"Draft Consultant" in res.data


def test_login_wrong_password(client, free_user):
    res = login(client, "free@test.com", "wrongpassword")
    assert b"Invalid email or password" in res.data


def test_login_unknown_email(client):
    res = login(client, "nobody@test.com")
    assert b"Invalid email or password" in res.data


def test_logout(client, free_user):
    login(client, "free@test.com")
    res = client.get("/logout", follow_redirects=True)
    assert res.status_code == 200
    assert b"Sign in" in res.data


def test_index_requires_auth(client):
    res = client.get("/", follow_redirects=True)
    assert b"Sign in" in res.data


def test_authenticated_user_redirected_from_login(client, free_user):
    login(client, "free@test.com")
    res = client.get("/login", follow_redirects=True)
    assert b"Draft Consultant" in res.data
