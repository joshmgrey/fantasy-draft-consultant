"""Operational health endpoints on the core service (app/health.py)."""


def test_healthz_is_ok_and_unauthenticated(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_readyz_ok_when_db_reachable(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


def test_readyz_returns_503_when_db_unreachable(client, monkeypatch):
    from app.extensions import db

    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db.session, "execute", boom)

    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "unavailable"
    assert body["checks"]["database"] == "down"
