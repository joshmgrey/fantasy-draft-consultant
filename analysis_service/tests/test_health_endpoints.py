"""Liveness / readiness endpoints on the analysis service (service/api.py).

Both are unauthenticated — orchestrator probes carry no actor token.
"""


def test_healthz_ok_without_token(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "anthropic_configured" in body


def test_readyz_ok_when_db_reachable(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


def test_readyz_returns_503_when_db_unreachable(client, monkeypatch):
    from analysis_service.service.persistence import db

    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db.session, "execute", boom)

    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.get_json()
    assert body["status"] == "unavailable"
    assert body["checks"]["database"] == "down"
