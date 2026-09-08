"""Operational health endpoints for the core service.

Not a domain context — these exist purely so an orchestrator (Kubernetes,
docker-compose, Render) can tell whether the process is alive and whether it
can actually serve traffic. Both routes are unauthenticated and side-effect
free.

    GET /healthz   liveness  — the worker is up. No I/O. Restart the pod if this fails.
    GET /readyz    readiness — the DB answers. Pull the pod out of the Service
                               (stop routing traffic) if this fails, but don't
                               restart it — the dependency, not the app, is down.
"""

from flask import Blueprint
from sqlalchemy import text

from .extensions import db

bp = Blueprint("health", __name__)


@bp.get("/healthz")
def healthz():
    return {"status": "ok"}


@bp.get("/readyz")
def readyz():
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        return {"status": "unavailable", "checks": {"database": "down"}}, 503
    return {"status": "ok", "checks": {"database": "ok"}}
