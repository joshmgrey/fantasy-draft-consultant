import logging
import time

import anthropic
from flask import Blueprint, current_app, g, request

from analysis_core import anthropic_client
from analysis_core.contract import (
    ANALYZE_PATH,
    HEALTH_PATH,
    NOTICE_WITHHELD,
    AnalyzeResponse,
    ContractError,
    ErrorCode,
)
from analysis_core.services import parse_verdict, validate_player_name

from .auth import require_actor
from .persistence import AnalysisRecord, db
from .schemas import analyze_response, error_response, parse_analyze_request

log = logging.getLogger("analysis_service")

bp = Blueprint("analysis_service", __name__)

# The model id analyze_player asks for. Surfaced in the response so the core app
# and its clients can record which model produced a verdict.
MODEL = "claude-opus-4-7"

_HISTORY_MAX = 100


@bp.get(HEALTH_PATH)
def healthz():
    return {
        "status": "ok",
        "anthropic_configured": anthropic_client.client is not None,
    }


@bp.post(ANALYZE_PATH)
@require_actor
def create_analysis():
    try:
        req = parse_analyze_request()
    except ContractError as e:
        return error_response(ErrorCode.MALFORMED_REQUEST, str(e))

    try:
        name = validate_player_name(req.player_name)
    except ValueError as e:
        return error_response(
            ErrorCode.INVALID_PLAYER_NAME, str(e), request_id=req.request_id
        )

    if anthropic_client.client is None:
        return error_response(
            ErrorCode.UNAVAILABLE,
            "ANTHROPIC_API_KEY is not set",
            request_id=req.request_id,
        )

    log.info(
        "analyze player=%r actor=%s request_id=%s",
        name, g.actor.sub, req.request_id,
    )

    started = time.monotonic()
    try:
        raw_text = anthropic_client.analyze_player(name)
    except anthropic.RateLimitError:
        return error_response(
            ErrorCode.UPSTREAM_RATE_LIMITED,
            "Anthropic rate limit hit",
            request_id=req.request_id,
        )
    except anthropic.APIStatusError as e:
        return error_response(
            ErrorCode.UPSTREAM_ERROR,
            f"API error ({e.status_code}): {e.message}",
            request_id=req.request_id,
        )
    except anthropic.APIConnectionError:
        return error_response(
            ErrorCode.UPSTREAM_ERROR,
            "Could not reach Anthropic API",
            request_id=req.request_id,
        )
    latency_ms = int((time.monotonic() - started) * 1000)

    verdict = parse_verdict(name, raw_text)
    withheld = raw_text == anthropic_client.WITHHELD_MESSAGE
    notice = NOTICE_WITHHELD if withheld else None

    analysis_id = _record(
        name=name,
        request_id=req.request_id,
        verdict=verdict,
        latency_ms=latency_ms,
        injection_flagged=withheld,
    )

    return analyze_response(
        AnalyzeResponse.from_verdict(
            verdict, model=MODEL, notice=notice, analysis_id=analysis_id
        )
    )


@bp.get(ANALYZE_PATH)
@require_actor
def list_analyses():
    """History for a user. Cross-service reads go through this, never a DB join."""
    if not current_app.config.get("PERSIST_ANALYSES", True):
        return {"items": []}

    user_id = request.args.get("user_id") or g.actor.sub
    try:
        limit = min(int(request.args.get("limit", 20)), _HISTORY_MAX)
    except (TypeError, ValueError):
        return error_response(ErrorCode.MALFORMED_REQUEST, "limit must be an integer")

    rows = (
        AnalysisRecord.query.filter_by(requested_by=user_id)
        .order_by(AnalysisRecord.id.desc())
        .limit(limit)
        .all()
    )
    return {"items": [r.to_history_item() for r in rows]}


def _record(*, name, request_id, verdict, latency_ms, injection_flagged):
    """Write-through to the audit log. A failure here is logged, not raised —
    a persistence outage must not break analysis."""
    if not current_app.config.get("PERSIST_ANALYSES", True):
        return None
    try:
        row = AnalysisRecord(
            requested_by=g.actor.sub,
            request_id=request_id,
            player_name=name,
            model=MODEL,
            risk_score=verdict.risk_score,
            verdict=verdict.verdict,
            reason=verdict.reason,
            latency_ms=latency_ms,
            injection_flagged=injection_flagged,
        )
        db.session.add(row)
        db.session.commit()
        return row.public_id
    except Exception:
        db.session.rollback()
        log.exception("failed to persist analysis record")
        return None
