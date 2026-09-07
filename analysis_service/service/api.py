import logging

import anthropic
from flask import Blueprint, g

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
from .schemas import analyze_response, error_response, parse_analyze_request

log = logging.getLogger("analysis_service")

bp = Blueprint("analysis_service", __name__)

# The model id analyze_player asks for. Surfaced in the response so the core app
# and its clients can record which model produced a verdict.
MODEL = "claude-opus-4-7"


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

    verdict = parse_verdict(name, raw_text)
    notice = (
        NOTICE_WITHHELD
        if raw_text == anthropic_client.WITHHELD_MESSAGE
        else None
    )
    return analyze_response(
        AnalyzeResponse.from_verdict(verdict, model=MODEL, notice=notice)
    )
