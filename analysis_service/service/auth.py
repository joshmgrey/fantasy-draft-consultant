"""Service-to-service auth: verify the HMAC actor token minted by the core app.

Deliberately imports nothing from the identity context. The token *is* the
identity assertion; the shared secret is the trust. All this module learns is
``{sub, plan}`` — it makes no authorization decision from them.
"""

from functools import wraps

from flask import current_app, g, request

from analysis_core.contract import (
    ErrorCode,
    TokenError,
    parse_bearer,
    verify_actor,
)

from .schemas import error_response


def require_actor(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        secret = current_app.config.get("ANALYSIS_TOKEN_SECRET")
        if not secret:
            # The service is misconfigured — not the caller's fault, so this is
            # a 503, not a 401.
            return error_response(
                ErrorCode.UNAVAILABLE, "actor-token secret is not configured"
            )
        try:
            token = parse_bearer(request.headers.get("Authorization"))
            g.actor = verify_actor(
                token,
                secret=secret,
                leeway_seconds=current_app.config.get("TOKEN_LEEWAY_SECONDS", 0),
            )
        except TokenError as e:
            return error_response(ErrorCode.UNAUTHORIZED, str(e))
        return view(*args, **kwargs)

    return wrapper
