"""The wire contract between the core app and the standalone analysis service.

Imported by BOTH sides so the request/response shapes and the actor-token
scheme cannot drift. Pure stdlib — no Flask, no requests, no anthropic.

HTTP surface (v1)::

    POST /v1/analyses
        Authorization: Bearer <actor token>     # sign_actor / verify_actor
        body:  AnalyzeRequest.to_wire()
      200  AnalyzeResponse.to_wire()
      4xx/5xx  ErrorResponse.to_wire()

    GET /healthz
      200  {"status": "ok", "anthropic_configured": <bool>}

The actor token is how "who is asking" crosses the boundary: the core app
signs a short-lived token naming the user; the analysis service verifies it
with the shared secret and never touches the identity context or its tables.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from analysis_core.models import PlayerVerdict

__all__ = [
    "API_VERSION",
    "ANALYZE_PATH",
    "HEALTH_PATH",
    "DEFAULT_TOKEN_TTL_SECONDS",
    "NOTICE_WITHHELD",
    "ContractError",
    "ErrorCode",
    "ERROR_STATUS",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ErrorResponse",
    "ActorToken",
    "TokenError",
    "TokenInvalid",
    "TokenExpired",
    "sign_actor",
    "verify_actor",
    "bearer_header",
    "parse_bearer",
]

API_VERSION = "v1"
ANALYZE_PATH = f"/{API_VERSION}/analyses"
HEALTH_PATH = "/healthz"

DEFAULT_TOKEN_TTL_SECONDS = 120

# ``notice`` value the service sets when analyze_player's leak-guard fired and
# there is no real verdict to return (the request still succeeds with 200 and
# null verdict fields, matching the pre-split behavior).
NOTICE_WITHHELD = "analysis_withheld_suspected_injection"


class ContractError(ValueError):
    """A payload did not match the wire contract (-> ErrorCode.MALFORMED_REQUEST)."""


class ErrorCode:
    INVALID_PLAYER_NAME = "invalid_player_name"
    UNAUTHORIZED = "unauthorized"
    MALFORMED_REQUEST = "malformed_request"
    UPSTREAM_RATE_LIMITED = "upstream_rate_limited"
    UPSTREAM_ERROR = "upstream_error"
    UNAVAILABLE = "unavailable"


ERROR_STATUS = {
    ErrorCode.INVALID_PLAYER_NAME: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.MALFORMED_REQUEST: 422,
    ErrorCode.UPSTREAM_RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_ERROR: 502,
    ErrorCode.UNAVAILABLE: 503,
}


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ContractError(message)


def _opt_str(value: Any, field: str) -> Optional[str]:
    _require(value is None or isinstance(value, str), f"{field} must be a string or null")
    return value


# --------------------------------------------------------------------------- #
# Request / response bodies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalyzeRequest:
    player_name: str
    request_id: Optional[str] = None

    def to_wire(self) -> dict:
        return {"player_name": self.player_name, "request_id": self.request_id}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AnalyzeRequest":
        _require(isinstance(data, Mapping), "body must be a JSON object")
        name = data.get("player_name")
        # Only a type check here. Format rules live in validate_player_name and
        # produce ErrorCode.INVALID_PLAYER_NAME, not MALFORMED_REQUEST.
        _require(isinstance(name, str), "player_name must be a string")
        return cls(player_name=name, request_id=_opt_str(data.get("request_id"), "request_id"))


@dataclass(frozen=True)
class AnalyzeResponse:
    player: str
    risk_score: Optional[int] = None
    verdict: Optional[str] = None
    reason: Optional[str] = None
    model: Optional[str] = None
    notice: Optional[str] = None
    analysis_id: Optional[str] = None

    def to_wire(self) -> dict:
        return {
            "player": self.player,
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "reason": self.reason,
            "model": self.model,
            "notice": self.notice,
            "analysis_id": self.analysis_id,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "AnalyzeResponse":
        _require(isinstance(data, Mapping), "body must be a JSON object")
        player = data.get("player")
        _require(isinstance(player, str), "player must be a string")
        risk = data.get("risk_score")
        _require(
            risk is None or (isinstance(risk, int) and not isinstance(risk, bool)),
            "risk_score must be an integer or null",
        )
        return cls(
            player=player,
            risk_score=risk,
            verdict=_opt_str(data.get("verdict"), "verdict"),
            reason=_opt_str(data.get("reason"), "reason"),
            model=_opt_str(data.get("model"), "model"),
            notice=_opt_str(data.get("notice"), "notice"),
            analysis_id=_opt_str(data.get("analysis_id"), "analysis_id"),
        )

    @classmethod
    def from_verdict(
        cls,
        verdict: PlayerVerdict,
        *,
        model: Optional[str] = None,
        notice: Optional[str] = None,
        analysis_id: Optional[str] = None,
    ) -> "AnalyzeResponse":
        return cls(
            player=verdict.player,
            risk_score=verdict.risk_score,
            verdict=verdict.verdict,
            reason=verdict.reason,
            model=model,
            notice=notice,
            analysis_id=analysis_id,
        )

    def to_verdict(self) -> PlayerVerdict:
        return PlayerVerdict(
            player=self.player,
            risk_score=self.risk_score,
            verdict=self.verdict,
            reason=self.reason,
        )


@dataclass(frozen=True)
class ErrorResponse:
    error: str
    detail: str = ""
    request_id: Optional[str] = None

    @property
    def status_code(self) -> int:
        return ERROR_STATUS.get(self.error, 500)

    def to_wire(self) -> dict:
        return {"error": self.error, "detail": self.detail, "request_id": self.request_id}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ErrorResponse":
        _require(isinstance(data, Mapping), "body must be a JSON object")
        code = data.get("error")
        _require(isinstance(code, str) and bool(code), "error must be a non-empty string")
        detail = data.get("detail", "")
        _require(isinstance(detail, str), "detail must be a string")
        return cls(
            error=code,
            detail=detail,
            request_id=_opt_str(data.get("request_id"), "request_id"),
        )


# --------------------------------------------------------------------------- #
# Actor token: a short-lived HMAC-signed statement of who is asking
# --------------------------------------------------------------------------- #
class TokenError(Exception):
    """Base class for actor-token failures (-> ErrorCode.UNAUTHORIZED)."""


class TokenInvalid(TokenError):
    """Structure or signature is wrong."""


class TokenExpired(TokenError):
    """Token is past its expiry."""


@dataclass(frozen=True)
class ActorToken:
    sub: str          # user id, as the core app sees it
    plan: str
    iat: int
    exp: int


_BEARER_PREFIX = "Bearer "


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(body: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return _b64u_encode(digest)


def sign_actor(
    *,
    user_id: str,
    plan: str,
    secret: str,
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
    now: Optional[float] = None,
) -> str:
    """Return a ``<payload>.<signature>`` token naming the actor."""
    if not secret:
        raise TokenInvalid("no signing secret configured")
    issued = int(now if now is not None else time.time())
    payload = {
        "sub": str(user_id),
        "plan": str(plan),
        "iat": issued,
        "exp": issued + int(ttl_seconds),
    }
    body = _b64u_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{body}.{_sign(body, secret)}"


def verify_actor(
    token: str,
    *,
    secret: str,
    now: Optional[float] = None,
    leeway_seconds: int = 0,
) -> ActorToken:
    """Verify signature and expiry, returning the :class:`ActorToken`.

    Raises :class:`TokenInvalid` for a bad structure/signature and
    :class:`TokenExpired` once ``exp`` (plus ``leeway_seconds``) has passed.
    """
    if not secret:
        raise TokenInvalid("no signing secret configured")
    if not isinstance(token, str) or token.count(".") != 1:
        raise TokenInvalid("malformed token")
    body, signature = token.split(".")
    if not hmac.compare_digest(signature, _sign(body, secret)):
        raise TokenInvalid("bad signature")
    try:
        payload = json.loads(_b64u_decode(body))
    except Exception as e:  # any decode failure means the token is not ours
        raise TokenInvalid("undecodable payload") from e
    try:
        actor = ActorToken(
            sub=str(payload["sub"]),
            plan=str(payload["plan"]),
            iat=int(payload["iat"]),
            exp=int(payload["exp"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise TokenInvalid("missing or malformed claims") from e
    current = int(now if now is not None else time.time())
    if current > actor.exp + int(leeway_seconds):
        raise TokenExpired("token expired")
    return actor


def bearer_header(token: str) -> str:
    return f"{_BEARER_PREFIX}{token}"


def parse_bearer(header: Optional[str]) -> str:
    """Pull the token out of an ``Authorization`` header value."""
    if not isinstance(header, str) or not header.startswith(_BEARER_PREFIX):
        raise TokenInvalid("missing bearer token")
    return header[len(_BEARER_PREFIX):].strip()
