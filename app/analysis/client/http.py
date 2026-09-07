"""HTTP analysis client — calls the standalone analysis service.

Translates the wire contract (analysis_core.contract) and every transport /
status failure into the client error taxonomy, so the route layer handles both
modes with the same ``except`` clauses.
"""

import requests

from analysis_core.contract import (
    ANALYZE_PATH,
    AnalyzeRequest,
    AnalyzeResponse,
    ContractError,
    DEFAULT_TOKEN_TTL_SECONDS,
    ErrorCode,
    ErrorResponse,
    bearer_header,
    sign_actor,
)
from analysis_core.models import PlayerVerdict

from .base import (
    BUSY_MESSAGE,
    Actor,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisServiceDown,
    AnalysisUnavailable,
)


class HttpAnalysisClient:
    def __init__(
        self,
        *,
        base_url: str,
        token_secret: str,
        connect_timeout: float = 3.0,
        read_timeout: float = 90.0,
        token_ttl: int = DEFAULT_TOKEN_TTL_SECONDS,
        session=None,
    ):
        self._base_url = (base_url or "").rstrip("/")
        self._token_secret = token_secret or ""
        self._timeout = (connect_timeout, read_timeout)
        self._token_ttl = token_ttl
        # A pooled session keeps the TCP connection warm across requests.
        self._session = session if session is not None else requests.Session()

    def analyze(self, *, player_name: str, actor: Actor) -> PlayerVerdict:
        if not self._base_url:
            raise AnalysisNotConfigured("ANALYSIS_SERVICE_URL is not set")
        if not self._token_secret:
            raise AnalysisNotConfigured("ANALYSIS_TOKEN_SECRET is not set")

        token = sign_actor(
            user_id=actor.user_id,
            plan=actor.plan,
            secret=self._token_secret,
            ttl_seconds=self._token_ttl,
        )
        try:
            resp = self._session.post(
                f"{self._base_url}{ANALYZE_PATH}",
                json=AnalyzeRequest(player_name=player_name).to_wire(),
                headers={"Authorization": bearer_header(token)},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise AnalysisServiceDown(BUSY_MESSAGE) from e

        return self._interpret(resp)

    @staticmethod
    def _interpret(resp) -> PlayerVerdict:
        status = getattr(resp, "status_code", 0)

        if status == 200:
            try:
                return AnalyzeResponse.from_wire(resp.json()).to_verdict()
            except (ValueError, ContractError) as e:
                raise AnalysisUnavailable(
                    "analysis service returned an unreadable response"
                ) from e

        code = None
        detail = f"HTTP {status}"
        try:
            err = ErrorResponse.from_wire(resp.json())
            code, detail = err.error, (err.detail or err.error)
        except (ValueError, ContractError):
            pass

        if status == 400 or code == ErrorCode.INVALID_PLAYER_NAME:
            raise AnalysisBadRequest(detail)
        if status == 401:
            raise AnalysisNotConfigured(
                "analysis service rejected our token (check ANALYSIS_TOKEN_SECRET)"
            )
        if status == 429 or code == ErrorCode.UPSTREAM_RATE_LIMITED:
            raise AnalysisRateLimited(BUSY_MESSAGE)
        if status == 502 or code == ErrorCode.UPSTREAM_ERROR:
            raise AnalysisUnavailable(detail)
        if status == 503 or status >= 500:
            raise AnalysisServiceDown(BUSY_MESSAGE)
        # Anything else (e.g. 422 = we built a bad request) is our problem, not
        # something the user can retry into.
        raise AnalysisUnavailable(detail)
