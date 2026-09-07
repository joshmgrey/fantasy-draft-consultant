"""The analysis-client contract: protocol, error taxonomy, caller identity,
and the factory that selects an implementation.

Nothing here imports Flask or ``analysis_core`` — implementations do. Keeping
this module dependency-light lets both the route layer and the tests depend on
the vocabulary without pulling in a backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from analysis_core.contract import DEFAULT_TOKEN_TTL_SECONDS
from analysis_core.models import PlayerVerdict

# Shown to the end user whenever the analysis path is temporarily unusable
# (upstream rate limit, or the analysis service itself is down). Kept here so
# both client implementations and the route agree on the wording.
BUSY_MESSAGE = "The analysis service is busy right now. Please try again in a minute."


@dataclass(frozen=True)
class Actor:
    """Who an analysis is being run for.

    Carried across the boundary so the analysis side can attribute / log a
    request without importing the identity context. The analysis side never
    makes authorization decisions from this — quota and plan enforcement stay
    in the core app.
    """

    user_id: str
    plan: str = "free"

    @classmethod
    def from_user(cls, user) -> "Actor":
        return cls(user_id=str(user.get_id()), plan=getattr(user, "plan", "free"))


# --------------------------------------------------------------------------- #
# Error taxonomy
#
# Implementations translate their backend-specific failures (an ``anthropic``
# exception in-process, an HTTP status over the network) into these, so the
# route layer maps exceptions to responses exactly once and identically for
# both modes.
# --------------------------------------------------------------------------- #
class AnalysisError(Exception):
    """Base class for every analysis-client failure."""


class AnalysisBadRequest(AnalysisError):
    """The request was rejected as invalid (e.g. bad player name). -> HTTP 400."""


class AnalysisRateLimited(AnalysisError):
    """An upstream rate limit was hit and could not be recovered. -> HTTP 503."""


class AnalysisUnavailable(AnalysisError):
    """The analysis backend (Anthropic) errored. -> HTTP 502."""


class AnalysisServiceDown(AnalysisError):
    """The analysis service itself is unreachable or failing (transport error,
    5xx, or it reported it cannot serve). Retryable. -> HTTP 503.

    Only the HTTP client raises this; in-process mode has no such failure."""


class AnalysisNotConfigured(AnalysisError):
    """The analysis backend is not configured (e.g. no API key, or the shared
    token secret / service URL is missing). -> HTTP 500."""


@runtime_checkable
class AnalysisClient(Protocol):
    def analyze(self, *, player_name: str, actor: Actor) -> PlayerVerdict:
        """Analyze ``player_name`` on behalf of ``actor``.

        Returns a :class:`~analysis_core.models.PlayerVerdict` (fields may be
        ``None`` if the model produced no structured verdict). Raises an
        :class:`AnalysisError` subclass on failure.
        """


def build_analysis_client(config) -> AnalysisClient:
    """Construct the client implementation named by ``config['ANALYSIS_MODE']``.

    ``config`` is anything with a ``.get`` (a Flask ``app.config`` or a plain
    dict). Defaults to ``"inprocess"`` — the monolith's behavior.
    """
    mode = (config.get("ANALYSIS_MODE") or "inprocess").strip().lower()
    if mode == "inprocess":
        from .inprocess import InProcessAnalysisClient

        return InProcessAnalysisClient()
    if mode == "http":
        from .http import HttpAnalysisClient

        return HttpAnalysisClient(
            base_url=config.get("ANALYSIS_SERVICE_URL", ""),
            token_secret=config.get("ANALYSIS_TOKEN_SECRET", ""),
            connect_timeout=float(config.get("ANALYSIS_CONNECT_TIMEOUT", 3.0)),
            read_timeout=float(config.get("ANALYSIS_READ_TIMEOUT", 90.0)),
            token_ttl=int(config.get("ANALYSIS_TOKEN_TTL", DEFAULT_TOKEN_TTL_SECONDS)),
        )
    raise RuntimeError(
        f"Unknown ANALYSIS_MODE {mode!r} (expected 'inprocess' or 'http')"
    )
