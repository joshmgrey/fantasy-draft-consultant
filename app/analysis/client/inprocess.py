"""In-process analysis client — runs the analysis in the current process.

This is the default and reproduces the pre-split monolith exactly: it calls
``analysis_core`` directly and translates its failures into the client error
taxonomy. The mapping mirrors the error handling that used to live inline in
the ``/analyze`` route.
"""

import anthropic

from analysis_core import anthropic_client as _core
from analysis_core.models import PlayerVerdict
from analysis_core.services import parse_verdict, validate_player_name

from .base import (
    BUSY_MESSAGE,
    Actor,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisUnavailable,
)


class InProcessAnalysisClient:
    def analyze(self, *, player_name: str, actor: Actor) -> PlayerVerdict:
        if _core.client is None:
            raise AnalysisNotConfigured(
                "ANTHROPIC_API_KEY environment variable is not set."
            )

        try:
            name = validate_player_name(player_name)
        except ValueError as e:
            raise AnalysisBadRequest(str(e)) from e

        try:
            raw_text = _core.analyze_player(name)
        except anthropic.RateLimitError as e:
            # RateLimitError is a subclass of APIStatusError — catch it first.
            raise AnalysisRateLimited(BUSY_MESSAGE) from e
        except anthropic.APIStatusError as e:
            raise AnalysisUnavailable(
                f"API error ({e.status_code}): {e.message}"
            ) from e
        except anthropic.APIConnectionError as e:
            raise AnalysisUnavailable(
                "Could not reach Anthropic API. Check your connection."
            ) from e

        return parse_verdict(name, raw_text)
