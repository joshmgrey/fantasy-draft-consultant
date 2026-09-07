"""Client interface the core app uses to reach the analysis context.

The core app never calls ``analysis_core`` directly from a route. It goes
through an ``AnalysisClient``. Two implementations exist:

* ``InProcessAnalysisClient`` — runs the analysis in this process (the default;
  identical behavior to the pre-split monolith).
* an HTTP implementation (added in a later step) that calls the standalone
  analysis service over the network.

The active implementation is chosen by ``ANALYSIS_MODE`` and stored on
``app.extensions["analysis_client"]`` by the app factory.
"""

from .base import (
    BUSY_MESSAGE,
    Actor,
    AnalysisClient,
    AnalysisError,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisServiceDown,
    AnalysisUnavailable,
    build_analysis_client,
)

__all__ = [
    "BUSY_MESSAGE",
    "Actor",
    "AnalysisClient",
    "AnalysisError",
    "AnalysisBadRequest",
    "AnalysisNotConfigured",
    "AnalysisRateLimited",
    "AnalysisServiceDown",
    "AnalysisUnavailable",
    "build_analysis_client",
]
