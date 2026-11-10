"""Serve the trained next-season PPR points-per-game model.

The model is trained offline by ``python -m ml.train`` (see ``ml/``) and
exported as one joblib artifact holding the fitted scikit-learn estimator, the
feature order, and the latest season's feature rows for every player. Serving
is a dictionary lookup plus one ``predict`` call — no pandas, no network.

The projection is optional by design. If the artifact is missing, was built
with a different scikit-learn version, or has no row for the player (a rookie,
or someone with too few games), :meth:`Projector.project` returns ``None`` and
the analysis runs exactly as it did before the model existed.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ARTIFACT_FORMAT = "ppr-ppg-projection/1"
DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "ppr_projection.joblib"
ARTIFACT_ENV = "PROJECTION_MODEL_PATH"

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NON_ALPHA_RE = re.compile(r"[^a-z ]+")


def normalize_name(name: str) -> str:
    """Name key shared by training and serving.

    ``"Ja'Marr Chase"`` -> ``"jamarr chase"``, ``"Kenneth Walker III"`` ->
    ``"kenneth walker"``, ``"Amon-Ra St. Brown"`` -> ``"amonra st brown"``.
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.lower().replace("-", "").replace("'", "").replace(".", "")
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _SUFFIX_RE.sub(" ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class Projection:
    player: str            # display name from the stats
    position: str
    team: str
    season: int            # the season being projected
    projected_ppg: float
    last_ppg: float        # PPR points per game in the feature season
    last_games: int
    typical_error: float   # test-season MAE, in PPR points per game

    def prompt_block(self) -> str:
        """Trusted context for the LLM. Built from our own model output and
        stats fields, never from user input or retrieved web content."""
        return (
            '<model_projection trust_level="trusted">\n'
            f"In-house statistical model (scikit-learn, trained on past NFL box scores): "
            f"{self.player} ({self.position}, {self.team}) projects to "
            f"{self.projected_ppg:.1f} PPR points per game in {self.season}, versus "
            f"{self.last_ppg:.1f} over {self.last_games} games in {self.season - 1}. "
            f"Typical error is about ±{self.typical_error:.1f} points per game. "
            "The model only sees past stats: it does not know about offseason moves, "
            "depth-chart changes, or injuries, so weigh it against current news.\n"
            "</model_projection>"
        )


class Projector:
    def __init__(self, artifact: dict):
        self._model = artifact["model"]
        self._players = artifact["players"]
        self.season = int(artifact["target_season"])
        self.typical_error = float(artifact["test_mae"])
        self.model_name = artifact.get("model_name", "")

    @classmethod
    def load(cls, path: Path) -> Optional["Projector"]:
        """Load an artifact, or return ``None`` (logged) if it can't be used."""
        path = Path(path)
        if not path.exists():
            log.info("projection model not found at %s; projections disabled", path)
            return None
        try:
            import joblib
            import sklearn

            artifact = joblib.load(path)
        except Exception:
            log.exception("could not load projection model from %s", path)
            return None
        if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
            log.warning("projection model at %s has an unknown format; ignoring it", path)
            return None
        if artifact.get("sklearn_version") != sklearn.__version__:
            # A pickled estimator is only guaranteed to behave the same under
            # the scikit-learn version that wrote it.
            log.warning(
                "projection model was built with scikit-learn %s but %s is installed; "
                "retrain with `python -m ml.train`. Projections disabled.",
                artifact.get("sklearn_version"), sklearn.__version__,
            )
            return None
        return cls(artifact)

    def project(self, player_name: str) -> Optional[Projection]:
        row = self._players.get(normalize_name(player_name))
        if row is None:
            return None
        import numpy as np

        vec = np.array(
            [[np.nan if v is None else v for v in row["features"]]], dtype=float
        )
        predicted = float(self._model.predict(vec)[0])
        return Projection(
            player=row["name"],
            position=row["position"],
            team=row["team"],
            season=self.season,
            projected_ppg=round(max(predicted, 0.0), 1),
            last_ppg=float(row["ppg"]),
            last_games=int(row["games"]),
            typical_error=self.typical_error,
        )


_lock = threading.Lock()
_loaded = False
_projector: Optional[Projector] = None


def get_projector() -> Optional[Projector]:
    """Process-wide projector, loaded once on first use."""
    global _loaded, _projector
    if not _loaded:
        with _lock:
            if not _loaded:
                path = Path(os.environ.get(ARTIFACT_ENV) or DEFAULT_ARTIFACT_PATH)
                _projector = Projector.load(path)
                _loaded = True
    return _projector


def project_player(player_name: str) -> Optional[Projection]:
    """Projection for ``player_name``, or ``None``. Never raises: the model is
    an enhancement, and a failure here must not break the analysis."""
    try:
        projector = get_projector()
        return projector.project(player_name) if projector else None
    except Exception:
        log.exception("projection failed for %r", player_name)
        return None


def reset_projector() -> None:
    """Forget the cached projector (tests, or after retraining)."""
    global _loaded, _projector
    with _lock:
        _loaded = False
        _projector = None
