"""Serve the trained next-season projection models.

Three models, each trained offline (see ``ml/``) and exported as one joblib
artifact holding the fitted scikit-learn estimator, the feature order, and the
latest season's feature rows:

* PPR points per game for QB/RB/WR/TE (``python -m ml.train``), keyed by
  normalized player name
* D/ST fantasy points per game (``python -m ml.train_dst``), keyed by team
  abbreviation; a typed name like "Steelers D/ST" is routed here by
  :func:`analysis_core.teams.resolve_dst`
* kicker fantasy points per game (``python -m ml.train_kicker``), keyed by
  normalized name; asked only when the player model doesn't know the name

Serving is a dictionary lookup plus one ``predict`` call — no pandas, no network.

The projection is optional by design. If the artifact is missing, was built
with a different scikit-learn version, or has no row for the player (a rookie,
or someone with too few games), :meth:`Projector.project` returns ``None`` and
the analysis runs exactly as it did before the model existed. The same holds
for the D/ST and kicker artifacts.
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

from .season import current_season
from .teams import resolve_dst

log = logging.getLogger(__name__)

ARTIFACT_FORMAT = "ppr-ppg-projection/1"
DEFAULT_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "ppr_projection.joblib"
ARTIFACT_ENV = "PROJECTION_MODEL_PATH"

DST_ARTIFACT_FORMAT = "dst-ppg-projection/1"
DEFAULT_DST_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "dst_projection.joblib"
DST_ARTIFACT_ENV = "DST_PROJECTION_MODEL_PATH"
DST_POSITION = "D/ST"

KICKER_ARTIFACT_FORMAT = "kicker-ppg-projection/1"
DEFAULT_KICKER_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "kicker_projection.joblib"
KICKER_ARTIFACT_ENV = "KICKER_PROJECTION_MODEL_PATH"
KICKER_POSITION = "K"

# What prompt_block calls each model's points, and what it can't see.
_UNITS = {
    DST_POSITION: (
        "standard D/ST fantasy points",
        "The model only sees past team stats and next season's schedule: it does not "
        "know about free agency, the draft, coaching changes, or injuries, so weigh it "
        "against current news.",
    ),
    KICKER_POSITION: (
        "standard kicker fantasy points",
        "The model only sees past kicking stats and the team's offense and next-season "
        "schedule: it does not know about signings, cuts, or a new team, so weigh it "
        "against current news.",
    ),
}
_PPR_UNIT = (
    "PPR points",
    "The model only sees past stats: it does not know about offseason moves, "
    "depth-chart changes, or injuries, so weigh it against current news.",
)

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
    last_ppg: float        # points per game in the feature season
    last_games: int
    typical_error: float   # test-season MAE, in points per game

    @property
    def is_dst(self) -> bool:
        return self.position == DST_POSITION

    @property
    def scoring(self) -> str:
        """The scoring ``projected_ppg`` is in: "D/ST", "K", or "PPR"."""
        return self.position if self.position in _UNITS else "PPR"

    def prompt_block(self) -> str:
        """Trusted context for the LLM. Built from our own model output and
        stats fields, never from user input or retrieved web content."""
        unit, caveat = _UNITS.get(self.position, _PPR_UNIT)
        return (
            '<model_projection trust_level="trusted">\n'
            f"In-house statistical model (scikit-learn, trained on past NFL box scores): "
            f"{self.player} ({self.position}, {self.team}) projects to "
            f"{self.projected_ppg:.1f} {unit} per game in {self.season}, versus "
            f"{self.last_ppg:.1f} over {self.last_games} games in {self.season - 1}. "
            f"Typical error is about ±{self.typical_error:.1f} points per game. "
            f"{caveat}\n"
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
    def load(cls, path: Path, fmt: str = ARTIFACT_FORMAT) -> Optional["Projector"]:
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
        if not isinstance(artifact, dict) or artifact.get("format") != fmt:
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
        return self.project_key(normalize_name(player_name))

    def project_key(self, key: str) -> Optional[Projection]:
        """Projection for an exact artifact key (a normalized player name, or
        a team abbreviation in the D/ST artifact)."""
        row = self._players.get(key)
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
# kind -> loaded Projector, or None when it couldn't be loaded.
_projectors: dict = {}
_ARTIFACTS = {
    "player": (ARTIFACT_ENV, DEFAULT_ARTIFACT_PATH, ARTIFACT_FORMAT),
    "dst": (DST_ARTIFACT_ENV, DEFAULT_DST_ARTIFACT_PATH, DST_ARTIFACT_FORMAT),
    "kicker": (KICKER_ARTIFACT_ENV, DEFAULT_KICKER_ARTIFACT_PATH, KICKER_ARTIFACT_FORMAT),
}
_TRAIN_COMMANDS = {"player": "ml.train", "dst": "ml.train_dst", "kicker": "ml.train_kicker"}


def _get(kind: str) -> Optional[Projector]:
    if kind not in _projectors:
        with _lock:
            if kind not in _projectors:
                env, default, fmt = _ARTIFACTS[kind]
                path = Path(os.environ.get(env) or default)
                _projectors[kind] = Projector.load(path, fmt)
    return _projectors[kind]


def get_projector() -> Optional[Projector]:
    """Process-wide player projector, loaded once on first use."""
    return _get("player")


def get_dst_projector() -> Optional[Projector]:
    """Process-wide D/ST projector, loaded once on first use."""
    return _get("dst")


def get_kicker_projector() -> Optional[Projector]:
    """Process-wide kicker projector, loaded once on first use."""
    return _get("kicker")


_stale_warned: set = set()


def _current(kind: str) -> Optional[Projector]:
    """The ``kind`` projector, or ``None`` if it is missing or was built for
    an earlier season (warned once per kind and season)."""
    projector = _get(kind)
    if projector is None:
        return None
    season = current_season()
    if projector.season != season:
        if (kind, projector.season) not in _stale_warned:
            _stale_warned.add((kind, projector.season))
            log.warning(
                "%s projection model targets the %s season but it is now %s; "
                "retrain with `python -m ml.download && python -m %s`. "
                "Projections disabled.",
                kind, projector.season, season, _TRAIN_COMMANDS[kind],
            )
        return None
    return projector


def project_player(player_name: str) -> Optional[Projection]:
    """Projection for ``player_name``, or ``None``. Never raises: the model is
    an enhancement, and a failure here must not break the analysis.

    A team defense ("Steelers D/ST", "49ers Defense") is projected by the D/ST
    model; anything else by the player model, then by the kicker model if the
    player model doesn't know the name.

    Only serves an artifact built for the *current* fantasy season. Checked on
    every call rather than at load time, so a process that is running when the
    season rolls over (May 1) stops serving last year's projections too.
    """
    try:
        team = resolve_dst(player_name)
        if team:
            dst = _current("dst")
            return dst.project_key(team) if dst else None
        for kind in ("player", "kicker"):
            projector = _current(kind)
            found = projector.project(player_name) if projector else None
            if found is not None:
                return found
        return None
    except Exception:
        log.exception("projection failed for %r", player_name)
        return None


def reset_projector() -> None:
    """Forget the cached projector (tests, or after retraining)."""
    with _lock:
        _projectors.clear()
        _stale_warned.clear()
