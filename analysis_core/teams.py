"""NFL teams, and recognizing a team defense (D/ST) in a typed name.

Shared by D/ST training (``ml/dst_features.py``) and serving
(``analysis_core/projection.py``), so pure Python: no pandas.
"""

from __future__ import annotations

import re
from typing import Optional

# nflverse abbreviation -> (city, nickname). nflverse uses LA for the Rams.
TEAMS = {
    "ARI": ("Arizona", "Cardinals"),
    "ATL": ("Atlanta", "Falcons"),
    "BAL": ("Baltimore", "Ravens"),
    "BUF": ("Buffalo", "Bills"),
    "CAR": ("Carolina", "Panthers"),
    "CHI": ("Chicago", "Bears"),
    "CIN": ("Cincinnati", "Bengals"),
    "CLE": ("Cleveland", "Browns"),
    "DAL": ("Dallas", "Cowboys"),
    "DEN": ("Denver", "Broncos"),
    "DET": ("Detroit", "Lions"),
    "GB": ("Green Bay", "Packers"),
    "HOU": ("Houston", "Texans"),
    "IND": ("Indianapolis", "Colts"),
    "JAX": ("Jacksonville", "Jaguars"),
    "KC": ("Kansas City", "Chiefs"),
    "LA": ("Los Angeles", "Rams"),
    "LAC": ("Los Angeles", "Chargers"),
    "LV": ("Las Vegas", "Raiders"),
    "MIA": ("Miami", "Dolphins"),
    "MIN": ("Minnesota", "Vikings"),
    "NE": ("New England", "Patriots"),
    "NO": ("New Orleans", "Saints"),
    "NYG": ("New York", "Giants"),
    "NYJ": ("New York", "Jets"),
    "PHI": ("Philadelphia", "Eagles"),
    "PIT": ("Pittsburgh", "Steelers"),
    "SEA": ("Seattle", "Seahawks"),
    "SF": ("San Francisco", "49ers"),
    "TB": ("Tampa Bay", "Buccaneers"),
    "TEN": ("Tennessee", "Titans"),
    "WAS": ("Washington", "Commanders"),
}

# Older abbreviations still found in historical nflverse files, mapped to the
# franchise's current one so a team's seasons line up across a move.
FRANCHISE_MOVES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Common alternate abbreviations people type.
_ABBR_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "SFO": "SF", "GNB": "GB",
                 "KAN": "KC", "NWE": "NE", "NOR": "NO", "TAM": "TB"}

# Old nicknames people still type.
_NICKNAME_ALIASES = {
    "redskins": "WAS", "washington redskins": "WAS", "football team": "WAS",
    "washington football team": "WAS", "oakland raiders": "LV", "san diego chargers": "LAC",
    "st louis rams": "LA", "la rams": "LA", "la chargers": "LAC", "ny giants": "NYG",
    "ny jets": "NYJ", "niners": "SF", "bucs": "TB", "pats": "NE", "jags": "JAX",
}

# Words that mark the input as a team defense rather than a player.
_DST_MARKER_RE = re.compile(
    r"\b(d\s*/\s*st|dst|d\s*st|def|defen[cs]e|special teams|d\s*&\s*st)\b"
)


def current_abbr(abbr: str) -> str:
    return FRANCHISE_MOVES.get(abbr, abbr)


def dst_display_name(abbr: str) -> str:
    city, nickname = TEAMS[abbr]
    return f"{city} {nickname} D/ST"


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9/& ]+", " ", text.lower()).split())


def _lookup_tables():
    by_phrase = {}
    city_count = {}
    for abbr, (city, nickname) in TEAMS.items():
        city_count[city.lower()] = city_count.get(city.lower(), 0) + 1
    for abbr, (city, nickname) in TEAMS.items():
        by_phrase[f"{city} {nickname}".lower()] = abbr
        by_phrase[nickname.lower()] = abbr
        # A shared city ("New York", "Los Angeles") can't name one team.
        if city_count[city.lower()] == 1:
            by_phrase[city.lower()] = abbr
    by_phrase.update(_NICKNAME_ALIASES)
    abbrs = {a.lower(): a for a in TEAMS}
    abbrs.update({k.lower(): v for k, v in _ABBR_ALIASES.items()})
    abbrs.update({k.lower(): v for k, v in FRANCHISE_MOVES.items()})
    return by_phrase, abbrs


_BY_PHRASE, _BY_ABBR = _lookup_tables()


def resolve_dst(name: str) -> Optional[str]:
    """The team abbreviation if ``name`` names a team defense, else ``None``.

    Accepts "Steelers D/ST", "Pittsburgh Steelers", "PIT DST", "49ers Defense",
    "Bills DEF". A bare team name ("Steelers") counts; a bare abbreviation
    ("PIT") only counts with a D/ST marker, since short strings are too easy to
    confuse with something else.
    """
    text = _clean(name)
    has_marker = bool(_DST_MARKER_RE.search(text))
    rest = " ".join(_DST_MARKER_RE.sub(" ", text).replace("/", " ").replace("&", " ").split())
    if not rest:
        return None
    if rest in _BY_PHRASE:
        return _BY_PHRASE[rest]
    if has_marker and rest in _BY_ABBR:
        return _BY_ABBR[rest]
    return None
