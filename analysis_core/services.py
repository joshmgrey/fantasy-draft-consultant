import re
from .models import PlayerVerdict

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z '\-\.]{0,48}$")
_MAX_NAME_LEN = 50
_TAG_RE = re.compile(r"<[^>]+>")


def validate_player_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Player name cannot be empty.")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"Player name too long (max {_MAX_NAME_LEN} characters).")
    if not _NAME_RE.match(name):
        raise ValueError(
            "Player name contains invalid characters. "
            "Use letters, spaces, apostrophes, hyphens, or periods only."
        )
    return name


def parse_verdict(player_name: str, text: str) -> PlayerVerdict:
    result = PlayerVerdict(player=player_name, risk_score=None, verdict=None, reason=None)

    for line in text.strip().splitlines():
        line = _TAG_RE.sub("", line).strip()
        if line.upper().startswith("RISK:"):
            val = line.split(":", 1)[1].strip().split("/")[0].strip()
            try:
                result.risk_score = int(val)
            except ValueError:
                pass
        elif line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip()
            result.verdict = "Draft" if "draft" in val.lower() else "Pass"
        elif line.upper().startswith("REASON:"):
            result.reason = line.split(":", 1)[1].strip()

    return result
