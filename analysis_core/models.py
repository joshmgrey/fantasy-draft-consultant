from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerVerdict:
    player: str
    risk_score: Optional[int]
    verdict: Optional[str]
    reason: Optional[str]
    # The in-house model's points-per-game projection for the season, when it
    # has one for this player (see analysis_core/projection.py), and which
    # scoring it is in: "PPR" for a player, "D/ST" for a team defense.
    projected_ppg: Optional[float] = None
    projection_scoring: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "reason": self.reason,
            "projected_ppg": self.projected_ppg,
            "projection_scoring": self.projection_scoring,
        }
