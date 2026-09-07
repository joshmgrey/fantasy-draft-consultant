from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerVerdict:
    player: str
    risk_score: Optional[int]
    verdict: Optional[str]
    reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "reason": self.reason,
        }
