"""The fantasy season calendar, shared by the core app and the analysis service.

A season runs from just after one NFL Draft to just after the next. The NFL
Draft ends in late April, so the season rolls over on ``ROLLOVER`` (May 1):
from then on, the new season's rookies are placed and fantasy draft prep is
for that year. Until the next rollover it is still that season, including
while games are being played.

    2026-04-30 -> 2025      2026-05-01 -> 2026      2026-10-12 -> 2026

Season passes expire at the next rollover, so a pass sold for a season covers
that season's fantasy drafts and the games that follow.
"""

from datetime import date
from typing import Optional

# (month, day) the season rolls over: just after the NFL Draft.
ROLLOVER = (5, 1)


def current_season(today: Optional[date] = None) -> int:
    today = today or date.today()
    return today.year if (today.month, today.day) >= ROLLOVER else today.year - 1


def season_expiry(today: Optional[date] = None) -> date:
    """When a pass sold ``today`` stops working: the next rollover."""
    return date(current_season(today) + 1, *ROLLOVER)
