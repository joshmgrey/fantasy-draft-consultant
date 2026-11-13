"""Feature engineering for the next-season D/ST fantasy points-per-game model.

One row per (team, season t). Features come only from season t, plus the
season t + 1 *schedule* (which opponents a defense will face is published in
May, before any draft, so it is fair game). The label is the team's D/ST
fantasy points per game in season t + 1. Nothing from a t + 1 game result may
leak into the features; ``tests/test_dst_features.py`` checks that.

Inputs are nflverse files (https://github.com/nflverse/nflverse-data):

* ``stats_team_week_<season>.csv`` — per-team, per-game box scores
* ``games.csv`` — every game's teams and final score (and future schedules)

D/ST fantasy points use common standard scoring (see :data:`SCORING` and
:func:`points_allowed_points`). Leagues differ — some also score yards
allowed or return yards — so treat the numbers as "standard D/ST".

Training-only module (pandas). Serving loads the trained artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis_core.teams import current_abbr

from .download import raw_file

SCORING = {
    "sacks": 1,
    "interceptions": 2,
    "fumble_recoveries": 2,
    "safeties": 2,
    "touchdowns": 6,   # defensive and special-teams return TDs
    "blocked_kicks": 2,
}

# (most points allowed, fantasy points), checked in order.
POINTS_ALLOWED_TIERS = ((0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1))
POINTS_ALLOWED_FLOOR = -4  # 35 or more

# Roof types that count as an indoor game (``games.csv`` ``roof`` column).
INDOOR_ROOFS = ("dome", "closed")

_WEEK_COLUMNS = [
    "season", "week", "team", "season_type", "opponent_team",
    "def_sacks", "def_qb_hits", "def_interceptions", "def_pass_defended",
    "def_fumbles_forced", "fumble_recovery_opp", "def_tds", "special_teams_tds",
    "def_safeties", "def_punt_blocks", "def_pat_blocks", "def_fg_blocks",
    "passing_yards", "rushing_yards", "passing_interceptions", "fumbles_lost_total",
    "sacks_suffered",
]

# Season-t per-game rates that describe a defense.
_RATES = [
    "sacks", "qb_hits", "interceptions", "passes_defended", "forced_fumbles",
    "fumble_recoveries", "touchdowns", "blocks_safeties", "points_allowed",
    "pa_points", "yards_allowed", "points_scored",
]

# The model's inputs, in a fixed order. Saved in the artifact so serving
# builds exactly the same vector.
FEATURES = [
    "ppg", "ppg_prev", "ppg_change",
    *[f"{c}_pg" for c in _RATES],
    # Next season's schedule, rated by the opponents' season-t offenses.
    "sched_opp_points_pg", "sched_opp_giveaways_pg", "sched_opp_sacks_taken_pg",
]


def points_allowed_points(points_allowed) -> np.ndarray:
    pa = np.asarray(points_allowed, dtype=float)
    out = np.full(pa.shape, float(POINTS_ALLOWED_FLOOR))
    # Fill from the loosest tier to the tightest so the tightest wins.
    for most, pts in reversed(POINTS_ALLOWED_TIERS):
        out = np.where(pa <= most, pts, out)
    return out


def load_team_weeks(raw_dir: Path, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = raw_file(raw_dir, f"stats_team_week_{season}.csv")
        frames.append(pd.read_csv(path, usecols=_WEEK_COLUMNS, low_memory=False))
    return pd.concat(frames, ignore_index=True)


_GAME_COLUMNS = ("season", "game_type", "week", "home_team", "away_team",
                 "home_score", "away_score")
# Used when present: roof type (kicker indoor share) and the stadium, which
# fills in a roof that games.csv hasn't recorded yet for an upcoming game.
_OPTIONAL_GAME_COLUMNS = ("roof", "stadium_id")


def load_games(raw_dir: Path, require: Iterable[str] = ()) -> pd.DataFrame:
    """``games.csv``. ``require`` names optional columns the caller can't do
    without; requiring ``roof`` (the kicker model; D/ST doesn't use it) also
    fills missing roofs (:func:`fill_roofs`)."""
    wanted = {*_GAME_COLUMNS, *_OPTIONAL_GAME_COLUMNS}
    games = pd.read_csv(raw_file(raw_dir, "games.csv"), usecols=lambda c: c in wanted,
                        low_memory=False)
    missing = [c for c in (*_GAME_COLUMNS, *require) if c not in games.columns]
    if missing:
        raise ValueError(f"games.csv is missing expected columns: {missing}")
    return fill_roofs(games) if "roof" in require else games


def fill_roofs(games: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing roof (an upcoming game) with the roof most often recorded
    at that stadium in its latest *complete* season (every game's roof
    recorded), so a retractable roof follows current practice, not a
    decade-old one. A season in progress never counts (its early games are
    open, its late ones unrecorded), so a stadium with no complete season yet,
    such as a team's first year in a new building, stays NaN and is skipped."""
    if "roof" not in games or "stadium_id" not in games or games["roof"].notna().all():
        return games
    complete = (games.dropna(subset=["stadium_id"])
                .groupby(["stadium_id", "season"])["roof"].agg(lambda s: s.notna().all()))
    known = games.dropna(subset=["roof", "stadium_id"])
    is_complete = [complete.get((s, y), False)
                   for s, y in zip(known["stadium_id"], known["season"])]
    known = known[is_complete]
    latest = known[known["season"] == known.groupby("stadium_id")["season"].transform("max")]
    usual = latest.groupby("stadium_id")["roof"].agg(lambda s: s.mode().iat[0])
    out = games.copy()
    out["roof"] = out["roof"].fillna(out["stadium_id"].map(usual))
    return out


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games, one row per team per game (both sides). Shared
    with ``ml/kicker_features.py``.
    ``indoor`` is 1.0 for a dome or closed roof, 0.0 otherwise, and NaN when
    the roof is unknown (so a mean over a schedule skips it)."""
    reg = games[games["game_type"] == "REG"]
    roof = reg["roof"] if "roof" in reg else pd.Series(np.nan, index=reg.index)
    indoor = roof.isin(INDOOR_ROOFS).astype(float).where(roof.notna())
    home = pd.DataFrame({
        "season": reg["season"], "week": reg["week"],
        "team": reg["home_team"], "opponent": reg["away_team"],
        "points_scored": reg["home_score"], "points_allowed": reg["away_score"],
        "indoor": indoor,
    })
    away = pd.DataFrame({
        "season": reg["season"], "week": reg["week"],
        "team": reg["away_team"], "opponent": reg["home_team"],
        "points_scored": reg["away_score"], "points_allowed": reg["home_score"],
        "indoor": indoor,
    })
    out = pd.concat([home, away], ignore_index=True)
    out["team"] = out["team"].map(current_abbr)
    out["opponent"] = out["opponent"].map(current_abbr)
    return out


def weekly_points(weeks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season D/ST fantasy points for every team-game."""
    w = weeks[weeks["season_type"] == "REG"].copy()
    w["team"] = w["team"].map(current_abbr)
    w["opponent_team"] = w["opponent_team"].map(current_abbr)
    w = w.fillna({c: 0 for c in _WEEK_COLUMNS if c not in ("season", "week", "team",
                                                           "season_type", "opponent_team")})
    scores = team_games(games)[["season", "week", "team", "points_scored", "points_allowed"]]
    w = w.merge(scores, on=["season", "week", "team"], how="inner")

    # The opponent's offensive row in the same game gives yards allowed.
    opp = w[["season", "week", "team", "passing_yards", "rushing_yards"]].rename(
        columns={"team": "opponent_team", "passing_yards": "opp_pass", "rushing_yards": "opp_rush"}
    )
    w = w.merge(opp, on=["season", "week", "opponent_team"], how="left")

    w["sacks"] = w["def_sacks"]
    w["qb_hits"] = w["def_qb_hits"]
    w["interceptions"] = w["def_interceptions"]
    w["passes_defended"] = w["def_pass_defended"]
    w["forced_fumbles"] = w["def_fumbles_forced"]
    w["fumble_recoveries"] = w["fumble_recovery_opp"]
    w["touchdowns"] = w["def_tds"] + w["special_teams_tds"]
    w["blocked_kicks"] = w["def_punt_blocks"] + w["def_pat_blocks"] + w["def_fg_blocks"]
    w["blocks_safeties"] = w["blocked_kicks"] + w["def_safeties"]
    w["yards_allowed"] = w["opp_pass"].fillna(0) + w["opp_rush"].fillna(0)
    w["pa_points"] = points_allowed_points(w["points_allowed"])
    w["giveaways"] = w["passing_interceptions"] + w["fumbles_lost_total"]

    w["dst_points"] = (
        SCORING["sacks"] * w["sacks"]
        + SCORING["interceptions"] * w["interceptions"]
        + SCORING["fumble_recoveries"] * w["fumble_recoveries"]
        + SCORING["safeties"] * w["def_safeties"]
        + SCORING["touchdowns"] * w["touchdowns"]
        + SCORING["blocked_kicks"] * w["blocked_kicks"]
        + w["pa_points"]
    )
    return w.reset_index(drop=True)


def season_rows(weeks: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-team, per-season feature rows (no label yet)."""
    w = weekly_points(weeks, games)
    sums = w.groupby(["team", "season"]).agg(
        games=("week", "size"),
        dst_points=("dst_points", "sum"),
        giveaways=("giveaways", "sum"),
        sacks_taken=("sacks_suffered", "sum"),
        **{c: (c, "sum") for c in _RATES},
    ).reset_index()
    df = sums.copy()
    df["ppg"] = df["dst_points"] / df["games"]
    for c in _RATES:
        df[f"{c}_pg"] = df[c] / df["games"]

    df = df.sort_values(["team", "season"])
    prev = df.groupby("team")[["season", "ppg"]].shift(1)
    consecutive = prev["season"] == df["season"] - 1
    df["ppg_prev"] = prev["ppg"].where(consecutive)
    df["ppg_change"] = df["ppg"] - df["ppg_prev"]

    # Next season's schedule, rated by what each opponent's offense did in
    # season t: points scored, giveaways, and sacks taken per game.
    offense = df[["team", "season"]].copy()
    offense["opp_points_pg"] = df["points_scored_pg"]
    offense["opp_giveaways_pg"] = df["giveaways"] / df["games"]
    offense["opp_sacks_taken_pg"] = df["sacks_taken"] / df["games"]
    sched = team_games(games)[["season", "team", "opponent"]].copy()
    sched["season"] = sched["season"] - 1  # the season whose stats rate it
    sched = sched.merge(
        offense.rename(columns={"team": "opponent"}), on=["season", "opponent"], how="left"
    )
    strength = sched.groupby(["team", "season"]).agg(
        sched_opp_points_pg=("opp_points_pg", "mean"),
        sched_opp_giveaways_pg=("opp_giveaways_pg", "mean"),
        sched_opp_sacks_taken_pg=("opp_sacks_taken_pg", "mean"),
    ).reset_index()
    df = df.merge(strength, on=["team", "season"], how="left")
    return df.reset_index(drop=True)


def training_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Join each season-t row to the same team's season t + 1 D/ST points per
    game. ``target_season`` is t + 1 — split train/validation/test on it."""
    nxt = rows[["team", "season", "ppg"]].rename(
        columns={"season": "target_season", "ppg": "target_ppg"}
    )
    nxt["season"] = nxt["target_season"] - 1
    return rows.merge(nxt, on=["team", "season"], how="inner").reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)
