"""Feature engineering for the next-season kicker fantasy points-per-game model.

One row per (kicker, season t). Features come only from season t, plus the
season t + 1 *schedule* of the kicker's season-t team (published in May, so
known before any draft): how many of those games are indoors, and how many
points those opponents allowed in season t. The label is the kicker's
fantasy points per game in season t + 1. ``tests/test_kicker_features.py``
checks that no t + 1 result leaks into the features.

Inputs are nflverse files (https://github.com/nflverse/nflverse-data):

* ``stats_player_reg_<season>.csv`` — regular-season kicking totals
* ``players.csv`` — birth date, rookie season
* ``games.csv`` — final scores, roof type, and future schedules

Kicker fantasy points use common standard scoring (see :data:`SCORING`); a
blocked kick counts as a miss. Training-only module (pandas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis_core.projection import normalize_name
from analysis_core.teams import current_abbr

SCORING = {
    "pat_made": 1,
    "pat_missed": -1,
    "fg_0_39": 3,
    "fg_40_49": 4,
    "fg_50_plus": 5,
    "fg_missed": -1,
}

# Same threshold as the player model: fewer games is too noisy as a label.
MIN_GAMES = 4

INDOOR_ROOFS = ("dome", "closed")

_STAT_COLUMNS = [
    "player_id", "player_display_name", "position", "season", "recent_team", "games",
    "fg_made", "fg_att", "fg_long",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
    "fg_made_50_59", "fg_made_60_",
    "fg_missed_40_49", "fg_missed_50_59", "fg_missed_60_",
    "pat_made", "pat_att",
]

# The model's inputs, in a fixed order. Saved in the artifact so serving
# builds exactly the same vector.
FEATURES = [
    "ppg", "ppg_prev", "ppg_change", "games",
    "fg_att_pg", "fg_pct", "fg_40_49_att_pg", "fg_50_att_pg", "fg_50_pct",
    "pat_att_pg", "pat_pct", "fg_long",
    "age", "years_exp",
    "team_points_pg",
    # Next season's schedule for the kicker's season-t team.
    "sched_indoor_share", "sched_opp_points_allowed_pg",
]


def load_kicking(raw_dir: Path, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(raw_dir) / f"stats_player_reg_{season}.csv"
        frame = pd.read_csv(path, usecols=_STAT_COLUMNS, low_memory=False)
        frames.append(frame[frame["position"] == "K"])
    return pd.concat(frames, ignore_index=True)


def load_players(raw_dir: Path) -> pd.DataFrame:
    cols = ["gsis_id", "birth_date", "rookie_season"]
    return pd.read_csv(Path(raw_dir) / "players.csv", usecols=cols, low_memory=False)


def load_games(raw_dir: Path) -> pd.DataFrame:
    cols = ["season", "game_type", "week", "home_team", "away_team",
            "home_score", "away_score", "roof"]
    return pd.read_csv(Path(raw_dir) / "games.csv", usecols=cols, low_memory=False)


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    made_0_39 = df["fg_made_0_19"] + df["fg_made_20_29"] + df["fg_made_30_39"]
    made_50 = df["fg_made_50_59"] + df["fg_made_60_"]
    fg_missed = df["fg_att"] - df["fg_made"]           # blocked kicks count as misses
    pat_missed = df["pat_att"] - df["pat_made"]
    return (
        SCORING["pat_made"] * df["pat_made"]
        + SCORING["pat_missed"] * pat_missed
        + SCORING["fg_0_39"] * made_0_39
        + SCORING["fg_40_49"] * df["fg_made_40_49"]
        + SCORING["fg_50_plus"] * made_50
        + SCORING["fg_missed"] * fg_missed
    )


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games, one row per team per game (both sides)."""
    reg = games[games["game_type"] == "REG"]
    sides = []
    for team, opp, scored, allowed in (("home_team", "away_team", "home_score", "away_score"),
                                       ("away_team", "home_team", "away_score", "home_score")):
        sides.append(pd.DataFrame({
            "season": reg["season"], "team": reg[team].map(current_abbr),
            "opponent": reg[opp].map(current_abbr),
            "points_scored": reg[scored], "points_allowed": reg[allowed],
            "indoor": reg["roof"].isin(INDOOR_ROOFS).astype(float),
        }))
    return pd.concat(sides, ignore_index=True)


def season_rows(stats: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per-kicker, per-season feature rows (no label yet)."""
    df = stats[(stats["games"] > 0) & ((stats["fg_att"] + stats["pat_att"]) > 0)].copy()
    num = [c for c in _STAT_COLUMNS if c not in
           ("player_id", "player_display_name", "position", "season", "recent_team")]
    df[num] = df[num].fillna(0)
    df["team"] = df["recent_team"].map(current_abbr)

    df["points"] = fantasy_points(df)
    df["ppg"] = df["points"] / df["games"]
    df["fg_att_pg"] = df["fg_att"] / df["games"]
    df["fg_pct"] = (df["fg_made"] / df["fg_att"]).where(df["fg_att"] > 0)
    fg_40_49_att = df["fg_made_40_49"] + df["fg_missed_40_49"]
    fg_50_made = df["fg_made_50_59"] + df["fg_made_60_"]
    fg_50_att = fg_50_made + df["fg_missed_50_59"] + df["fg_missed_60_"]
    df["fg_40_49_att_pg"] = fg_40_49_att / df["games"]
    df["fg_50_att_pg"] = fg_50_att / df["games"]
    df["fg_50_pct"] = (fg_50_made / fg_50_att).where(fg_50_att > 0)
    df["pat_att_pg"] = df["pat_att"] / df["games"]
    df["pat_pct"] = (df["pat_made"] / df["pat_att"]).where(df["pat_att"] > 0)

    df = df.sort_values(["player_id", "season"])
    prev = df.groupby("player_id")[["season", "ppg"]].shift(1)
    consecutive = prev["season"] == df["season"] - 1
    df["ppg_prev"] = prev["ppg"].where(consecutive)
    df["ppg_change"] = df["ppg"] - df["ppg_prev"]

    bio = players.rename(columns={"gsis_id": "player_id"}).drop_duplicates("player_id")
    df = df.merge(bio, on="player_id", how="left")
    birth = pd.to_datetime(df["birth_date"], errors="coerce")
    kickoff = pd.to_datetime((df["season"] + 1).astype(str) + "-09-01")
    df["age"] = (kickoff - birth).dt.days / 365.25
    df["years_exp"] = (df["season"] - df["rookie_season"] + 1).clip(lower=1)

    tg = _team_games(games)
    team_season = tg.groupby(["team", "season"]).agg(
        team_points_pg=("points_scored", "mean"),
        points_allowed_pg=("points_allowed", "mean"),
    ).reset_index()
    df = df.merge(team_season[["team", "season", "team_points_pg"]],
                  on=["team", "season"], how="left")

    # Next season's schedule for this team, rated by season-t results.
    sched = tg[["season", "team", "opponent", "indoor"]].copy()
    sched["season"] = sched["season"] - 1
    sched = sched.merge(
        team_season[["team", "season", "points_allowed_pg"]].rename(columns={"team": "opponent"}),
        on=["season", "opponent"], how="left",
    )
    strength = sched.groupby(["team", "season"]).agg(
        sched_indoor_share=("indoor", "mean"),
        sched_opp_points_allowed_pg=("points_allowed_pg", "mean"),
    ).reset_index()
    df = df.merge(strength, on=["team", "season"], how="left")

    df["name_key"] = df["player_display_name"].map(normalize_name)
    return df.reset_index(drop=True)


def training_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Join each season-t row to the same kicker's season t + 1 points per
    game. Keeps pairs with at least ``MIN_GAMES`` in both seasons."""
    base = rows[rows["games"] >= MIN_GAMES]
    nxt = base[["player_id", "season", "ppg"]].rename(
        columns={"season": "target_season", "ppg": "target_ppg"}
    )
    nxt["season"] = nxt["target_season"] - 1
    return base.merge(nxt, on=["player_id", "season"], how="inner").reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)
