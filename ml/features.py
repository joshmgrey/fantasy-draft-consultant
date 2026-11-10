"""Feature engineering for the next-season PPR points-per-game model.

One row per (player, season t). Features come only from season t and earlier;
the label is the player's PPR points per game in season t + 1. Nothing from
t + 1 (or later) may leak into the features, and ``tests/test_ml_features.py``
checks that.

Inputs are nflverse files (https://github.com/nflverse/nflverse-data):

* ``stats_player_reg_<season>.csv`` — regular-season totals per player
* ``players.csv`` — birth date, rookie season, draft round

Training-only module: it needs pandas, which the deployed services do not
install. The services only load the trained artifact (see
``analysis_core/projection.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Shared with serving so a typed name maps to the same key the model was
# exported under.
from analysis_core.projection import normalize_name

POSITIONS = ("QB", "RB", "WR", "TE")

# A season with fewer games than this is too noisy to learn from as a label
# (and too thin to describe a player as a feature row).
MIN_GAMES = 4

# Undrafted players get this draft round so the feature stays numeric and
# sorts after round 7.
UNDRAFTED_ROUND = 8

_STAT_COLUMNS = [
    "player_id", "player_display_name", "position", "season", "recent_team",
    "games", "fantasy_points_ppr", "targets", "receptions", "target_share",
    "air_yards_share", "wopr", "carries", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "attempts", "passing_yards",
    "passing_tds", "passing_interceptions", "passing_epa", "rushing_epa",
    "receiving_epa",
]

# Per-game rates are what carry over between seasons; season totals mostly
# measure how many games a player happened to play.
_PER_GAME = [
    "targets", "receptions", "carries", "rushing_yards", "rushing_tds",
    "receiving_yards", "receiving_tds", "attempts", "passing_yards",
    "passing_tds", "passing_interceptions", "passing_epa", "rushing_epa",
    "receiving_epa",
]

# The model's inputs, in a fixed order. Saved in the artifact so serving
# builds exactly the same vector.
FEATURES = [
    "ppg", "games", "ppg_prev", "games_prev", "ppg_change",
    "target_share", "air_yards_share", "wopr",
    *[f"{c}_pg" for c in _PER_GAME],
    "age", "years_exp", "draft_round",
    *[f"pos_{p}" for p in POSITIONS],
]

def load_season_stats(raw_dir: Path, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(raw_dir) / f"stats_player_reg_{season}.csv"
        frames.append(pd.read_csv(path, usecols=_STAT_COLUMNS, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def load_players(raw_dir: Path) -> pd.DataFrame:
    cols = ["gsis_id", "birth_date", "rookie_season", "draft_round"]
    return pd.read_csv(Path(raw_dir) / "players.csv", usecols=cols, low_memory=False)


def season_rows(stats: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per-player, per-season feature rows (no label yet)."""
    df = stats[stats["position"].isin(POSITIONS) & (stats["games"] > 0)].copy()
    df["ppg"] = df["fantasy_points_ppr"] / df["games"]
    for col in _PER_GAME:
        df[f"{col}_pg"] = df[col].fillna(0) / df["games"]
    for col in ("target_share", "air_yards_share", "wopr"):
        df[col] = df[col].fillna(0.0)

    df = df.sort_values(["player_id", "season"])
    prev = df.groupby("player_id")[["season", "ppg", "games"]].shift(1)
    # Only count the previous row if it really was the previous season.
    consecutive = prev["season"] == df["season"] - 1
    df["ppg_prev"] = prev["ppg"].where(consecutive)
    df["games_prev"] = prev["games"].where(consecutive)
    df["ppg_change"] = df["ppg"] - df["ppg_prev"]

    bio = players.rename(columns={"gsis_id": "player_id"}).drop_duplicates("player_id")
    df = df.merge(bio, on="player_id", how="left")
    birth = pd.to_datetime(df["birth_date"], errors="coerce")
    # Age at the start of the season being predicted (t + 1), in years.
    kickoff = pd.to_datetime((df["season"] + 1).astype(str) + "-09-01")
    df["age"] = (kickoff - birth).dt.days / 365.25
    df["years_exp"] = (df["season"] - df["rookie_season"] + 1).clip(lower=1)
    df["draft_round"] = df["draft_round"].fillna(UNDRAFTED_ROUND)

    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)
    df["name_key"] = df["player_display_name"].map(normalize_name)
    return df.reset_index(drop=True)


def training_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Join each season-t row to the same player's season t + 1 PPR per game.

    Keeps pairs where the player played at least ``MIN_GAMES`` in both
    seasons. ``target_season`` is t + 1 — split train/validation/test on it.
    """
    base = rows[rows["games"] >= MIN_GAMES]
    nxt = rows.loc[rows["games"] >= MIN_GAMES, ["player_id", "season", "ppg"]].rename(
        columns={"season": "target_season", "ppg": "target_ppg"}
    )
    nxt["season"] = nxt["target_season"] - 1
    table = base.merge(nxt, on=["player_id", "season"], how="inner")
    return table.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)
