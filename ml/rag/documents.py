"""Build the retrieval corpus: one text document per player-season.

Each document describes a QB/RB/WR/TE season: role and production, snap
share, and the weekly injury reports (weeks listed, game statuses, injury
types, practices missed). Seasons with a following season also say what
happened next, which is what makes a past season useful as a *comparable*
for a current player.

Sources (nflverse, CC-BY 4.0): ``stats_player_reg_<season>.csv``,
``injuries_<season>.csv``, ``snap_counts_<season>.csv`` (from 2013), and
``players.csv`` for IDs, birth dates, and draft round.

Training-only: needs pandas. The services only read the indexed documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document

from analysis_core.retrieval import KIND_PLAYER_SEASON

from .. import features as F

# Report statuses that count as an injury designation for a game.
_GAME_STATUSES = ("Out", "Doubtful", "Questionable")
_DNP = "Did Not Participate In Practice"


def team_games(season: int) -> int:
    """Regular-season games per team (17 from 2021)."""
    return 17 if season >= 2021 else 16


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def load_injuries(raw_dir: Path, seasons: Iterable[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(raw_dir) / f"injuries_{season}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["season", "game_type", "week", "gsis_id",
                                     "report_primary_injury", "report_status", "practice_status"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["game_type"] == "REG"]
    # Rest days and personal absences are on the report too; they aren't injuries.
    not_injury = (
        df["report_primary_injury"].fillna("").astype(str).str.lower().str.startswith("not injury related")
        | df["practice_primary_injury"].fillna("").astype(str).str.lower().str.startswith("not injury related")
    )
    return df[~not_injury]


def load_snaps(raw_dir: Path, seasons: Iterable[int], players: pd.DataFrame) -> pd.DataFrame:
    """Average offensive snap share per (player_id, season), regular season."""
    frames = []
    for season in seasons:
        path = Path(raw_dir) / f"snap_counts_{season}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["season", "game_type", "pfr_player_id", "offense_pct"],
                             low_memory=False)
            frames.append(df[df["game_type"] == "REG"])
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "snap_share"])
    snaps = pd.concat(frames, ignore_index=True)
    ids = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    out = snaps.groupby(["gsis_id", "season"], as_index=False)["offense_pct"].mean()
    return out.rename(columns={"gsis_id": "player_id", "offense_pct": "snap_share"})


def injury_summaries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, season): weeks listed, status counts, top injuries, DNPs."""
    if injuries.empty:
        return pd.DataFrame(columns=["player_id", "season", "injury_text", "weeks_listed"])
    rows = []
    for (pid, season), g in injuries.groupby(["gsis_id", "season"]):
        listed = g[g["report_status"].isin(_GAME_STATUSES)]
        status = {s: int((listed["report_status"] == s).sum()) for s in _GAME_STATUSES}
        types = listed["report_primary_injury"].dropna().astype(str).str.strip()
        types = types[types != ""].value_counts()
        dnp = int((g["practice_status"] == _DNP).sum())
        if len(listed) == 0 and dnp == 0:
            continue
        parts = [f"listed for {plural(len(listed), 'game')} "
                 f"({status['Out']} out, {status['Doubtful']} doubtful, "
                 f"{status['Questionable']} questionable)"]
        if len(types):
            parts.append("injuries: " + ", ".join(
                f"{name.lower()} ({plural(int(n), 'week')})" for name, n in types.head(3).items()
            ))
        if dnp:
            parts.append(f"did not practice in {plural(dnp, 'report')}")
        rows.append({"player_id": pid, "season": int(season),
                     "injury_text": "; ".join(parts), "weeks_listed": len(listed)})
    return pd.DataFrame(rows)


def _role_text(r) -> str:
    pos = r.position
    if pos == "QB":
        return (f"{r.attempts_pg:.0f} pass attempts, {r.passing_yards_pg:.0f} passing yards, "
                f"{r.passing_tds_pg:.1f} passing TDs and {r.carries_pg:.1f} carries per game")
    if pos == "RB":
        return (f"{r.carries_pg:.1f} carries, {r.rushing_yards_pg:.0f} rushing yards, "
                f"{r.targets_pg:.1f} targets and {r.receptions_pg:.1f} receptions per game")
    return (f"{r.targets_pg:.1f} targets, {r.receptions_pg:.1f} receptions and "
            f"{r.receiving_yards_pg:.0f} receiving yards per game, "
            f"{100 * r.target_share:.0f}% target share")


def profile_text(r) -> str:
    """The part of a document known at the end of season t (no outcome)."""
    age = "" if pd.isna(r.age) else f", age {r.age - 1:.0f}"  # age during season t
    snap = "" if pd.isna(r.snap_share) else f" Played {100 * r.snap_share:.0f}% of offensive snaps."
    injury = (f" Injury report: {r.injury_text}." if isinstance(r.injury_text, str)
              else " Not listed on the injury report.")
    return (f"{r.player_display_name} ({r.position}, {r.recent_team}{age}) {r.season} season: "
            f"played {int(r.games)} of {team_games(int(r.season))} games, "
            f"{r.ppg:.1f} PPR points per game. "
            f"Role: {_role_text(r)}.{snap}{injury}")


def outcome_text(next_ppg: float, next_games: int, ppg: float) -> str:
    delta = next_ppg - ppg
    return (f" Next season: {next_ppg:.1f} PPR points per game in {plural(next_games, 'game')} "
            f"({'+' if delta >= 0 else ''}{delta:.1f}).")


def build_documents(raw_dir: Path, first: int, last: int) -> list[Document]:
    seasons = range(first, last + 1)
    players = pd.read_csv(Path(raw_dir) / "players.csv", low_memory=False)
    rows = F.season_rows(F.load_season_stats(raw_dir, seasons), F.load_players(raw_dir))
    rows = rows.merge(load_snaps(raw_dir, seasons, players), on=["player_id", "season"], how="left")
    rows = rows.merge(injury_summaries(load_injuries(raw_dir, seasons)),
                      on=["player_id", "season"], how="left")

    nxt = rows[["player_id", "season", "ppg", "games"]].rename(
        columns={"ppg": "next_ppg", "games": "next_games"})
    nxt["season"] = nxt["season"] - 1
    rows = rows.merge(nxt, on=["player_id", "season"], how="left")

    docs = []
    for r in rows.itertuples(index=False):
        has_next = not pd.isna(r.next_ppg)
        text = profile_text(r)
        if has_next:
            text += outcome_text(float(r.next_ppg), int(r.next_games), float(r.ppg))
        docs.append(Document(
            page_content=text,
            metadata={
                "kind": KIND_PLAYER_SEASON,
                "player_id": r.player_id,
                "name": r.player_display_name,
                "name_key": r.name_key,
                "position": r.position,
                "team": r.recent_team,
                "season": int(r.season),
                "games": int(r.games),
                "ppg": round(float(r.ppg), 2),
                "weeks_listed": 0 if pd.isna(r.weeks_listed) else int(r.weeks_listed),
                "has_next": bool(has_next),
                "next_ppg": round(float(r.next_ppg), 2) if has_next else None,
                "next_games": int(r.next_games) if has_next else None,
                # What the comparables query embeds: the season without its outcome.
                "profile": text if not has_next else profile_text(r),
            },
        ))
    return docs
