"""Download the nflverse files the model trains on and retrieval indexes.

    python -m ml.download                     # 2012 .. last completed season
    python -m ml.download --first 2015 --last 2025

Files land in ml/data/raw/ (gitignored). Data: nflverse
(https://github.com/nflverse/nflverse-data), CC-BY 4.0.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from analysis_core.season import current_season

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
RAW_DIR = Path(__file__).parent / "data" / "raw"


def stats_url(season: int) -> str:
    return f"{BASE}/stats_player/stats_player_reg_{season}.csv"


def injuries_url(season: int) -> str:
    return f"{BASE}/injuries/injuries_{season}.csv"


def snap_counts_url(season: int) -> str:
    return f"{BASE}/snap_counts/snap_counts_{season}.csv"


PLAYERS_URL = f"{BASE}/players/players.csv"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-draft-consultant-ml"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2012)
    # The latest season with final regular-season stats is the one before the
    # current fantasy season.
    parser.add_argument("--last", type=int, default=current_season() - 1)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    for season in range(args.first, args.last + 1):
        fetch(stats_url(season), args.out / f"stats_player_reg_{season}.csv")
        # Retrieval corpus (ml/rag): weekly injury reports and snap counts.
        fetch(injuries_url(season), args.out / f"injuries_{season}.csv")
        fetch(snap_counts_url(season), args.out / f"snap_counts_{season}.csv")
        print(f"stats, injuries, snap counts {season}")
    fetch(PLAYERS_URL, args.out / "players.csv")
    print("players")


if __name__ == "__main__":
    main()
