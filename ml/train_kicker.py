"""Train, evaluate, and export the next-season kicker fantasy points-per-game model.

    python -m ml.download
    python -m ml.train_kicker

About 30–40 kickers a season qualify, so models are picked on four rolling
validation seasons, scored once on the last season, and backtested over many
seasons (see ``ml/rolling.py``), against "same as last season" and "league
average" baselines.

The winner is refit on every season and exported with the latest season's
rows, keyed by normalized name (like the player model). Tracked in MLflow
(experiment and registered model ``kicker-projection``) when it is installed.

    python -m ml.train_kicker --no-mlflow
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import sklearn

from analysis_core.projection import DEFAULT_KICKER_ARTIFACT_PATH, KICKER_ARTIFACT_FORMAT
from analysis_core.season import current_season

from analysis_core.projection import normalize_name

from . import kicker_features as K
from . import rolling
from .download import raw_file
from .dst_features import load_games
from .features import load_players
from .download import TEAM_FIRST
from .tracking import NullTracker, make_tracker

ML_DIR = Path(__file__).parent
RAW_DIR = ML_DIR / "data" / "raw"
REPORT_DIR = ML_DIR / "reports"

EXPERIMENT = "kicker-projection"


def non_kicker_names(raw_dir: Path) -> set:
    """Normalized names of every non-kicker in ``players.csv`` (rookies included)."""
    people = pd.read_csv(raw_file(raw_dir, "players.csv"),
                         usecols=["display_name", "position"], low_memory=False)
    others = people[people["position"] != "K"]["display_name"].dropna()
    return set(others.map(normalize_name))


def serving_kickers(rows: pd.DataFrame, feature_season: int, ambiguous=frozenset()) -> dict:
    """Latest-season rows keyed by normalized name; on a name clash keep the
    kicker who played more games.

    Lookup is by name alone and falls through from the player model, so a key
    that also names a non-kicker (``ambiguous``) is left out: a rookie WR who
    shares a kicker's name must not get the kicker's projection.
    """
    latest = rows[(rows["season"] == feature_season) & (rows["games"] >= K.MIN_GAMES)]
    latest = latest[~latest["name_key"].isin(ambiguous)]
    latest = latest.sort_values("games", ascending=False).drop_duplicates("name_key")
    return {
        rec.name_key: {
            "name": rec.player_display_name,
            "position": "K",
            "team": rec.team,
            "ppg": round(float(rec.ppg), 2),
            "games": int(rec.games),
            "features": rolling.serving_vector(rec, K.FEATURES),
        }
        for rec in latest.itertuples(index=False)
    }


def run(raw_dir: Path, first: int, last: int, out: Path, report_dir: Path,
        tracker: Optional[NullTracker] = None, *, register: bool = True) -> dict:
    with tracker or NullTracker() as tr:
        return _run(raw_dir, first, last, out, report_dir, tr, register)


def _run(raw_dir, first, last, out, report_dir, tracker, register) -> dict:
    stats = K.load_kicking(raw_dir, range(first, last + 1))
    rows = K.season_rows(stats, load_players(raw_dir), load_games(raw_dir))
    table = K.training_table(rows)

    tracker.log_params({
        "first_season": first, "last_season": last,
        "validation_target_seasons":
            f"{last - rolling.VALIDATION_SEASONS}-{last - 1}",
        "test_target_season": last, "n_features": len(K.FEATURES),
        "min_games": K.MIN_GAMES, "rows_final": len(table),
        "sklearn_version": sklearn.__version__,
    })
    best, final, sections = rolling.evaluate(table, first, last, K.feature_matrix, tracker)
    report = {
        "trained_on": date.today().isoformat(),
        "data": "nflverse stats_player_reg + players + games (CC-BY 4.0)",
        "scoring": dict(K.SCORING),
        "seasons": [first, last],
        **sections,
        "sklearn_version": sklearn.__version__,
        "mlflow_run_id": tracker.run_id,
    }
    artifact = {
        "format": KICKER_ARTIFACT_FORMAT,
        "model": final,
        "features": list(K.FEATURES),
        "sklearn_version": sklearn.__version__,
        "feature_season": last,
        "target_season": last + 1,
        "model_name": best.name,
        "test_mae": report["test"]["model"]["mae"],
        "mlflow_run_id": tracker.run_id,
        "players": serving_kickers(rows, last, non_kicker_names(raw_dir)),
    }
    tracker.log_params({f"chosen_{k}": v for k, v in best.params.items()})
    rolling.save_and_track(
        artifact=artifact, report=report, markdown=render_report(report), out=out,
        report_dir=report_dir, stem="kicker", tracker=tracker, final_model=final,
        X=K.feature_matrix(table), register=register,
    )
    return report


def render_report(r: dict) -> str:
    s = r["scoring"]
    lines = [
        "# Kicker fantasy points-per-game model: evaluation",
        "",
        f"Generated by `python -m ml.train_kicker` on {r['trained_on']} "
        f"(scikit-learn {r['sklearn_version']}).",
        f"Data: {r['data']}, seasons {r['seasons'][0]}–{r['seasons'][1]}.",
        "",
        f"**Scoring (standard kicker):** PAT {s['pat_made']:+d}, missed PAT {s['pat_missed']:+d}; "
        f"FG 0–39 yds {s['fg_0_39']:+d}, 40–49 {s['fg_40_49']:+d}, 50+ {s['fg_50_plus']:+d}; "
        f"missed FG {s['fg_missed']:+d} (a blocked kick counts as a miss).",
        "",
        *rolling.render_results(r, "kicker"),
        "",
        "## Limitations",
        "",
        "- Kicker scoring swings a lot from year to year: it depends on how often the offense "
        "stalls in range, which barely carries over. Expect only a modest gain over the "
        "league average.",
        f"- Trained on kickers with at least {K.MIN_GAMES} games in both seasons, so it predicts "
        "scoring *when he kicks*, not whether he keeps the job.",
        "- Team context (offense, next season's schedule, indoor games) is the kicker's "
        "season-t team. A kicker who changes teams is projected as if he hadn't; the LLM's "
        "web search covers signings and cuts.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--first", type=int, default=TEAM_FIRST)
    parser.add_argument("--last", type=int, default=current_season() - 1)
    parser.add_argument("--out", type=Path, default=DEFAULT_KICKER_ARTIFACT_PATH)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-mlflow", action="store_true", help="train without MLflow tracking")
    parser.add_argument("--no-register", action="store_true",
                        help="track the run but don't register a model version")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)
    tracker = make_tracker(enabled=not args.no_mlflow, tracking_uri=args.tracking_uri,
                           run_name=f"train-kicker-{args.first}-{args.last}",
                           experiment=EXPERIMENT, registered_model=EXPERIMENT)
    report = run(args.raw, args.first, args.last, args.out, args.reports, tracker,
                 register=not args.no_register)
    print(json.dumps({k: report.get(k) for k in (
        "chosen_model", "test_mae_improvement_pct", "mlflow_run_id", "mlflow_model_version")}))
    print(json.dumps(report["test"]), json.dumps(report["backtest"]))


if __name__ == "__main__":
    main()
