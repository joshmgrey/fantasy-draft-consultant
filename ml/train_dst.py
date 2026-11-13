"""Train, evaluate, and export the next-season D/ST fantasy points-per-game model.

    python -m ml.download
    python -m ml.train_dst

There are only 32 teams a season, so one validation season is too noisy to
pick a model on. Instead:

* validation  target seasons last-4 .. last-1, each scored by a model trained
              only on earlier seasons (rolling origin); candidates are ranked
              by their mean MAE across those four seasons
* test        target season last, trained on every earlier season, scored once

D/ST scoring is notoriously hard to predict year to year, so the model is
reported against two baselines: "same as last season" and "league average"
(the mean of every earlier team-season). Beating the second is the real bar.

The winner is refit on every season and exported with the latest season's
rows, keyed by team abbreviation. Tracked in MLflow (experiment and registered
model ``dst-projection``) when it is installed.

    python -m ml.train_dst --no-mlflow
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from analysis_core.projection import DST_ARTIFACT_FORMAT, DEFAULT_DST_ARTIFACT_PATH
from analysis_core.season import current_season
from analysis_core.teams import dst_display_name

from . import dst_features as D
from .download import TEAM_FIRST
from .tracking import NullTracker, make_tracker
from .train import Candidate, candidates
from .train import metrics as _metrics

ML_DIR = Path(__file__).parent
RAW_DIR = ML_DIR / "data" / "raw"
REPORT_DIR = ML_DIR / "reports"

VALIDATION_SEASONS = 4
# The long backtest starts once this many seasons are available to train on.
BACKTEST_MIN_TRAIN_SEASONS = 8
EXPERIMENT = "dst-projection"


def metrics(y_true, y_pred) -> dict:
    """``ml.train.metrics``, with an undefined Spearman (a constant prediction,
    like the league average) stored as ``None`` so the JSON report stays valid."""
    out = _metrics(y_true, y_pred)
    if out["spearman"] != out["spearman"]:  # NaN
        out["spearman"] = None
    return out


def baselines(train: pd.DataFrame, df: pd.DataFrame) -> dict:
    return {
        "last_season": df["ppg"].to_numpy(),
        "league_average": np.full(len(df), train["target_ppg"].mean()),
    }


def _fit(cand: Candidate, rows: pd.DataFrame):
    return cand.build().fit(D.feature_matrix(rows), rows["target_ppg"].to_numpy())


def select_model(table: pd.DataFrame, last: int,
                 tracker: Optional[NullTracker] = None) -> tuple[Candidate, list[dict]]:
    """Rank candidates by mean MAE over the rolling validation seasons."""
    tracker = tracker or NullTracker()
    seasons = range(last - VALIDATION_SEASONS, last)
    results = []
    best = None
    for cand in candidates():
        maes = []
        for season in seasons:
            train = table[table["target_season"] < season]
            val = table[table["target_season"] == season]
            pred = _fit(cand, train).predict(D.feature_matrix(val))
            maes.append(metrics(val["target_ppg"].to_numpy(), pred)["mae"])
        score = {"mae": round(float(np.mean(maes)), 3),
                 "mae_by_season": dict(zip(map(str, seasons), maes))}
        results.append({"model": cand.name, **score})
        tracker.log_candidate(cand.name, cand.params, {"mae": score["mae"]})
        if best is None or score["mae"] < best[0]:
            best = (score["mae"], cand)

    baseline_maes = {}
    for season in seasons:
        train = table[table["target_season"] < season]
        val = table[table["target_season"] == season]
        for name, pred in baselines(train, val).items():
            baseline_maes.setdefault(name, []).append(
                metrics(val["target_ppg"].to_numpy(), pred)["mae"])
    for name, maes in baseline_maes.items():
        results.append({"model": f"baseline: {name}", "mae": round(float(np.mean(maes)), 3),
                        "mae_by_season": dict(zip(map(str, seasons), maes))})
    return best[1], results


def backtest(cand: Candidate, table: pd.DataFrame, first: int, last: int) -> dict:
    """Mean MAE and Spearman over every target season from
    ``first + BACKTEST_MIN_TRAIN_SEASONS`` to ``last``, each scored by a model
    trained only on earlier seasons. Steadier than any one 32-team season.

    The chosen model was picked on the last few of these seasons, so its
    numbers here are slightly optimistic; the baselines' are not.
    """
    seasons = list(range(first + BACKTEST_MIN_TRAIN_SEASONS, last + 1))
    per = {}
    for season in seasons:
        train = table[table["target_season"] < season]
        test = table[table["target_season"] == season]
        y = test["target_ppg"].to_numpy()
        preds = {"model": _fit(cand, train).predict(D.feature_matrix(test)),
                 **{f"baseline_{k}": v for k, v in baselines(train, test).items()}}
        for name, pred in preds.items():
            per.setdefault(name, []).append(metrics(y, pred))

    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 3) if values else None

    return {
        "target_seasons": [seasons[0], seasons[-1]],
        **{name: {"mae": mean([m["mae"] for m in ms]),
                  "spearman": mean([m["spearman"] for m in ms])}
           for name, ms in per.items()},
    }


def serving_teams(rows: pd.DataFrame, feature_season: int) -> dict:
    latest = rows[rows["season"] == feature_season]
    teams = {}
    for rec in latest.itertuples(index=False):
        vec = [None if pd.isna(v) else float(v) for v in (getattr(rec, f) for f in D.FEATURES)]
        teams[rec.team] = {
            "name": dst_display_name(rec.team),
            "position": "D/ST",
            "team": rec.team,
            "ppg": round(float(rec.ppg), 2),
            "games": int(rec.games),
            "features": vec,
        }
    return teams


def run(raw_dir: Path, first: int, last: int, out: Path, report_dir: Path,
        tracker: Optional[NullTracker] = None, *, register: bool = True) -> dict:
    with tracker or NullTracker() as tr:
        return _run(raw_dir, first, last, out, report_dir, tr, register)


def _run(raw_dir, first, last, out, report_dir, tracker, register) -> dict:
    weeks = D.load_team_weeks(raw_dir, range(first, last + 1))
    rows = D.season_rows(weeks, D.load_games(raw_dir))
    table = D.training_table(rows)

    tracker.log_params({
        "first_season": first, "last_season": last,
        "validation_target_seasons": f"{last - VALIDATION_SEASONS}-{last - 1}",
        "test_target_season": last, "n_features": len(D.FEATURES),
        "rows_final": len(table), "sklearn_version": sklearn.__version__,
    })
    best, leaderboard = select_model(table, last, tracker)

    train = table[table["target_season"] < last]
    test = table[table["target_season"] == last]
    y_test = test["target_ppg"].to_numpy()
    model = _fit(best, train)
    test_scores = {"model": metrics(y_test, model.predict(D.feature_matrix(test)))}
    for name, pred in baselines(train, test).items():
        test_scores[f"baseline_{name}"] = metrics(y_test, pred)

    model_mae = test_scores["model"]["mae"]

    def improvement(name):
        base = test_scores[f"baseline_{name}"]["mae"]
        return round(100 * (base - model_mae) / base, 1)

    long_run = backtest(best, table, first, last)
    final = _fit(best, table)
    report = {
        "trained_on": date.today().isoformat(),
        "data": "nflverse stats_team_week + games (CC-BY 4.0)",
        "scoring": {**D.SCORING, "points_allowed_tiers": D.POINTS_ALLOWED_TIERS,
                    "points_allowed_floor": D.POINTS_ALLOWED_FLOOR},
        "seasons": [first, last],
        "rows": {"train": len(train), "test": len(test), "final": len(table)},
        "split": {
            "validation_target_seasons": [last - VALIDATION_SEASONS, last - 1],
            "test_target_season": last,
        },
        "chosen_model": best.name,
        "validation_leaderboard": sorted(leaderboard, key=lambda r: r["mae"]),
        "test": test_scores,
        "backtest": long_run,
        "test_mae_improvement_pct": {
            "vs_last_season": improvement("last_season"),
            "vs_league_average": improvement("league_average"),
        },
        "sklearn_version": sklearn.__version__,
        "mlflow_run_id": tracker.run_id,
    }

    artifact = {
        "format": DST_ARTIFACT_FORMAT,
        "model": final,
        "features": list(D.FEATURES),
        "sklearn_version": sklearn.__version__,
        "feature_season": last,
        "target_season": last + 1,
        "model_name": best.name,
        "test_mae": model_mae,
        "mlflow_run_id": tracker.run_id,
        "players": serving_teams(rows, last),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dst_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_dir / "DST_REPORT.md").write_text(render_report(report), encoding="utf-8")

    tracker.log_params({"chosen_model": best.name,
                        **{f"chosen_{k}": v for k, v in best.params.items()}})
    tracker.log_metrics({"test": test_scores,
                         "test_mae_improvement_pct": report["test_mae_improvement_pct"],
                         "backtest": long_run})
    tracker.log_artifacts([report_dir / "DST_REPORT.md", report_dir / "dst_metrics.json", out])
    tracker.log_model(final, D.feature_matrix(table), register=register)
    report["mlflow_model_version"] = tracker.model_version
    return report


def render_report(r: dict) -> str:
    t = r["test"]
    b = r["backtest"]
    imp = r["test_mae_improvement_pct"]
    v0, v1 = r["split"]["validation_target_seasons"]
    s = r["scoring"]
    tiers = ", ".join(f"≤{most}: {pts:+d}" for most, pts in s["points_allowed_tiers"])

    def row(label, m):
        rank = "—" if m["spearman"] is None else m["spearman"]
        return f"| {label} | {m['n']} | {m['mae']} | {m['rmse']} | {rank} |"

    lines = [
        "# D/ST fantasy points-per-game model: evaluation",
        "",
        f"Generated by `python -m ml.train_dst` on {r['trained_on']} "
        f"(scikit-learn {r['sklearn_version']}).",
        f"Data: {r['data']}, seasons {r['seasons'][0]}–{r['seasons'][1]}.",
        "",
        f"**Scoring (standard D/ST):** sack {s['sacks']}, interception {s['interceptions']}, "
        f"fumble recovery {s['fumble_recoveries']}, safety {s['safeties']}, "
        f"defensive/return TD {s['touchdowns']}, blocked kick {s['blocked_kicks']}; "
        f"points allowed {tiers}, 35+: {s['points_allowed_floor']:+d}.",
        "",
        f"**Chosen model:** `{r['chosen_model']}` (lowest mean MAE over validation "
        f"seasons {v0}–{v1}).",
        f"**Test season {r['split']['test_target_season']}:** MAE {t['model']['mae']} vs "
        f"{t['baseline_last_season']['mae']} for last season ({imp['vs_last_season']}% lower) "
        f"and {t['baseline_league_average']['mae']} for the league average "
        f"({imp['vs_league_average']}% lower).",
        "",
        "## Test season",
        "",
        "| Predictor | n | MAE | RMSE | Spearman |",
        "|---|---|---|---|---|",
        row("model", t["model"]),
        row("baseline · last season", t["baseline_last_season"]),
        row("baseline · league average", t["baseline_league_average"]),
        "",
        "League average predicts the same number for every team, so its Spearman is undefined.",
        "",
        f"## Backtest, target seasons {b['target_seasons'][0]}–{b['target_seasons'][1]}",
        "",
        "Each season scored by a model trained only on earlier seasons; averaged over "
        "seasons. One 32-team season is noisy, so this is the steadier comparison. "
        "The model was chosen on the last few of these seasons, so its row is slightly "
        "optimistic.",
        "",
        "| Predictor | Mean MAE | Mean Spearman |",
        "|---|---|---|",
        f"| model | {b['model']['mae']} | {b['model']['spearman']} |",
        f"| baseline · last season | {b['baseline_last_season']['mae']} | "
        f"{b['baseline_last_season']['spearman']} |",
        f"| baseline · league average | {b['baseline_league_average']['mae']} | — |",
        "",
        f"## Validation leaderboard (mean MAE, target seasons {v0}–{v1})",
        "",
        "| Model | Mean MAE | MAE by season |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{m['model']}` | {m['mae']} | "
        + ", ".join(f"{k}: {v}" for k, v in m["mae_by_season"].items()) + " |"
        for m in r["validation_leaderboard"]
    ]
    lines += [
        "",
        "## Limitations",
        "",
        "- D/ST output swings a lot from year to year (turnovers and return TDs are "
        "largely luck), so even a good model is only modestly better than the league average.",
        "- The model mostly improves *how many* points it expects (it pulls extreme seasons "
        "back toward the average); it ranks teams about as well as last season's finish does, "
        "no better. Use it as a sanity check on a D/ST's price, not as a ranking.",
        "- Uses past team box scores and next season's schedule only. It does not know "
        "about free agency, the draft, coaching changes, or injuries; the LLM's web search covers those.",
        "- Points allowed is the opponent's final score, including points the offense gave up "
        "(pick-sixes, fumble returns). Some leagues exclude those.",
        "- 32 teams a season means every test score rests on 32 predictions.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--first", type=int, default=TEAM_FIRST)
    parser.add_argument("--last", type=int, default=current_season() - 1)
    parser.add_argument("--out", type=Path, default=DEFAULT_DST_ARTIFACT_PATH)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-mlflow", action="store_true", help="train without MLflow tracking")
    parser.add_argument("--no-register", action="store_true",
                        help="track the run but don't register a model version")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args(argv)
    tracker = make_tracker(enabled=not args.no_mlflow, tracking_uri=args.tracking_uri,
                           run_name=f"train-dst-{args.first}-{args.last}",
                           experiment=EXPERIMENT, registered_model=EXPERIMENT)
    report = run(args.raw, args.first, args.last, args.out, args.reports, tracker,
                 register=not args.no_register)
    print(json.dumps({k: report.get(k) for k in (
        "chosen_model", "test_mae_improvement_pct", "mlflow_run_id", "mlflow_model_version")}))
    print(json.dumps(report["test"]))


if __name__ == "__main__":
    main()
