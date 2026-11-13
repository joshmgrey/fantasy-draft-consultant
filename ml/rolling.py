"""Model selection and evaluation for the small-sample projection models
(D/ST in ``ml/train_dst.py``, kickers in ``ml/train_kicker.py``).

With ~32–40 rows a season, one validation season is too noisy to pick a model
on, so:

* validation  target seasons last-4 .. last-1, each scored by a model trained
              only on earlier seasons (rolling origin); candidates are ranked
              by their mean MAE across those four seasons
* test        target season last, trained on every earlier season, scored once
* backtest    every target season from first+8 .. last, same rolling scheme,
              averaged: steadier than any one season

The model is reported against two baselines: "same as last season" and
"league average" (the mean of every earlier row). Tables need the columns
``target_season``, ``ppg`` (season t) and ``target_ppg`` (season t + 1).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .tracking import NullTracker
from .train import Candidate, candidates
from .train import metrics as _metrics

VALIDATION_SEASONS = 4
# The long backtest starts once this many seasons are available to train on.
BACKTEST_MIN_TRAIN_SEASONS = 8

Matrix = Callable[[pd.DataFrame], np.ndarray]


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


def fit(cand: Candidate, rows: pd.DataFrame, matrix: Matrix):
    return cand.build().fit(matrix(rows), rows["target_ppg"].to_numpy())


def select_model(table: pd.DataFrame, last: int, matrix: Matrix,
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
            pred = fit(cand, train, matrix).predict(matrix(val))
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


def backtest(cand: Candidate, table: pd.DataFrame, first: int, last: int,
             matrix: Matrix) -> dict:
    """Mean MAE and Spearman over every target season from
    ``first + BACKTEST_MIN_TRAIN_SEASONS`` to ``last``, each scored by a model
    trained only on earlier seasons.

    The chosen model was picked on the last few of these seasons, so its
    numbers here are slightly optimistic; the baselines' are not.
    """
    seasons = [s for s in range(first + BACKTEST_MIN_TRAIN_SEASONS, last + 1)
               if (table["target_season"] == s).any()]
    per = {}
    for season in seasons:
        train = table[table["target_season"] < season]
        test = table[table["target_season"] == season]
        y = test["target_ppg"].to_numpy()
        preds = {"model": fit(cand, train, matrix).predict(matrix(test)),
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


def evaluate(table: pd.DataFrame, first: int, last: int, matrix: Matrix,
             tracker: Optional[NullTracker] = None) -> tuple[Candidate, object, dict]:
    """Pick, score, and refit a model. Returns the chosen candidate, the model
    refit on every row (for serving), and the report sections."""
    if last - first < BACKTEST_MIN_TRAIN_SEASONS:
        raise ValueError(
            f"seasons {first}-{last} are too few: rolling validation and the backtest "
            f"need at least {BACKTEST_MIN_TRAIN_SEASONS + 1} seasons (first <= "
            f"{last - BACKTEST_MIN_TRAIN_SEASONS} for last={last})"
        )
    best, leaderboard = select_model(table, last, matrix, tracker)

    train = table[table["target_season"] < last]
    test = table[table["target_season"] == last]
    y_test = test["target_ppg"].to_numpy()
    test_scores = {"model": metrics(y_test, fit(best, train, matrix).predict(matrix(test)))}
    for name, pred in baselines(train, test).items():
        test_scores[f"baseline_{name}"] = metrics(y_test, pred)
    model_mae = test_scores["model"]["mae"]

    def improvement(name):
        base = test_scores[f"baseline_{name}"]["mae"]
        return round(100 * (base - model_mae) / base, 1)

    sections = {
        "rows": {"train": len(train), "test": len(test), "final": len(table)},
        "split": {
            "validation_target_seasons": [last - VALIDATION_SEASONS, last - 1],
            "test_target_season": last,
        },
        "chosen_model": best.name,
        "validation_leaderboard": sorted(leaderboard, key=lambda r: r["mae"]),
        "test": test_scores,
        "backtest": backtest(best, table, first, last, matrix),
        "test_mae_improvement_pct": {
            "vs_last_season": improvement("last_season"),
            "vs_league_average": improvement("league_average"),
        },
    }
    return best, fit(best, table, matrix), sections


def render_results(r: dict, unit: str) -> list[str]:
    """Markdown for the chosen model, test season, backtest, and validation
    leaderboard. ``unit`` names one row ("team", "kicker")."""
    t = r["test"]
    b = r["backtest"]
    imp = r["test_mae_improvement_pct"]
    v0, v1 = r["split"]["validation_target_seasons"]

    def row(label, m):
        rank = "—" if m["spearman"] is None else m["spearman"]
        return f"| {label} | {m['n']} | {m['mae']} | {m['rmse']} | {rank} |"

    lines = [
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
        f"League average predicts the same number for every {unit}, so its Spearman is undefined.",
        "",
        f"## Backtest, target seasons {b['target_seasons'][0]}–{b['target_seasons'][1]}",
        "",
        "Each season scored by a model trained only on earlier seasons; averaged over "
        "seasons. One season is noisy, so this is the steadier comparison. "
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
    return lines


def save_and_track(*, artifact: dict, report: dict, markdown: str, out, report_dir,
                   stem: str, tracker: NullTracker, final_model, X, register: bool) -> None:
    """Write the serving artifact and ``<stem>_metrics.json`` /
    ``<STEM>_REPORT.md``, and log them, the scores, and the model to the tracker."""
    import json

    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{stem}_metrics.json"
    md_path = report_dir / f"{stem.upper()}_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    best = report["chosen_model"]
    tracker.log_params({"chosen_model": best})
    tracker.log_metrics({"test": report["test"],
                         "test_mae_improvement_pct": report["test_mae_improvement_pct"],
                         "backtest": report["backtest"]})
    tracker.log_artifacts([md_path, json_path, out])
    tracker.log_model(final_model, X, register=register)
    report["mlflow_model_version"] = tracker.model_version


def serving_vector(rec, features) -> list:
    return [None if pd.isna(v) else float(v) for v in (getattr(rec, f) for f in features)]
