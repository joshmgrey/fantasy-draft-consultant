"""Train, evaluate, and export the next-season PPR points-per-game model.

    python -m ml.download
    python -m ml.train

Splits are by season, never random, so the model is always scored on a season
it has not seen:

* train       target seasons first+1 .. last-2
* validation  target season  last-1   (picks the model)
* test        target season  last     (scored once, reported)

Every candidate must beat the naive baseline "next season = last season's
PPR points per game". The winner is refit on every season and exported with
the latest season's feature rows, so the services can project the upcoming
season without pandas.

Every run is tracked in MLflow when it is installed (see ``ml/tracking.py``):
one child run per candidate, the test scores and reports on the parent run,
and the final model registered as a new ``ppr-projection`` version.

    python -m ml.train --no-mlflow     # train without tracking
    mlflow ui --backend-store-uri sqlite:///ml/mlflow.db
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_core.projection import ARTIFACT_FORMAT, DEFAULT_ARTIFACT_PATH
from analysis_core.season import current_season

from . import features as F
from .tracking import NullTracker, make_tracker

ML_DIR = Path(__file__).parent
RAW_DIR = ML_DIR / "data" / "raw"
REPORT_DIR = ML_DIR / "reports"

SEED = 7


@dataclass
class Candidate:
    name: str
    build: Callable[[], object]
    params: dict


def candidates() -> list[Candidate]:
    def ridge(alpha):
        return lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)
        )

    def hgb(lr, leaf, depth):
        return lambda: HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=lr, max_iter=400,
            min_samples_leaf=leaf, max_depth=depth, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        )

    out = [
        Candidate(f"ridge(alpha={a})", ridge(a), {"model_type": "ridge", "alpha": a})
        for a in (1.0, 10.0, 100.0)
    ]
    for lr in (0.03, 0.06):
        for leaf in (20, 50):
            for depth in (3, 5):
                out.append(Candidate(
                    f"hgb(lr={lr},leaf={leaf},depth={depth})", hgb(lr, leaf, depth),
                    {"model_type": "hist_gradient_boosting", "learning_rate": lr,
                     "min_samples_leaf": leaf, "max_depth": depth, "max_iter": 400,
                     "loss": "absolute_error"},
                ))
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rank = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "spearman": round(float(rank), 3),
    }


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Overall, per-position, and fantasy-relevant (>= 10 PPR/game) scores."""
    y = df["target_ppg"].to_numpy()
    out = {"overall": metrics(y, pred)}
    for pos in F.POSITIONS:
        mask = (df["position"] == pos).to_numpy()
        if mask.any():
            out[pos] = metrics(y[mask], pred[mask])
    relevant = (df["ppg"] >= 10).to_numpy()
    if relevant.any():
        out["relevant"] = metrics(y[relevant], pred[relevant])
    return out


def baseline(df: pd.DataFrame) -> np.ndarray:
    return df["ppg"].to_numpy()


def split(table: pd.DataFrame, last_target: int):
    train = table[table["target_season"] <= last_target - 2]
    val = table[table["target_season"] == last_target - 1]
    test = table[table["target_season"] == last_target]
    return train, val, test


def select_model(
    train: pd.DataFrame, val: pd.DataFrame, tracker: Optional[NullTracker] = None
) -> tuple[Candidate, list[dict]]:
    tracker = tracker or NullTracker()
    X_tr, y_tr = F.feature_matrix(train), train["target_ppg"].to_numpy()
    X_val, y_val = F.feature_matrix(val), val["target_ppg"].to_numpy()
    results = []
    best: Optional[tuple[float, Candidate]] = None
    for cand in candidates():
        model = cand.build().fit(X_tr, y_tr)
        score = metrics(y_val, model.predict(X_val))
        results.append({"model": cand.name, **score})
        tracker.log_candidate(cand.name, cand.params, score)
        if best is None or score["mae"] < best[0]:
            best = (score["mae"], cand)
    return best[1], results


def serving_rows(rows: pd.DataFrame, feature_season: int) -> dict:
    """Latest-season feature rows keyed by normalized name.

    When two players share a name, keep the one who played more games.
    """
    latest = rows[(rows["season"] == feature_season) & (rows["games"] >= F.MIN_GAMES)]
    latest = latest.sort_values("games", ascending=False).drop_duplicates("name_key")
    players = {}
    for rec in latest.itertuples(index=False):
        vec = [None if pd.isna(v) else float(v) for v in (getattr(rec, f) for f in F.FEATURES)]
        players[rec.name_key] = {
            "name": rec.player_display_name,
            "position": rec.position,
            "team": rec.recent_team,
            "ppg": round(float(rec.ppg), 2),
            "games": int(rec.games),
            "features": vec,
        }
    return players


def run(raw_dir: Path, first: int, last: int, out: Path, report_dir: Path,
        tracker: Optional[NullTracker] = None, *, register: bool = True) -> dict:
    with tracker or NullTracker() as tr:
        return _run(raw_dir, first, last, out, report_dir, tr, register)


def _run(raw_dir, first, last, out, report_dir, tracker, register) -> dict:
    stats = F.load_season_stats(raw_dir, range(first, last + 1))
    rows = F.season_rows(stats, F.load_players(raw_dir))
    table = F.training_table(rows)

    train, val, test = split(table, last)
    tracker.log_params({
        "first_season": first, "last_season": last,
        "train_target_seasons": f"{first + 1}-{last - 2}",
        "validation_target_season": last - 1, "test_target_season": last,
        "n_features": len(F.FEATURES), "min_games": F.MIN_GAMES,
        "rows_train": len(train), "rows_validation": len(val), "rows_test": len(test),
        "seed": SEED, "sklearn_version": sklearn.__version__,
    })
    best, leaderboard = select_model(train, val, tracker)

    # Score the chosen configuration once on the untouched test season,
    # trained on train + validation.
    fit_rows = pd.concat([train, val])
    model = best.build().fit(F.feature_matrix(fit_rows), fit_rows["target_ppg"].to_numpy())
    test_scores = {
        "model": evaluate(test, model.predict(F.feature_matrix(test))),
        "baseline": evaluate(test, baseline(test)),
    }
    base_mae = test_scores["baseline"]["overall"]["mae"]
    model_mae = test_scores["model"]["overall"]["mae"]
    improvement = round(100 * (base_mae - model_mae) / base_mae, 1)

    # Refit on every labeled season for serving.
    final = best.build().fit(F.feature_matrix(table), table["target_ppg"].to_numpy())

    report = {
        "trained_on": date.today().isoformat(),
        "data": "nflverse stats_player_reg (CC-BY 4.0)",
        "seasons": [first, last],
        "rows": {"train": len(train), "validation": len(val), "test": len(test), "final": len(table)},
        "split": {
            "train_target_seasons": [first + 1, last - 2],
            "validation_target_season": last - 1,
            "test_target_season": last,
        },
        "chosen_model": best.name,
        "validation_leaderboard": sorted(leaderboard, key=lambda r: r["mae"]),
        "test": test_scores,
        "test_mae_improvement_vs_baseline_pct": improvement,
        "sklearn_version": sklearn.__version__,
        "mlflow_run_id": tracker.run_id,
    }

    artifact = {
        "format": ARTIFACT_FORMAT,
        "model": final,
        "features": list(F.FEATURES),
        "sklearn_version": sklearn.__version__,
        "feature_season": last,
        "target_season": last + 1,
        "model_name": best.name,
        "test_mae": model_mae,
        # Ties a deployed artifact back to the MLflow run that produced it.
        "mlflow_run_id": tracker.run_id,
        "players": serving_rows(rows, last),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (report_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")

    tracker.log_params({"chosen_model": best.name, **{f"chosen_{k}": v for k, v in best.params.items()}})
    tracker.log_metrics({
        "test": test_scores,
        "test_mae_improvement_vs_baseline_pct": improvement,
        "rows_final": len(table),
    })
    tracker.log_artifacts([report_dir / "REPORT.md", report_dir / "metrics.json", out])
    tracker.log_model(final, F.feature_matrix(table), register=register)
    report["mlflow_model_version"] = tracker.model_version
    return report


def render_report(r: dict) -> str:
    def row(label, m):
        return f"| {label} | {m['n']} | {m['mae']} | {m['rmse']} | {m['spearman']} |"

    t = r["test"]
    lines = [
        "# PPR points-per-game model: evaluation",
        "",
        f"Generated by `python -m ml.train` on {r['trained_on']} (scikit-learn {r['sklearn_version']}).",
        f"Data: {r['data']}, seasons {r['seasons'][0]}–{r['seasons'][1]}.",
        "",
        f"**Chosen model:** `{r['chosen_model']}` (lowest validation MAE).",
        f"**Test season {r['split']['test_target_season']}:** MAE {t['model']['overall']['mae']} vs "
        f"{t['baseline']['overall']['mae']} for the last-season baseline "
        f"({r['test_mae_improvement_vs_baseline_pct']}% lower).",
        "",
        "## Test season, model vs baseline",
        "",
        "| Slice | Model n | MAE | RMSE | Spearman |",
        "|---|---|---|---|---|",
    ]
    for key in ["overall", *F.POSITIONS, "relevant"]:
        if key in t["model"]:
            lines.append(row(f"model · {key}", t["model"][key]))
            lines.append(row(f"baseline · {key}", t["baseline"][key]))
    lines += [
        "",
        "`relevant` = players who scored at least 10 PPR points per game the season before.",
        "",
        f"## Validation leaderboard (target season {r['split']['validation_target_season']})",
        "",
        "| Model | MAE | RMSE | Spearman |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{m['model']}` | {m['mae']} | {m['rmse']} | {m['spearman']} |"
        for m in r["validation_leaderboard"]
    ]
    lines += [
        "",
        "## Limitations",
        "",
        "- Uses only past box-score stats. It does not know about offseason trades, "
        "depth-chart changes, coaching changes, or rookies; the LLM's web search covers those.",
        f"- Trained only on players who played at least {F.MIN_GAMES} games in both seasons, "
        "so it predicts how a player scores *when he plays*, not whether he stays healthy.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--first", type=int, default=2012)
    parser.add_argument("--last", type=int, default=current_season() - 1)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-mlflow", action="store_true", help="train without MLflow tracking")
    parser.add_argument("--no-register", action="store_true",
                        help="track the run but don't register a model version")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ml/mlflow.db)")
    args = parser.parse_args(argv)
    tracker = make_tracker(enabled=not args.no_mlflow, tracking_uri=args.tracking_uri,
                           run_name=f"train-{args.first}-{args.last}")
    report = run(args.raw, args.first, args.last, args.out, args.reports, tracker,
                 register=not args.no_register)
    print(json.dumps({k: report.get(k) for k in (
        "chosen_model", "test_mae_improvement_vs_baseline_pct",
        "mlflow_run_id", "mlflow_model_version")}))
    print(json.dumps(report["test"]["model"]["overall"]), json.dumps(report["test"]["baseline"]["overall"]))


if __name__ == "__main__":
    main()
