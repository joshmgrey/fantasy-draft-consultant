"""MLflow experiment tracking for ml/train.py.

The end-to-end test trains on small synthetic data against a throwaway
SQLite tracking store, then reads the runs, artifacts, and registered model
back through the MLflow client.
"""

import builtins
import logging
from pathlib import Path

import joblib
import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

from ml import features as F  # noqa: E402
from ml import train  # noqa: E402
from ml import tracking  # noqa: E402
from ml.tracking import MlflowTracker, NullTracker, make_tracker  # noqa: E402
from tests.test_ml_features import write_synthetic_raw  # noqa: E402


def sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_flatten_nests_with_slashes_and_keeps_numbers_only():
    nested = {"test": {"model": {"overall": {"n": 10, "mae": 2.4}}, "note": "x"},
              "flag": True, "pct": 10.9}
    assert tracking.flatten(nested) == {
        "test/model/overall/n": 10.0,
        "test/model/overall/mae": 2.4,
        "pct": 10.9,
    }


def test_sqlite_artifacts_live_next_to_the_database(tmp_path):
    location = tracking.artifact_location(sqlite_uri(tmp_path / "sub" / "mlflow.db"))
    assert location == (tmp_path / "sub" / "mlartifacts").resolve().as_uri()
    assert (tmp_path / "sub" / "mlartifacts").is_dir()


def test_tracking_server_chooses_its_own_artifact_location():
    assert tracking.artifact_location("http://mlflow.internal:5000") is None


def test_default_uri_honors_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid")
    assert tracking.default_tracking_uri() == "http://example.invalid"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert tracking.default_tracking_uri().endswith("ml/mlflow.db")


def test_disabled_tracking_is_a_null_tracker():
    assert type(make_tracker(enabled=False)) is NullTracker


def test_missing_mlflow_falls_back_to_null_tracker(monkeypatch, caplog):
    real_import = builtins.__import__

    def no_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlflow)
    with caplog.at_level(logging.WARNING):
        tracker = make_tracker()
    assert type(tracker) is NullTracker
    assert "mlflow is not installed" in caplog.text


def test_null_tracker_records_nothing():
    with NullTracker() as t:
        t.log_params({"a": 1})
        t.log_metrics({"b": 2})
        t.log_candidate("c", {}, {})
        t.log_artifacts([Path("nope")])
        t.log_model(object(), np.zeros((1, 1)), register=True)
    assert t.run_id is None and t.model_version is None


def test_failed_training_marks_the_run_failed(tmp_path):
    uri = sqlite_uri(tmp_path / "mlflow.db")
    tracker = make_tracker(tracking_uri=uri, run_name="boom")
    with pytest.raises(RuntimeError):
        with tracker:
            raise RuntimeError("training crashed")
    run = MlflowClient(tracking_uri=uri).get_run(tracker.run_id)
    assert run.info.status == "FAILED"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tracked_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("tracked")
    raw = root / "raw"
    raw.mkdir()
    write_synthetic_raw(raw, n_players=80)
    uri = sqlite_uri(root / "mlflow.db")
    tracker = make_tracker(tracking_uri=uri, run_name="train-test")
    assert isinstance(tracker, MlflowTracker)
    out = root / "model.joblib"
    report = train.run(raw, 2012, 2019, out, root / "reports", tracker)
    return {"uri": uri, "report": report, "tracker": tracker, "out": out, "raw": raw}


def test_parent_run_holds_split_choice_and_test_scores(tracked_run):
    client = MlflowClient(tracking_uri=tracked_run["uri"])
    run = client.get_run(tracked_run["tracker"].run_id)
    report = tracked_run["report"]

    assert run.info.status == "FINISHED"
    assert run.info.run_name == "train-test"
    params = run.data.params
    assert params["test_target_season"] == "2019"
    assert params["train_target_seasons"] == "2013-2017"
    assert params["chosen_model"] == report["chosen_model"]
    assert params["n_features"] == str(len(F.FEATURES))

    metrics = run.data.metrics
    assert metrics["test/model/overall/mae"] == report["test"]["model"]["overall"]["mae"]
    assert metrics["test/baseline/overall/mae"] == report["test"]["baseline"]["overall"]["mae"]
    assert metrics["test_mae_improvement_vs_baseline_pct"] == report["test_mae_improvement_vs_baseline_pct"]


def test_one_child_run_per_candidate(tracked_run):
    client = MlflowClient(tracking_uri=tracked_run["uri"])
    parent = client.get_run(tracked_run["tracker"].run_id)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    assert sorted(r.info.run_name for r in children) == sorted(c.name for c in train.candidates())
    by_name = {r.info.run_name: r for r in children}
    best = by_name[tracked_run["report"]["chosen_model"]]
    assert best.data.metrics["validation/mae"] == min(
        r.data.metrics["validation/mae"] for r in children
    )
    assert "model_type" in best.data.params


def test_reports_and_serving_artifact_are_logged(tracked_run):
    client = MlflowClient(tracking_uri=tracked_run["uri"])
    names = {a.path for a in client.list_artifacts(tracked_run["tracker"].run_id)}
    assert {"REPORT.md", "metrics.json", "model.joblib"} <= names


def test_run_id_is_stamped_on_the_report_and_serving_artifact(tracked_run):
    run_id = tracked_run["tracker"].run_id
    assert tracked_run["report"]["mlflow_run_id"] == run_id
    assert joblib.load(tracked_run["out"])["mlflow_run_id"] == run_id


def test_model_is_registered_and_reloads_with_same_predictions(tracked_run):
    assert tracked_run["report"]["mlflow_model_version"] == "1"
    loaded = mlflow.sklearn.load_model(f"models:/{tracking.REGISTERED_MODEL}/1")

    served = joblib.load(tracked_run["out"])
    X = np.array([
        [np.nan if v is None else v for v in row["features"]]
        for row in list(served["players"].values())[:10]
    ], dtype=float)
    np.testing.assert_allclose(loaded.predict(X), served["model"].predict(X))


def test_no_register_skips_the_model_registry(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_synthetic_raw(raw, n_players=40)
    uri = sqlite_uri(tmp_path / "mlflow.db")
    tracker = make_tracker(tracking_uri=uri)
    report = train.run(raw, 2012, 2019, tmp_path / "m.joblib", tmp_path / "r", tracker,
                       register=False)
    assert report["mlflow_model_version"] is None
    assert MlflowClient(tracking_uri=uri).search_registered_models() == []


def test_cli_no_mlflow_flag(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_synthetic_raw(raw, n_players=40)
    seen = {}
    real = train.make_tracker

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(train, "make_tracker", spy)
    train.main(["--raw", str(raw), "--first", "2012", "--last", "2019", "--no-mlflow",
                "--out", str(tmp_path / "m.joblib"), "--reports", str(tmp_path / "r")])
    assert seen["enabled"] is False
    assert joblib.load(tmp_path / "m.joblib")["mlflow_run_id"] is None
