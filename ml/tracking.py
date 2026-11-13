"""Experiment tracking for ``python -m ml.train``, backed by MLflow.

One training run becomes one MLflow *parent* run, holding the data window,
the split, the chosen model's test scores, the report files, and the final
model, which is registered as a new version of ``ppr-projection``. Each model
configuration tried during selection becomes a nested *child* run with its
hyperparameters and validation scores, so runs can be compared side by side in
``mlflow ui``.

MLflow is a training-only dependency (``requirements-dev.txt``). If it is not
installed, or training runs with ``--no-mlflow``, :class:`NullTracker` keeps
the same interface and records nothing, so training works the same either way.

By default runs go to a SQLite database and an artifact folder under ``ml/``
(both gitignored). Set ``MLFLOW_TRACKING_URI`` to use a tracking server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

log = logging.getLogger(__name__)

ML_DIR = Path(__file__).parent
DEFAULT_DB = ML_DIR / "mlflow.db"
DEFAULT_ARTIFACTS = ML_DIR / "mlartifacts"
EXPERIMENT = "ppr-projection"
REGISTERED_MODEL = "ppr-projection"


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{DEFAULT_DB.as_posix()}"


def artifact_location(tracking_uri: str) -> Optional[str]:
    """For a local SQLite store, keep artifacts in ``mlartifacts/`` next to
    the database (MLflow would otherwise use ``./mlruns`` under whatever
    directory training ran from). A tracking server decides its own."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    folder = Path(tracking_uri[len(prefix):]).parent / DEFAULT_ARTIFACTS.name
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve().as_uri()


def flatten(metrics: Mapping, prefix: str = "") -> dict:
    """``{"test": {"model": {"mae": 2.4}}}`` -> ``{"test/model/mae": 2.4}``.

    Keeps numbers only (``n`` counts included), so a nested report section can
    be logged in one call.
    """
    out = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


class NullTracker:
    """Same interface as :class:`MlflowTracker`, records nothing."""

    run_id: Optional[str] = None
    model_version: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_params(self, params: Mapping) -> None:
        pass

    def log_metrics(self, metrics: Mapping) -> None:
        pass

    def log_candidate(self, name: str, params: Mapping, metrics: Mapping) -> None:
        pass

    def log_artifacts(self, paths: Iterable[Path]) -> None:
        pass

    def log_model(self, model, X_example: np.ndarray, *, register: bool) -> None:
        pass


class MlflowTracker(NullTracker):
    def __init__(self, mlflow, *, tracking_uri: str, experiment: str = EXPERIMENT,
                 run_name: Optional[str] = None, registered_model: str = REGISTERED_MODEL):
        self._mlflow = mlflow
        self._registered_model = registered_model
        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._run_name = run_name
        self._run = None

    def __enter__(self):
        mlflow = self._mlflow
        mlflow.set_tracking_uri(self._tracking_uri)
        if mlflow.get_experiment_by_name(self._experiment) is None:
            mlflow.create_experiment(
                self._experiment, artifact_location=artifact_location(self._tracking_uri)
            )
        mlflow.set_experiment(self._experiment)
        self._run = mlflow.start_run(run_name=self._run_name)
        self.run_id = self._run.info.run_id
        return self

    def __exit__(self, exc_type, *exc):
        self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        return False

    def log_params(self, params: Mapping) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping) -> None:
        self._mlflow.log_metrics(flatten(metrics))

    def log_candidate(self, name: str, params: Mapping, metrics: Mapping) -> None:
        with self._mlflow.start_run(run_name=name, nested=True):
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow.log_metrics(flatten(metrics, "validation"))

    def log_artifacts(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self._mlflow.log_artifact(str(path))

    def log_model(self, model, X_example: np.ndarray, *, register: bool) -> None:
        from mlflow.models import infer_signature

        example = X_example[:5]
        info = self._mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            registered_model_name=self._registered_model if register else None,
            skops_trusted_types=skops_trusted_types(model),
        )
        version = getattr(info, "registered_model_version", None)
        self.model_version = None if version is None else str(version)


def skops_trusted_types(model) -> list[str]:
    """Types skops would refuse to load from this model's file.

    MLflow saves scikit-learn models with skops, which (unlike pickle) only
    loads types it has been told to trust. The tree internals of
    HistGradientBoostingRegressor aren't on skops' default list. These are
    the types of a model this script just trained, so trusting exactly those
    is safe; anything else in the file would still be rejected.
    """
    import skops.io as sio

    return sio.get_untrusted_types(data=sio.dumps(model))


def make_tracker(*, enabled: bool = True, tracking_uri: Optional[str] = None,
                 run_name: Optional[str] = None, experiment: str = EXPERIMENT,
                 registered_model: str = REGISTERED_MODEL) -> NullTracker:
    """An MLflow tracker, or a :class:`NullTracker` when disabled or when
    MLflow isn't installed (logged, not raised)."""
    if not enabled:
        return NullTracker()
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401  (registers the flavor)
    except ImportError:
        log.warning("mlflow is not installed; training without experiment tracking. "
                    "pip install -r requirements-dev.txt to enable it.")
        return NullTracker()
    return MlflowTracker(mlflow, tracking_uri=tracking_uri or default_tracking_uri(),
                         experiment=experiment, run_name=run_name,
                         registered_model=registered_model)
