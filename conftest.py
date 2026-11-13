"""Repo-wide test setup, shared by tests/ and analysis_service/tests/."""

import pytest

from analysis_core import projection, retrieval


@pytest.fixture(autouse=True)
def _no_projection_model(monkeypatch, tmp_path):
    """Run every test without the trained projection artifacts (player and D/ST).

    Tests should not depend on whatever model ``python -m ml.train`` last
    wrote. Projection tests opt in by pointing ``PROJECTION_MODEL_PATH`` at an
    artifact they build themselves (see ``tests/test_projection.py``).
    """
    monkeypatch.setenv(projection.ARTIFACT_ENV, str(tmp_path / "no-model.joblib"))
    monkeypatch.setenv(projection.DST_ARTIFACT_ENV, str(tmp_path / "no-dst-model.joblib"))
    # Any MLflow tracking a test triggers goes to a throwaway store, never
    # ml/mlflow.db.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    # No retrieval store either: tests that exercise RAG build their own.
    monkeypatch.delenv(retrieval.DATABASE_ENV, raising=False)
    projection.reset_projector()
    retrieval.reset_retriever()
    yield
    projection.reset_projector()
    retrieval.reset_retriever()
