"""Repo-wide test setup, shared by tests/ and analysis_service/tests/."""

import pytest

from analysis_core import projection


@pytest.fixture(autouse=True)
def _no_projection_model(monkeypatch, tmp_path):
    """Run every test without the trained projection artifact.

    Tests should not depend on whatever model ``python -m ml.train`` last
    wrote. Projection tests opt in by pointing ``PROJECTION_MODEL_PATH`` at an
    artifact they build themselves (see ``tests/test_projection.py``).
    """
    monkeypatch.setenv(projection.ARTIFACT_ENV, str(tmp_path / "no-model.joblib"))
    projection.reset_projector()
    yield
    projection.reset_projector()
