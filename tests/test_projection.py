"""Serving the PPR projection model: loading, lookup, fallbacks, and how the
projection reaches the prompt and the API response."""

import logging

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LinearRegression

from analysis_core import anthropic_client, projection
from analysis_core.season import current_season
from analysis_core.projection import (
    ARTIFACT_FORMAT,
    Projection,
    Projector,
    normalize_name,
    project_player,
)
from app.analysis.client.base import Actor
from app.analysis.client.inprocess import InProcessAnalysisClient

N_FEATURES = 3
CLEAN = "RISK: 3/10\nVERDICT: Draft\nREASON: Fine value."


def _artifact(players=None, **overrides):
    # y = sum(features): easy to predict by hand.
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    model = LinearRegression().fit(X, X.sum(axis=1))
    art = {
        "format": ARTIFACT_FORMAT,
        "model": model,
        "features": ["a", "b", "c"],
        "sklearn_version": sklearn.__version__,
        "feature_season": current_season() - 1,
        "target_season": current_season(),
        "model_name": "linear",
        "test_mae": 2.4,
        "players": players if players is not None else {
            "kenneth walker": {
                "name": "Kenneth Walker III", "position": "RB", "team": "SEA",
                "ppg": 11.29, "games": 17, "features": [10.0, 2.0, 0.5],
            },
            "jamarr chase": {
                "name": "Ja'Marr Chase", "position": "WR", "team": "CIN",
                "ppg": 19.6, "games": 16, "features": [15.0, None, 3.0],
            },
            "negative guy": {
                "name": "Negative Guy", "position": "TE", "team": "KC",
                "ppg": 1.0, "games": 5, "features": [-9.0, -9.0, -9.0],
            },
        },
    }
    art.update(overrides)
    return art


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    """Point the service at an artifact this test controls."""
    path = tmp_path / "model.joblib"

    def write(**overrides):
        joblib.dump(_artifact(**overrides), path)
        monkeypatch.setenv(projection.ARTIFACT_ENV, str(path))
        projection.reset_projector()
        return path

    return write


# --------------------------------------------------------------------------- #
# Name keys
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, key", [
    ("Ja'Marr Chase", "jamarr chase"),
    ("Kenneth Walker III", "kenneth walker"),
    ("Amon-Ra St. Brown", "amonra st brown"),
    ("  MARVIN   HARRISON JR. ", "marvin harrison"),
    ("Michael Pittman Jr.", "michael pittman"),
    ("José Núñez", "jose nunez"),
])
def test_normalize_name(raw, key):
    assert normalize_name(raw) == key


# --------------------------------------------------------------------------- #
# Lookup and prediction
# --------------------------------------------------------------------------- #
def test_projects_known_player_with_typed_variant_of_name(model_path):
    model_path()
    p = project_player("kenneth walker iii")
    assert p == Projection(
        player="Kenneth Walker III", position="RB", team="SEA", season=current_season(),
        projected_ppg=12.5, last_ppg=11.29, last_games=17, typical_error=2.4,
    )


def test_missing_feature_is_passed_as_nan(model_path, monkeypatch):
    model_path()
    seen = {}
    projector = projection.get_projector()
    real = projector._model.predict

    def spy(X):
        seen["X"] = X
        return real(np.nan_to_num(X))

    monkeypatch.setattr(projector._model, "predict", spy)
    projector.project("Ja'Marr Chase")
    assert np.isnan(seen["X"][0, 1])


def test_unknown_player_gets_no_projection(model_path):
    model_path()
    assert project_player("Some Rookie") is None


def test_projection_is_never_negative(model_path):
    model_path()
    assert project_player("Negative Guy").projected_ppg == 0.0


def test_projector_is_loaded_once(model_path, monkeypatch):
    model_path()
    calls = []
    real_load = Projector.load
    monkeypatch.setattr(Projector, "load", classmethod(lambda cls, p, *a: calls.append(p) or real_load(p, *a)))
    project_player("Ja'Marr Chase")
    project_player("Kenneth Walker")
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Fallbacks: a bad or missing model disables projections, never errors
# --------------------------------------------------------------------------- #
def test_no_artifact_means_no_projection():
    # The autouse fixture points at a file that doesn't exist.
    assert projection.get_projector() is None
    assert project_player("Ja'Marr Chase") is None


def test_wrong_sklearn_version_disables_projection(model_path, caplog):
    model_path(sklearn_version="0.0.1")
    with caplog.at_level(logging.WARNING):
        assert project_player("Ja'Marr Chase") is None
    assert "retrain" in caplog.text


@pytest.mark.parametrize("fmt", ["something-else/1", None])
def test_unknown_artifact_format_disables_projection(model_path, fmt):
    model_path(format=fmt)
    assert project_player("Ja'Marr Chase") is None


def test_unreadable_artifact_disables_projection(tmp_path, monkeypatch):
    path = tmp_path / "junk.joblib"
    path.write_bytes(b"not a pickle")
    monkeypatch.setenv(projection.ARTIFACT_ENV, str(path))
    projection.reset_projector()
    assert project_player("Ja'Marr Chase") is None


def test_stale_season_artifact_is_not_served(model_path, caplog):
    model_path(target_season=current_season() - 1, feature_season=current_season() - 2)
    with caplog.at_level(logging.WARNING):
        assert project_player("Ja'Marr Chase") is None
        assert project_player("Kenneth Walker") is None
    assert caplog.text.count("retrain") == 1  # warned once, not per request


def test_projections_stop_when_the_season_rolls_over_mid_process(model_path, monkeypatch):
    model_path()
    assert project_player("Kenneth Walker") is not None
    rolled = current_season() + 1
    monkeypatch.setattr(projection, "current_season", lambda: rolled)
    assert project_player("Kenneth Walker") is None


def test_prediction_error_is_swallowed(model_path, monkeypatch):
    model_path()
    projector = projection.get_projector()

    def boom(X):
        raise RuntimeError("bad model")

    monkeypatch.setattr(projector._model, "predict", boom)
    assert project_player("Ja'Marr Chase") is None


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SAMPLE = Projection(
    player="Ja'Marr Chase", position="WR", team="CIN", season=2026,
    projected_ppg=18.4, last_ppg=19.6, last_games=16, typical_error=2.4,
)


def test_prompt_block_is_trusted_and_states_the_numbers():
    block = SAMPLE.prompt_block()
    assert block.startswith('<model_projection trust_level="trusted">')
    assert block.endswith("</model_projection>")
    assert "18.4 PPR points per game in 2026" in block
    assert "19.6 over 16 games in 2025" in block
    assert "±2.4" in block


def test_projection_sits_outside_the_untrusted_fence():
    content = anthropic_client._build_user_content(
        "Ja'Marr Chase", "some search result", SAMPLE
    )
    fence = content.index(anthropic_client.UNTRUSTED_OPEN)
    assert content.index("<model_projection") < fence
    assert content.count("<model_projection") == 1


def test_prompt_has_no_projection_block_without_a_projection():
    content = anthropic_client._build_user_content("Ja'Marr Chase", None, None)
    assert "<model_projection" not in content


def test_system_prompt_still_trips_no_leak_sentinel_on_projection_text():
    # The projection text must never look like a system-prompt leak.
    assert not anthropic_client._looks_like_prompt_leak(SAMPLE.prompt_block())


# --------------------------------------------------------------------------- #
# Callers
# --------------------------------------------------------------------------- #
@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(anthropic_client, "client", object())


def test_inprocess_client_passes_projection_and_returns_it(model_path, configured, monkeypatch):
    model_path()
    seen = {}

    def fake(name, projection=None, **_):
        seen["projection"] = projection
        return CLEAN

    monkeypatch.setattr(anthropic_client, "analyze_player", fake)
    verdict = InProcessAnalysisClient().analyze(
        player_name="Kenneth Walker III", actor=Actor(user_id="1")
    )
    assert seen["projection"].projected_ppg == 12.5
    assert verdict.projected_ppg == 12.5
    assert verdict.verdict == "Draft"


def test_inprocess_client_without_projection(configured, monkeypatch):
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name, projection=None, **_: CLEAN)
    verdict = InProcessAnalysisClient().analyze(
        player_name="Kenneth Walker III", actor=Actor(user_id="1")
    )
    assert verdict.projected_ppg is None
