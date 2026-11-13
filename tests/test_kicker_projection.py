"""Serving the kicker model: name lookup falls through from the player model,
fallbacks, the prompt wording, and the scoring label on the response."""

import logging

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LinearRegression

from analysis_core import anthropic_client, projection
from analysis_core.projection import KICKER_ARTIFACT_FORMAT, Projection, project_player
from analysis_core.season import current_season
from app.analysis.client.base import Actor
from app.analysis.client.inprocess import InProcessAnalysisClient
from tests.test_projection import _artifact

CLEAN = "RISK: 3/10\nVERDICT: Pass\nREASON: Kickers go last."


def _kicker_artifact(**overrides):
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    art = {
        "format": KICKER_ARTIFACT_FORMAT,
        "model": LinearRegression().fit(X, X.sum(axis=1)),
        "features": ["a", "b"],
        "sklearn_version": sklearn.__version__,
        "feature_season": current_season() - 1,
        "target_season": current_season(),
        "model_name": "linear",
        "test_mae": 1.4,
        "players": {
            "brandon aubrey": {"name": "Brandon Aubrey", "position": "K", "team": "DAL",
                               "ppg": 10.59, "games": 17, "features": [5.0, 4.0]},
            # Same key as a player in the player artifact: the player model wins.
            "kenneth walker": {"name": "Kenneth Walker", "position": "K", "team": "XXX",
                               "ppg": 1.0, "games": 4, "features": [0.0, 0.0]},
        },
    }
    art.update(overrides)
    return art


@pytest.fixture
def models(tmp_path, monkeypatch):
    def write(kicker=True, player=True, **kicker_overrides):
        if player:
            path = tmp_path / "player.joblib"
            joblib.dump(_artifact(), path)
            monkeypatch.setenv(projection.ARTIFACT_ENV, str(path))
        if kicker:
            path = tmp_path / "kicker.joblib"
            joblib.dump(_kicker_artifact(**kicker_overrides), path)
            monkeypatch.setenv(projection.KICKER_ARTIFACT_ENV, str(path))
        projection.reset_projector()
    return write


def test_kicker_is_projected_by_the_kicker_model(models):
    models()
    assert project_player("brandon aubrey") == Projection(
        player="Brandon Aubrey", position="K", team="DAL", season=current_season(),
        projected_ppg=9.0, last_ppg=10.59, last_games=17, typical_error=1.4,
    )


def test_player_model_wins_a_name_clash(models):
    models()
    assert project_player("Kenneth Walker III").position == "RB"


def test_kicker_works_without_a_player_artifact(models):
    models(player=False)
    assert project_player("Brandon Aubrey").position == "K"


def test_unknown_name_gets_no_projection(models):
    models()
    assert project_player("Some Rookie") is None


def test_no_kicker_artifact_means_no_kicker_projection(models):
    models(kicker=False)
    assert project_player("Brandon Aubrey") is None


def test_stale_kicker_artifact_warns_once_and_names_its_command(models, caplog):
    models(target_season=current_season() - 1, feature_season=current_season() - 2)
    with caplog.at_level(logging.WARNING):
        assert project_player("Brandon Aubrey") is None
        assert project_player("Brandon Aubrey") is None
    assert caplog.text.count("retrain") == 1
    assert "ml.train_kicker" in caplog.text


def test_wrong_format_kicker_artifact_is_ignored(models):
    models(format="dst-ppg-projection/1")
    assert project_player("Brandon Aubrey") is None


def test_kicker_prompt_block_names_kicker_scoring():
    block = Projection(
        player="Brandon Aubrey", position="K", team="DAL", season=2026,
        projected_ppg=9.0, last_ppg=10.6, last_games=17, typical_error=1.4,
    ).prompt_block()
    assert "9.0 standard kicker fantasy points per game in 2026" in block
    assert "PPR" not in block
    assert not anthropic_client._looks_like_prompt_leak(block)


def test_inprocess_client_labels_kicker_scoring(models, monkeypatch):
    models()
    monkeypatch.setattr(anthropic_client, "client", object())
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name, **_: CLEAN)
    verdict = InProcessAnalysisClient().analyze(
        player_name="Brandon Aubrey", actor=Actor(user_id="1"))
    assert (verdict.projected_ppg, verdict.projection_scoring) == (9.0, "K")
