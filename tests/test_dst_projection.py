"""Serving the D/ST model: routing a team-defense name to it, fallbacks, the
prompt wording, and the scoring label on the response."""

import joblib
import numpy as np
import pytest
import sklearn
from sklearn.linear_model import LinearRegression

from analysis_core import anthropic_client, projection
from analysis_core.projection import DST_ARTIFACT_FORMAT, Projection, project_player
from analysis_core.season import current_season
from app.analysis.client.base import Actor
from app.analysis.client.inprocess import InProcessAnalysisClient
from tests.test_projection import _artifact

CLEAN = "RISK: 4/10\nVERDICT: Pass\nREASON: Streamable at its ADP."


def _dst_artifact(**overrides):
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    art = {
        "format": DST_ARTIFACT_FORMAT,
        "model": LinearRegression().fit(X, X.sum(axis=1)),
        "features": ["a", "b"],
        "sklearn_version": sklearn.__version__,
        "feature_season": current_season() - 1,
        "target_season": current_season(),
        "model_name": "linear",
        "test_mae": 1.6,
        "players": {
            "PIT": {"name": "Pittsburgh Steelers D/ST", "position": "D/ST", "team": "PIT",
                    "ppg": 6.88, "games": 17, "features": [4.0, 2.5]},
        },
    }
    art.update(overrides)
    return art


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Write a player artifact and a D/ST artifact this test controls."""
    def write(dst=True, player=True, **dst_overrides):
        if player:
            path = tmp_path / "player.joblib"
            joblib.dump(_artifact(), path)
            monkeypatch.setenv(projection.ARTIFACT_ENV, str(path))
        if dst:
            path = tmp_path / "dst.joblib"
            joblib.dump(_dst_artifact(**dst_overrides), path)
            monkeypatch.setenv(projection.DST_ARTIFACT_ENV, str(path))
        projection.reset_projector()
    return write


@pytest.mark.parametrize("name", ["Steelers D/ST", "PIT DST", "Pittsburgh Steelers Defense"])
def test_team_defense_is_projected_by_the_dst_model(models, name):
    models()
    assert project_player(name) == Projection(
        player="Pittsburgh Steelers D/ST", position="D/ST", team="PIT",
        season=current_season(), projected_ppg=6.5, last_ppg=6.88, last_games=17,
        typical_error=1.6,
    )


def test_players_still_go_to_the_player_model(models):
    models()
    assert project_player("Kenneth Walker III").position == "RB"


def test_team_not_in_artifact_gets_no_projection(models):
    models()
    assert project_player("Bills D/ST") is None


def test_dst_name_never_falls_back_to_the_player_model(models):
    models(dst=False)
    assert project_player("Steelers D/ST") is None


def test_player_projection_works_without_a_dst_artifact(models):
    models(dst=False)
    assert project_player("Kenneth Walker III") is not None


def test_stale_dst_artifact_is_not_served(models):
    models(target_season=current_season() - 1, feature_season=current_season() - 2)
    assert project_player("Steelers D/ST") is None


def test_wrong_format_dst_artifact_is_ignored(models):
    models(format="ppr-ppg-projection/1")
    assert project_player("Steelers D/ST") is None


DST_SAMPLE = Projection(
    player="Pittsburgh Steelers D/ST", position="D/ST", team="PIT", season=2026,
    projected_ppg=6.4, last_ppg=6.9, last_games=17, typical_error=1.6,
)


def test_dst_prompt_block_names_dst_scoring():
    block = DST_SAMPLE.prompt_block()
    assert "6.4 standard D/ST fantasy points per game in 2026" in block
    assert "PPR" not in block
    assert "next season's schedule" in block
    assert not anthropic_client._looks_like_prompt_leak(block)


def test_inprocess_client_labels_projection_scoring(models, monkeypatch):
    models()
    monkeypatch.setattr(anthropic_client, "client", object())
    monkeypatch.setattr(anthropic_client, "analyze_player", lambda name, **_: CLEAN)
    client = InProcessAnalysisClient()
    dst = client.analyze(player_name="Steelers D/ST", actor=Actor(user_id="1"))
    assert (dst.projected_ppg, dst.projection_scoring) == (6.5, "D/ST")
    player = client.analyze(player_name="Kenneth Walker III", actor=Actor(user_id="1"))
    assert player.projection_scoring == "PPR"
