"""Builders shared by several test modules: a small synthetic league (teams,
round-robin schedule, relocated franchises) and a tiny player-model artifact.

Kept here rather than imported from another test module, so changing one
test file's internals can't silently change another's fixtures.
"""

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

from analysis_core.projection import ARTIFACT_FORMAT
from analysis_core.season import current_season

TEAMS = ["PIT", "BAL", "CLE", "CIN", "LV", "KC", "DEN", "LAC"]


def schedule_abbr(team, season):
    # The schedule file uses a franchise's old abbreviation before a move;
    # team stats always use the current one.
    if team == "LV" and season < 2020:
        return "OAK"
    if team == "LAC" and season < 2017:
        return "SD"
    return team


def round_robin(teams):
    """Circle method: every team plays every other team once."""
    teams = list(teams)
    weeks = []
    for _ in range(len(teams) - 1):
        weeks.append([(teams[i], teams[-1 - i]) for i in range(len(teams) // 2)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return weeks



def player_artifact(players=None, **overrides):
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
