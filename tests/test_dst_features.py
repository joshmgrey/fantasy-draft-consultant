"""D/ST scoring, team-season features, and the D/ST training pipeline.

Uses a small synthetic league shaped like the nflverse files, so it runs
offline and fast.
"""

import json

import numpy as np
import pandas as pd
import pytest

from analysis_core.projection import DST_ARTIFACT_FORMAT, Projector
from ml import dst_features as D
from ml import train_dst
from tests.helpers import TEAMS, round_robin, schedule_abbr

STAT_COLS = [c for c in D._WEEK_COLUMNS
             if c not in ("season", "week", "team", "season_type", "opponent_team")]


def synthetic_league(seasons, seed=0):
    rng = np.random.default_rng(seed)
    week_rows, game_rows = [], []
    # Each team has a persistent "quality" so seasons are correlated.
    quality = {t: rng.normal(0, 1) for t in TEAMS}
    for season in seasons:
        for week, pairs in enumerate(round_robin(TEAMS), start=1):
            for home, away in pairs:
                scores = {}
                for team, opp in ((home, away), (away, home)):
                    q = quality[team] + rng.normal(0, 0.5)
                    row = {c: 0.0 for c in STAT_COLS}
                    row.update(
                        season=season, week=week, team=team, season_type="REG",
                        opponent_team=opp,
                        def_sacks=max(0, round(2.5 + q + rng.normal())),
                        def_qb_hits=max(0, round(5 + q + rng.normal())),
                        def_interceptions=rng.poisson(0.8),
                        def_pass_defended=rng.poisson(5),
                        fumble_recovery_opp=rng.poisson(0.5),
                        passing_yards=rng.integers(150, 320),
                        rushing_yards=rng.integers(60, 160),
                        passing_interceptions=rng.poisson(0.8),
                        fumbles_lost_total=rng.poisson(0.5),
                        sacks_suffered=rng.poisson(2.5),
                    )
                    week_rows.append(row)
                    scores[opp] = int(max(0, round(21 - 3 * q + rng.normal(0, 7))))
                game_rows.append({
                    "season": season, "game_type": "REG", "week": week,
                    "home_team": schedule_abbr(home, season),
                    "away_team": schedule_abbr(away, season),
                    "home_score": scores[home], "away_score": scores[away],
                })
    return pd.DataFrame(week_rows), pd.DataFrame(game_rows)


def write_raw(raw, seasons, *, schedule_next=True, seed=0):
    weeks, games = synthetic_league(seasons, seed)
    if schedule_next:
        # The following season's schedule is published before it is played.
        _, nxt_games = synthetic_league([max(seasons) + 1], seed + 1)
        nxt_games[["home_score", "away_score"]] = np.nan
        games = pd.concat([games, nxt_games], ignore_index=True)
    for season, frame in weeks.groupby("season"):
        frame.to_csv(raw / f"stats_team_week_{season}.csv", index=False)
    games.to_csv(raw / "games.csv", index=False)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_points_allowed_tiers():
    pa = [0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 52]
    assert D.points_allowed_points(pa).tolist() == [10, 7, 7, 4, 4, 1, 1, 0, 0, -1, -1, -4, -4]


def one_game(**home_stats):
    blank = {c: 0 for c in STAT_COLS}
    home = {**blank, "season": 2020, "week": 1, "team": "PIT", "season_type": "REG",
            "opponent_team": "LV", **home_stats}
    away = {**blank, "season": 2020, "week": 1, "team": "LV", "season_type": "REG",
            "opponent_team": "PIT", "passing_yards": 200, "rushing_yards": 90}
    games = pd.DataFrame([{"season": 2020, "game_type": "REG", "week": 1, "home_team": "PIT",
                           "away_team": "LV", "home_score": 24, "away_score": 10}])
    return D.weekly_points(pd.DataFrame([home, away]), games).set_index("team")


def test_weekly_points_add_up():
    w = one_game(def_sacks=2, def_interceptions=1, fumble_recovery_opp=1, def_tds=1,
                 def_safeties=1, def_punt_blocks=1)
    # 2 sacks + INT 2 + recovery 2 + TD 6 + safety 2 + block 2 + 10 allowed (+4).
    assert w.loc["PIT", "dst_points"] == 2 + 2 + 2 + 6 + 2 + 2 + 4
    assert w.loc["PIT", "yards_allowed"] == 290
    assert w.loc["PIT", "points_allowed"] == 10


def test_return_tds_count_once_each():
    w = one_game(def_tds=1, special_teams_tds=1)
    assert w.loc["PIT", "touchdowns"] == 2


def test_postseason_is_ignored():
    w = one_game(def_sacks=1)
    assert len(w) == 2  # one regular-season row per team
    blank = {c: 0 for c in STAT_COLS}
    post = pd.DataFrame([{**blank, "season": 2020, "week": 19, "team": "PIT",
                          "season_type": "POST", "opponent_team": "LV", "def_sacks": 9}])
    games = pd.DataFrame([{"season": 2020, "game_type": "POST", "week": 19, "home_team": "PIT",
                           "away_team": "LV", "home_score": 0, "away_score": 0}])
    assert D.weekly_points(post, games).empty


# --------------------------------------------------------------------------- #
# Season rows
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def league():
    weeks, games = synthetic_league(range(2016, 2022))
    return weeks, games, D.season_rows(weeks, games)


def test_one_row_per_team_season(league):
    _, _, rows = league
    assert len(rows) == len(TEAMS) * 6
    assert set(rows["team"]) == set(TEAMS)  # OAK/SD folded into LV/LAC
    assert (rows["games"] == len(TEAMS) - 1).all()


def test_relocated_franchise_keeps_its_history(league):
    _, _, rows = league
    lv = rows[rows["team"] == "LV"].set_index("season")
    # 2020 is the first LV season in the schedule file; 2019's row is still "previous".
    assert lv.loc[2020, "ppg_prev"] == pytest.approx(lv.loc[2019, "ppg"])


def test_season_ppg_is_mean_of_weekly_points(league):
    weeks, games, rows = league
    w = D.weekly_points(weeks, games)
    expected = w[(w.team == "PIT") & (w.season == 2018)]["dst_points"].mean()
    got = rows[(rows.team == "PIT") & (rows.season == 2018)]["ppg"].iloc[0]
    assert got == pytest.approx(expected)


def test_schedule_strength_uses_next_seasons_opponents_rated_by_this_season(league):
    weeks, games, rows = league
    pit = rows[(rows.team == "PIT") & (rows.season == 2019)].iloc[0]
    sched = D.team_games(games)
    opps = sched[(sched.team == "PIT") & (sched.season == 2020)]["opponent"]
    off = rows[rows.season == 2019].set_index("team")["points_scored_pg"]
    assert pit["sched_opp_points_pg"] == pytest.approx(off.loc[opps].mean())


def test_no_future_results_leak_into_features():
    weeks, games = synthetic_league(range(2016, 2021))
    base = D.season_rows(weeks, games)

    # Rewrite every 2020 result (box scores and scores); the 2020 schedule stays.
    w2, g2 = weeks.copy(), games.copy()
    later = w2["season"] == 2020
    w2.loc[later, "def_sacks"] += 7
    w2.loc[later, "passing_interceptions"] += 3
    g2.loc[g2["season"] == 2020, ["home_score", "away_score"]] = 50
    changed = D.season_rows(w2, g2)

    cols = ["team", "season", *D.FEATURES]
    a = base[base.season < 2020][cols].sort_values(["team", "season"]).reset_index(drop=True)
    b = changed[changed.season < 2020][cols].sort_values(["team", "season"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_training_table_labels_with_next_season(league):
    _, _, rows = league
    table = D.training_table(rows)
    assert (table["target_season"] == table["season"] + 1).all()
    r = table[(table.team == "KC") & (table.season == 2017)].iloc[0]
    nxt = rows[(rows.team == "KC") & (rows.season == 2018)]["ppg"].iloc[0]
    assert r["target_ppg"] == pytest.approx(nxt)


# --------------------------------------------------------------------------- #
# Training end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("dst")
    raw = root / "raw"
    raw.mkdir()
    write_raw(raw, list(range(2008, 2020)))
    out = root / "dst.joblib"
    report = train_dst.run(raw, 2008, 2019, out, root / "reports")
    return {"root": root, "out": out, "report": report}


def test_report_scores_model_and_both_baselines(trained):
    r = trained["report"]
    assert r["split"] == {"validation_target_seasons": [2015, 2018], "test_target_season": 2019}
    assert set(r["test"]) == {"model", "baseline_last_season", "baseline_league_average"}
    assert r["test"]["baseline_league_average"]["spearman"] is None
    assert r["backtest"]["target_seasons"] == [2016, 2019]
    names = [m["model"] for m in r["validation_leaderboard"]]
    assert "baseline: league_average" in names and r["chosen_model"] in names


def test_report_files_are_valid_json_and_markdown(trained):
    reports = trained["root"] / "reports"
    data = json.loads((reports / "dst_metrics.json").read_text(encoding="utf-8"))
    assert data["chosen_model"] == trained["report"]["chosen_model"]
    md = (reports / "DST_REPORT.md").read_text(encoding="utf-8")
    assert "Backtest" in md and "None" not in md


def test_artifact_serves_every_team_for_next_season(trained):
    projector = Projector.load(trained["out"], DST_ARTIFACT_FORMAT)
    assert projector is not None
    assert projector.season == 2020
    p = projector.project_key("LV")
    assert p.player == "Las Vegas Raiders D/ST"
    assert p.position == "D/ST" and p.scoring == "D/ST"
    assert p.last_games == len(TEAMS) - 1
    assert all(projector.project_key(t) is not None for t in TEAMS)


def test_player_loader_rejects_the_dst_artifact(trained):
    assert Projector.load(trained["out"]) is None
