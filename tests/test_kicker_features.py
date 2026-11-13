"""Kicker scoring, kicker-season features, and the kicker training pipeline.

Uses a small synthetic league shaped like the nflverse files, so it runs
offline and fast.
"""

import json

import numpy as np
import pandas as pd
import pytest

from analysis_core.projection import KICKER_ARTIFACT_FORMAT, Projector
from ml import kicker_features as K
from ml import train_kicker
from ml import dst_features as D
from tests.helpers import TEAMS, round_robin, schedule_abbr

NUMERIC = [c for c in K._STAT_COLUMNS if c not in (*K._TEXT_COLUMNS, "season")]


def kick_row(pid, season, team, *, games=16, name=None, position="K", **stats):
    row = {c: 0 for c in NUMERIC}
    row["fg_blocked_list"] = ""
    row.update(player_id=pid, player_display_name=name or f"Kicker {pid}",
               position=position, season=season, recent_team=team, games=games, **stats)
    return row


def games_for(seasons, seed=0, indoor=("DEN",)):
    """Round-robin schedule; ``indoor`` teams' home games are in a dome."""
    rng = np.random.default_rng(seed)
    out = []
    for season in seasons:
        for week, pairs in enumerate(round_robin(TEAMS), start=1):
            for home, away in pairs:
                out.append({
                    "season": season, "game_type": "REG", "week": week,
                    "home_team": schedule_abbr(home, season),
                    "away_team": schedule_abbr(away, season),
                    "home_score": int(rng.integers(10, 35)),
                    "away_score": int(rng.integers(10, 35)),
                    "roof": "dome" if home in indoor else "outdoors",
                })
    return pd.DataFrame(out)


def synthetic_kickers(seasons, seed=0):
    rng = np.random.default_rng(seed)
    skill = {t: rng.normal(0, 1) for t in TEAMS}
    rows = []
    for season in seasons:
        for i, team in enumerate(TEAMS):
            q = skill[team] + rng.normal(0, 0.5)
            made = [int(max(0, rng.poisson(lam))) for lam in (1, 8 + q, 8 + q, 6 + q, 3 + q, 0.2)]
            rows.append(kick_row(
                f"K{i}", season, team, games=len(TEAMS) - 1, name=f"Kicker {chr(65 + i)}",
                fg_made_0_19=made[0], fg_made_20_29=made[1], fg_made_30_39=made[2],
                fg_made_40_49=made[3], fg_made_50_59=made[4], fg_made_60_=made[5],
                fg_missed_40_49=int(rng.poisson(1)), fg_missed_50_59=int(rng.poisson(1)),
                pat_made=int(rng.poisson(30)), fg_long=int(rng.integers(45, 62)),
            ))
            r = rows[-1]
            r["fg_made"] = sum(made)
            r["fg_att"] = r["fg_made"] + r["fg_missed_40_49"] + r["fg_missed_50_59"]
            r["pat_att"] = r["pat_made"] + int(rng.poisson(1))
    return pd.DataFrame(rows)


def players_csv(n=len(TEAMS), extra=()):
    """players.csv rows: one kicker per team, plus ``extra`` (name, position,
    last season)."""
    rows = [{"gsis_id": f"K{i}", "display_name": f"Kicker {chr(65 + i)}", "position": "K",
             "birth_date": "1995-09-01", "rookie_season": 2005, "draft_round": None,
             "last_season": 2020, "status": "ACT"}
            for i in range(n)]
    rows += [{"gsis_id": f"X{i}", "display_name": name, "position": pos,
              "birth_date": None, "rookie_season": 2019, "draft_round": 1,
              "last_season": last, "status": status}
             for i, (name, pos, last, status) in enumerate(extra)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_fantasy_points_by_distance_and_misses():
    df = pd.DataFrame([kick_row(
        "A", 2020, "PIT", fg_made_20_29=1, fg_made_40_49=1, fg_made_50_59=1, fg_made_60_=1,
        fg_made=4, fg_att=6, pat_made=3, pat_att=4,
    )])
    # 3 + 4 + 5 + 5 for the makes, -2 for two misses (one could be a block),
    # 3 PATs, -1 for the missed PAT.
    assert K.fantasy_points(df).iloc[0] == 3 + 4 + 5 + 5 - 2 + 3 - 1


# --------------------------------------------------------------------------- #
# Season rows
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def league():
    stats = synthetic_kickers(range(2016, 2022))
    games = games_for(range(2016, 2023))
    return stats, games, K.season_rows(stats, players_csv(), games)


def test_loader_keeps_only_kickers(tmp_path):
    pd.DataFrame([
        kick_row("K1", 2018, "PIT", fg_att=1, fg_made=1),
        kick_row("P1", 2018, "PIT", position="P", fg_att=1, fg_made=1),
    ]).to_csv(tmp_path / "stats_player_reg_2018.csv", index=False)
    assert K.load_kicking(tmp_path, [2018])["player_id"].tolist() == ["K1"]


def test_kicker_who_never_kicked_is_dropped(league):
    stats, games, _ = league
    idle = pd.DataFrame([kick_row("K9", 2018, "PIT", games=2)])
    rows = K.season_rows(pd.concat([stats, idle]), players_csv(), games)
    assert "K9" not in set(rows.player_id)


def test_blocked_kicks_count_at_their_distance():
    b40, b50 = K.blocked_by_distance(pd.Series(["43;51;55", "", None, "38", "49", "NA;52 ; x"]))
    assert b40.tolist() == [1, 0, 0, 0, 1, 0]
    assert b50.tolist() == [2, 0, 0, 0, 0, 1]  # junk tokens are skipped


def test_blocked_long_kicks_are_attempts_but_not_makes(league):
    stats, games, rows = league
    blocked = stats.copy()
    sel = (blocked.player_id == "K0") & (blocked.season == 2018)
    blocked.loc[sel, "fg_blocked_list"] = "52;54"
    blocked.loc[sel, "fg_att"] += 2
    before = rows[(rows.player_id == "K0") & (rows.season == 2018)].iloc[0]
    after = K.season_rows(blocked, players_csv(), games)
    after = after[(after.player_id == "K0") & (after.season == 2018)].iloc[0]
    assert after.fg_50_att_pg == pytest.approx(before.fg_50_att_pg + 2 / before.games)
    assert after.fg_50_pct < before.fg_50_pct
    assert after.ppg == pytest.approx(before.ppg - 2 / before.games)  # two misses


def test_rates_and_percentages(league):
    stats, _, rows = league
    r = rows[(rows.player_id == "K0") & (rows.season == 2018)].iloc[0]
    s = stats[(stats.player_id == "K0") & (stats.season == 2018)].iloc[0]
    assert r.fg_pct == pytest.approx(s.fg_made / s.fg_att)
    assert r.pat_att_pg == pytest.approx(s.pat_att / s.games)
    assert r.ppg == pytest.approx(K.fantasy_points(stats.loc[[s.name]]).iloc[0] / s.games)
    assert r.age == pytest.approx(24.0, abs=0.01)  # at the start of 2019
    assert r.years_exp == 2018 - 2005 + 1


def test_previous_season_only_if_consecutive(league):
    _, _, rows = league
    k = rows[rows.player_id == "K1"].set_index("season")
    assert np.isnan(k.loc[2016, "ppg_prev"])
    assert k.loc[2017, "ppg_prev"] == pytest.approx(k.loc[2016, "ppg"])


def test_team_context_and_next_seasons_schedule(league):
    _, games, rows = league
    den = rows[(rows.team == "DEN") & (rows.season == 2019)].iloc[0]
    # DEN plays 3 or 4 home games (all indoors) out of 7 in 2020.
    tg = D.team_games(games)
    assert den.sched_indoor_share == pytest.approx(
        tg[(tg.team == "DEN") & (tg.season == 2020)]["indoor"].mean())
    assert 0 < den.sched_indoor_share < 1
    assert den.team_points_pg == pytest.approx(
        tg[(tg.team == "DEN") & (tg.season == 2019)]["points_scored"].mean())


def test_relocated_team_gets_its_schedule(league):
    _, _, rows = league
    lv = rows[(rows.team == "LV") & (rows.season == 2019)].iloc[0]
    # 2020 is LV in the schedule file, 2019 is OAK: both fold into LV.
    assert not np.isnan(lv.sched_opp_points_allowed_pg)
    assert not np.isnan(lv.team_points_pg)


def test_no_future_results_leak_into_features():
    stats = synthetic_kickers(range(2016, 2021))
    games = games_for(range(2016, 2022))
    base = K.season_rows(stats, players_csv(), games)

    s2, g2 = stats.copy(), games.copy()
    s2.loc[s2.season == 2020, ["fg_made_50_59", "fg_made", "fg_att"]] += 9
    g2.loc[g2.season == 2020, ["home_score", "away_score"]] = 60
    changed = K.season_rows(s2, players_csv(), g2)

    cols = ["player_id", "season", *K.FEATURES]
    a = base[base.season < 2020][cols].sort_values(["player_id", "season"]).reset_index(drop=True)
    b = changed[changed.season < 2020][cols].sort_values(["player_id", "season"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_training_table_needs_min_games_in_both_seasons(league):
    stats, games, _ = league
    short = stats.copy()
    short.loc[(short.player_id == "K2") & (short.season == 2018), "games"] = K.MIN_GAMES - 1
    table = K.training_table(K.season_rows(short, players_csv(), games))
    k2 = table[table.player_id == "K2"]
    assert 2018 not in set(k2.season)          # as a feature row
    assert 2018 not in set(k2.target_season)   # as a label
    assert (table.target_season == table.season + 1).all()


# --------------------------------------------------------------------------- #
# Training end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("kicker")
    raw = root / "raw"
    raw.mkdir()
    seasons = list(range(2008, 2020))
    stats = synthetic_kickers(seasons)
    for season, frame in stats.groupby("season"):
        frame.to_csv(raw / f"stats_player_reg_{season}.csv", index=False)
    games = games_for(seasons + [2020])
    games.loc[games.season == 2020, ["home_score", "away_score"]] = np.nan
    games.to_csv(raw / "games.csv", index=False)
    # An incoming WR who shares kicker B's name: kicker B must not be served.
    # An active TE who missed 2019 hurt shares kicker E's name: not served either.
    # A safety named like kicker C retired long ago: kicker C still is.
    players_csv(extra=[("Kicker B", "WR", 2020, "ACT"), ("Kicker E", "TE", 2018, "ACT"),
                       ("Kicker C", "SAF", 1994, "CUT")]).to_csv(raw / "players.csv", index=False)
    out = root / "kicker.joblib"
    report = train_kicker.run(raw, 2008, 2019, out, root / "reports")
    return {"root": root, "out": out, "report": report}


def test_report_scores_model_and_both_baselines(trained):
    r = trained["report"]
    assert r["split"] == {"validation_target_seasons": [2015, 2018], "test_target_season": 2019}
    assert set(r["test"]) == {"model", "baseline_last_season", "baseline_league_average"}
    assert r["scoring"] == K.SCORING
    md = (trained["root"] / "reports" / "KICKER_REPORT.md").read_text(encoding="utf-8")
    assert "Backtest" in md and "None" not in md
    data = json.loads((trained["root"] / "reports" / "kicker_metrics.json").read_text("utf-8"))
    assert data["chosen_model"] == r["chosen_model"]


def test_artifact_serves_kickers_by_name(trained):
    projector = Projector.load(trained["out"], KICKER_ARTIFACT_FORMAT)
    assert projector.season == 2020
    p = projector.project("kicker d")
    assert (p.player, p.position, p.team, p.scoring) == ("Kicker D", "K", "CIN", "K")
    assert projector.project("kicker b") is None  # also a current WR's name
    assert projector.project("kicker e") is None  # an active TE who sat out 2019
    assert projector.project("kicker c") is not None  # namesake retired in 1994
    assert len(projector._players) == len(TEAMS) - 2
    assert Projector.load(trained["out"]) is None  # not a player-model artifact


def test_too_few_seasons_is_a_clear_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    stats = synthetic_kickers(range(2015, 2020))
    for season, frame in stats.groupby("season"):
        frame.to_csv(raw / f"stats_player_reg_{season}.csv", index=False)
    games_for(range(2015, 2021)).to_csv(raw / "games.csv", index=False)
    players_csv().to_csv(raw / "players.csv", index=False)
    with pytest.raises(ValueError, match="too few"):
        train_kicker.run(raw, 2015, 2019, tmp_path / "k.joblib", tmp_path / "reports")


def test_missing_raw_file_says_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="later --first"):
        K.load_kicking(tmp_path, [2002])
    with pytest.raises(FileNotFoundError) as err:
        D.load_games(tmp_path)
    assert "without --rag-only" in str(err.value) and "--first" not in str(err.value)


def test_unknown_roof_takes_the_stadium_roof_or_stays_unknown():
    games = pd.DataFrame([
        {"season": 2020, "game_type": "REG", "week": 1, "home_team": "ATL", "away_team": "NO",
         "home_score": 20, "away_score": 17, "roof": "closed", "stadium_id": "ATL00"},
        {"season": 2021, "game_type": "REG", "week": 1, "home_team": "ATL", "away_team": "TB",
         "home_score": None, "away_score": None, "roof": None, "stadium_id": "ATL00"},
        {"season": 2021, "game_type": "REG", "week": 2, "home_team": "NE", "away_team": "ATL",
         "home_score": None, "away_score": None, "roof": None, "stadium_id": "NEW00"},
    ])
    tg = D.team_games(games)
    atl = tg[tg.team == "ATL"].set_index(["season", "week"])["indoor"]
    assert atl.loc[(2021, 1)] == 1.0          # filled from the stadium's other games
    assert np.isnan(atl.loc[(2021, 2)])       # no record anywhere: unknown, not outdoors


def test_games_csv_missing_a_required_column_is_an_error(tmp_path):
    pd.DataFrame([{"season": 2020, "game_type": "REG", "week": 1, "home_team": "PIT",
                   "away_team": "LV", "home_points": 3, "away_score": 0}]).to_csv(
        tmp_path / "games.csv", index=False)
    with pytest.raises(ValueError, match="home_score"):
        D.load_games(tmp_path)
