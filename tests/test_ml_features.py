"""Feature engineering and the training pipeline (ml/).

Uses small synthetic nflverse-shaped tables, so it runs offline and fast.
"""

import json

import numpy as np
import pandas as pd
import pytest

from analysis_core.projection import Projector
from ml import features as F
from ml import train

STAT_DEFAULTS = {c: 0.0 for c in F._STAT_COLUMNS}


def stat(player_id, season, *, name=None, position="WR", games=16, ppr=160.0, **extra):
    row = dict(STAT_DEFAULTS)
    row.update(
        player_id=player_id, player_display_name=name or f"Player {player_id}",
        position=position, season=season, recent_team="AAA", games=games,
        fantasy_points_ppr=ppr, **extra,
    )
    return row


def players_table(rows):
    return pd.DataFrame(rows, columns=["gsis_id", "birth_date", "rookie_season", "draft_round"])


# --------------------------------------------------------------------------- #
# season_rows
# --------------------------------------------------------------------------- #
@pytest.fixture
def rows():
    stats = pd.DataFrame([
        stat("A", 2020, games=10, ppr=100.0, targets=50, target_share=None),
        stat("A", 2021, games=16, ppr=240.0, targets=128, target_share=0.25),
        stat("A", 2023, games=8, ppr=80.0),               # skipped 2022
        stat("K", 2021, position="K", ppr=150.0),         # kicker: dropped
        stat("Z", 2021, games=0, ppr=0.0),                # no games: dropped
    ])
    bio = players_table([
        ("A", "1999-09-01", 2020, None),                   # undrafted
    ])
    return F.season_rows(stats, bio)


def test_per_game_rates_and_ppg(rows):
    a21 = rows[(rows.player_id == "A") & (rows.season == 2021)].iloc[0]
    assert a21.ppg == 15.0
    assert a21.targets_pg == 8.0
    assert a21.target_share == 0.25


def test_missing_shares_become_zero(rows):
    a20 = rows[(rows.player_id == "A") & (rows.season == 2020)].iloc[0]
    assert a20.target_share == 0.0


def test_previous_season_only_counts_if_consecutive(rows):
    a = rows[rows.player_id == "A"].set_index("season")
    assert np.isnan(a.loc[2020, "ppg_prev"])
    assert a.loc[2021, "ppg_prev"] == 10.0
    assert a.loc[2021, "ppg_change"] == 5.0
    assert np.isnan(a.loc[2023, "ppg_prev"])  # 2022 missing, not 2021's value


def test_bio_features(rows):
    a21 = rows[(rows.player_id == "A") & (rows.season == 2021)].iloc[0]
    # Age at the start of the 2022 season (the one being predicted).
    assert a21.age == pytest.approx(23.0, abs=0.01)
    assert a21.years_exp == 2
    assert a21.draft_round == F.UNDRAFTED_ROUND
    assert (a21.pos_WR, a21.pos_RB) == (1.0, 0.0)


def test_only_skill_positions_with_games(rows):
    assert set(rows.player_id) == {"A"}


def test_features_never_use_the_label_season():
    """Changing season t+1 must not change any season-t feature."""
    base = [stat("A", 2020, ppr=100.0), stat("A", 2021, ppr=200.0)]
    changed = [stat("A", 2020, ppr=100.0), stat("A", 2021, ppr=999.0, games=4)]
    bio = players_table([("A", "1998-01-01", 2019, 2)])
    r1 = F.season_rows(pd.DataFrame(base), bio)
    r2 = F.season_rows(pd.DataFrame(changed), bio)
    f1 = r1[r1.season == 2020][F.FEATURES].to_numpy()
    f2 = r2[r2.season == 2020][F.FEATURES].to_numpy()
    np.testing.assert_array_equal(np.nan_to_num(f1, nan=-1), np.nan_to_num(f2, nan=-1))


# --------------------------------------------------------------------------- #
# training_table
# --------------------------------------------------------------------------- #
def test_training_table_pairs_next_season_and_filters_short_seasons():
    stats = pd.DataFrame([
        stat("A", 2020, games=16, ppr=160.0),
        stat("A", 2021, games=10, ppr=200.0),
        stat("B", 2020, games=3, ppr=60.0),    # too few games as a feature row
        stat("B", 2021, games=16, ppr=160.0),
        stat("C", 2020, games=16, ppr=160.0),
        stat("C", 2021, games=2, ppr=40.0),    # too few games as a label
    ])
    table = F.training_table(F.season_rows(stats, players_table([])))
    assert list(table.player_id) == ["A"]
    assert table.iloc[0].target_ppg == 20.0
    assert table.iloc[0].target_season == 2021


# --------------------------------------------------------------------------- #
# End to end: download format -> train -> artifact the service can load
# --------------------------------------------------------------------------- #
def write_synthetic_raw(raw_dir, first=2012, last=2019, n_players=120, seed=0):
    """True process: next PPG regresses toward 10, so last season's PPG (the
    baseline) is beatable."""
    rng = np.random.default_rng(seed)
    talent = rng.uniform(4, 22, n_players)
    ppg = talent.copy()
    for season in range(first, last + 1):
        rows = []
        for i in range(n_players):
            games = int(rng.integers(6, 18))
            rows.append(stat(
                f"P{i}", season, name=f"Player {i}", games=games,
                ppr=round(ppg[i] * games, 1), targets=ppg[i] * games / 2,
            ))
        pd.DataFrame(rows, columns=F._STAT_COLUMNS).to_csv(
            raw_dir / f"stats_player_reg_{season}.csv", index=False
        )
        ppg = 0.5 * ppg + 0.5 * 10 + rng.normal(0, 0.5, n_players)
    bio = [(f"P{i}", "1997-06-01", 2011, 1 + i % 7) for i in range(n_players)]
    players_table(bio).to_csv(raw_dir / "players.csv", index=False)


def test_train_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_synthetic_raw(raw)
    out = tmp_path / "model.joblib"
    report = train.run(raw, 2012, 2019, out, tmp_path / "reports")

    assert report["split"] == {
        "train_target_seasons": [2013, 2017],
        "validation_target_season": 2018,
        "test_target_season": 2019,
    }
    model_mae = report["test"]["model"]["overall"]["mae"]
    base_mae = report["test"]["baseline"]["overall"]["mae"]
    assert model_mae < base_mae
    assert report["test_mae_improvement_vs_baseline_pct"] > 0

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["chosen_model"] == report["chosen_model"]
    assert "Limitations" in (tmp_path / "reports" / "REPORT.md").read_text(encoding="utf-8")

    projector = Projector.load(out)
    assert projector.season == 2020
    p = projector.project("player 3")
    assert p is not None and p.last_games >= F.MIN_GAMES
    assert p.typical_error == model_mae


def test_split_is_by_season():
    table = pd.DataFrame({"target_season": [2013, 2016, 2017, 2018, 2019]})
    tr, val, te = train.split(table, 2019)
    assert list(tr.target_season) == [2013, 2016, 2017]
    assert list(val.target_season) == [2018]
    assert list(te.target_season) == [2019]


def test_serving_rows_keep_the_player_with_more_games_on_name_clash():
    stats = pd.DataFrame([
        stat("A", 2021, name="Mike Williams", games=6, ppr=60.0),
        stat("B", 2021, name="Mike Williams", games=15, ppr=180.0),
        stat("C", 2021, name="Short Season", games=2, ppr=30.0),
    ])
    players = train.serving_rows(F.season_rows(stats, players_table([])), 2021)
    assert players["mike williams"]["games"] == 15
    assert "short season" not in players
    assert len(players["mike williams"]["features"]) == len(F.FEATURES)


# --------------------------------------------------------------------------- #
# Download script (network stubbed)
# --------------------------------------------------------------------------- #
def test_download_fetches_each_season_and_players(tmp_path, monkeypatch):
    from ml import download

    fetched = []
    monkeypatch.setattr(download, "fetch", lambda url, dest: fetched.append((url, dest.name)))
    download.main(["--first", "2020", "--last", "2021", "--out", str(tmp_path)])
    assert fetched == [
        (download.stats_url(2020), "stats_player_reg_2020.csv"),
        (download.stats_url(2021), "stats_player_reg_2021.csv"),
        (download.PLAYERS_URL, "players.csv"),
    ]


def test_fetch_writes_the_response_body(tmp_path, monkeypatch):
    import io

    from ml import download

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(req, timeout):
        seen["ua"] = req.get_header("User-agent")
        return FakeResponse(b"a,b\n1,2\n")

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "sub" / "x.csv"
    download.fetch("https://example.invalid/x.csv", dest)
    assert dest.read_bytes() == b"a,b\n1,2\n"
    assert seen["ua"] == "fantasy-draft-consultant-ml"


def test_train_cli(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_synthetic_raw(raw, n_players=60)
    out = tmp_path / "model.joblib"
    train.main([
        "--raw", str(raw), "--first", "2012", "--last", "2019",
        "--out", str(out), "--reports", str(tmp_path / "reports"),
    ])
    assert out.exists()
    assert "chosen_model" in capsys.readouterr().out
