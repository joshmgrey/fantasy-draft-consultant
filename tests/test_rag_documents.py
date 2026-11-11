"""Building, indexing, and evaluating the retrieval corpus (ml/rag), on
small synthetic nflverse-shaped files."""

import json

import numpy as np
import pandas as pd
import pytest

from analysis_core.retrieval import KIND_PLAYER_SEASON
from ml import features as F
from ml.rag import documents as D
from ml.rag import evaluate, index
from tests.test_ml_features import players_table, stat, write_synthetic_raw

INJ_COLS = ["season", "game_type", "team", "week", "gsis_id", "position", "full_name",
            "report_primary_injury", "report_status", "practice_primary_injury", "practice_status"]


def injury(pid, season, week, status, injury_type, practice="Limited Participation in Practice",
           game_type="REG"):
    return [season, game_type, "AAA", week, pid, "WR", pid, injury_type, status, injury_type, practice]


@pytest.fixture
def raw(tmp_path):
    stats = pd.DataFrame([
        stat("A", 2020, name="Alpha One", games=10, ppr=150.0, targets=90, receptions=60,
             receiving_yards=800, target_share=0.24),
        stat("A", 2021, name="Alpha One", games=17, ppr=255.0, targets=150, receptions=100,
             receiving_yards=1300, target_share=0.28),
        stat("B", 2021, name="Bravo Two", position="RB", games=16, ppr=192.0, carries=250,
             rushing_yards=1100, targets=40, receptions=30),
    ])
    for season, g in stats.groupby("season"):
        g.reindex(columns=F._STAT_COLUMNS).to_csv(tmp_path / f"stats_player_reg_{season}.csv", index=False)
    bio = pd.DataFrame({
        "gsis_id": ["A", "B"], "birth_date": ["1998-09-01", "1997-01-01"],
        "rookie_season": [2019, 2018], "draft_round": [2, None], "pfr_id": ["AlphOn00", "BravTw00"],
    })
    bio.to_csv(tmp_path / "players.csv", index=False)
    pd.DataFrame([
        injury("A", 2020, 3, "Questionable", "Hamstring"),
        injury("A", 2020, 4, "Out", "Hamstring", practice="Did Not Participate In Practice"),
        injury("A", 2020, 5, "Out", "Ankle"),
        injury("A", 2020, 6, None, None, practice="Did Not Participate In Practice"),
        injury("A", 2020, 7, "Out", "Not injury related - personal matter"),
        injury("A", 2020, 19, "Out", "Knee", game_type="WC"),     # playoffs: ignored
    ], columns=INJ_COLS).to_csv(tmp_path / "injuries_2020.csv", index=False)
    pd.DataFrame([
        {"season": 2020, "game_type": "REG", "pfr_player_id": "AlphOn00", "offense_pct": 0.8},
        {"season": 2020, "game_type": "REG", "pfr_player_id": "AlphOn00", "offense_pct": 0.6},
        {"season": 2020, "game_type": "DIV", "pfr_player_id": "AlphOn00", "offense_pct": 0.1},
    ]).to_csv(tmp_path / "snap_counts_2020.csv", index=False)
    return tmp_path


def by(docs, pid, season):
    return next(d for d in docs if d.metadata["player_id"] == pid and d.metadata["season"] == season)


def test_document_text_and_metadata(raw):
    docs = D.build_documents(raw, 2020, 2021)
    a20 = by(docs, "A", 2020)
    text = a20.page_content
    assert text.startswith("Alpha One (WR, AAA, age 22) 2020 season: played 10 of 16 games, 15.0 PPR")
    assert "9.0 targets, 6.0 receptions and 80 receiving yards per game, 24% target share" in text
    assert "Played 70% of offensive snaps." in text                  # REG only: (0.8 + 0.6) / 2
    assert "listed for 3 games (2 out, 0 doubtful, 1 questionable)" in text
    assert "hamstring (2 weeks), ankle (1 week)" in text
    assert "personal" not in text and "knee" not in text            # non-injury and playoffs dropped
    assert "did not practice in 2 reports" in text
    assert text.endswith("Next season: 15.0 PPR points per game in 17 games (+0.0).")

    m = a20.metadata
    assert m["kind"] == KIND_PLAYER_SEASON and m["name_key"] == "alpha one"
    assert (m["has_next"], m["next_ppg"], m["next_games"], m["weeks_listed"]) == (True, 15.0, 17, 3)
    assert "Next season" not in m["profile"]                       # the vector never sees the outcome


def test_latest_season_has_no_outcome(raw):
    docs = D.build_documents(raw, 2020, 2021)
    a21 = by(docs, "A", 2021)
    assert a21.metadata["has_next"] is False and a21.metadata["next_ppg"] is None
    assert "Next season" not in a21.page_content
    assert "played 17 of 17 games" in a21.page_content            # 17-game seasons from 2021
    assert "Not listed on the injury report." in a21.page_content


def test_role_text_by_position(raw):
    rb = by(D.build_documents(raw, 2020, 2021), "B", 2021).page_content
    assert "15.6 carries, 69 rushing yards, 2.5 targets and 1.9 receptions per game" in rb


def test_missing_injury_and_snap_files_still_build(tmp_path):
    write_synthetic_raw(tmp_path, first=2018, last=2019, n_players=5)
    p = pd.read_csv(tmp_path / "players.csv")
    p["pfr_id"] = None
    p.to_csv(tmp_path / "players.csv", index=False)
    docs = D.build_documents(tmp_path, 2018, 2019)
    assert len(docs) == 10
    assert all("Not listed on the injury report." in d.page_content for d in docs)
    assert all("offensive snaps" not in d.page_content for d in docs)


def test_plural_and_team_games():
    assert D.plural(1, "game") == "1 game" and D.plural(3, "game") == "3 games"
    assert D.team_games(2020) == 16 and D.team_games(2021) == 17


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
class RecordingEmbeddings:
    model_name = "fake"

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [1.0, float(len(text))]


def test_index_embeds_profiles_and_writes_in_batches(raw, monkeypatch):
    added, opened = [], {}

    class FakeStore:
        def add_embeddings(self, texts, embeddings, metadatas, ids):
            added.append((texts, embeddings, metadatas, ids))

    def fake_open(url, emb, *, create=False, pre_delete=False):
        opened.update(url=url, create=create, pre_delete=pre_delete)
        return FakeStore()

    monkeypatch.setattr(index, "open_store", fake_open)
    emb = RecordingEmbeddings()
    n = index.index("postgresql://x/y", raw, 2020, 2021, embeddings=emb, batch=2)
    assert n == 3
    assert opened == {"url": "postgresql://x/y", "create": True, "pre_delete": True}
    assert [len(b[0]) for b in added] == [2, 1]
    texts, vecs, metas, ids = added[0]
    assert emb.calls[0] == [m["profile"] for m in metas]           # vectors embed profiles
    assert any("Next season" in t for t in texts)                   # stored text keeps outcomes
    assert all(i == f"{m['player_id']}-{m['season']}" for i, m in zip(ids, metas))


def test_index_cli_requires_a_database(monkeypatch, capsys):
    monkeypatch.delenv("RAG_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit):
        index.main([])
    assert "RAG_DATABASE_URL" in capsys.readouterr().err


def test_index_cli(raw, monkeypatch, capsys):
    monkeypatch.setattr(index, "index", lambda url, raw_dir, first, last: 42)
    index.main(["--database-url", "postgresql://x/y", "--raw", str(raw)])
    assert "indexed 42" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class ShareEmbeddings:
    """Embeds the target-share / carries numbers so similar roles are close,
    and injury listings so injury seasons match the injury query."""

    model_name = "share-fake"

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 1.0]

    @staticmethod
    def _vec(t):
        ppg = float(t.split(" PPR points per game")[0].rsplit(", ", 1)[1])
        listed = "listed for" in t
        return [ppg / 30, 1.0, 3.0 if listed else 0.01]


def test_evaluate_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_synthetic_raw(raw, first=2014, last=2019, n_players=60)
    p = pd.read_csv(raw / "players.csv")
    p["pfr_id"] = None
    p.to_csv(raw / "players.csv", index=False)
    rows = []
    for i in range(0, 60, 3):           # one clearly worst injury season per player
        for w in range(4):
            rows.append(injury(f"P{i}", 2018, w + 1, "Out", "Knee"))
    pd.DataFrame(rows, columns=INJ_COLS).to_csv(raw / "injuries_2018.csv", index=False)

    report = evaluate.run(raw, 2014, 2019, tmp_path / "reports", embeddings=ShareEmbeddings())
    comps = report["comparables"]["mae"]
    assert comps["overall"]["n"] > 0
    assert set(comps["overall"]) == {"n", "no_change", "random_comparables",
                                     "position_average_change", "similar_comparables"}
    inj = report["injury_ranking"]
    assert inj["n"] == 20
    assert inj["hit_at_1/injury_query"] == 1.0
    saved = json.loads((tmp_path / "reports" / "rag_metrics.json").read_text(encoding="utf-8"))
    assert saved["embedding_model"] == "share-fake"
    assert "Hit@1" in (tmp_path / "reports" / "RAG_REPORT.md").read_text(encoding="utf-8")


def test_evaluate_cli(monkeypatch, capsys, tmp_path):
    fake = {"comparables": {"mae": {"overall": {"n": 1}}}, "injury_ranking": {"n": 2}}
    monkeypatch.setattr(evaluate, "run", lambda raw, first, last, reports: fake)
    evaluate.main(["--reports", str(tmp_path)])
    out = capsys.readouterr().out
    assert '"n": 1' in out and '"n": 2' in out


def test_mae_and_normalize():
    assert evaluate.mae([1, 2], [2, 4]) == 1.5
    np.testing.assert_allclose(np.linalg.norm(evaluate.normalize(np.array([[3.0, 4.0]])), axis=1), [1.0])
