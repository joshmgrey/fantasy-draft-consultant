"""Serving-side retrieval (analysis_core/retrieval.py), without Postgres or
the embedding model: a fake store evaluates the same metadata filters
PGVector would, and fake embeddings stand in for fastembed."""

import logging
import sys
import types

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from analysis_core import anthropic_client, retrieval
from analysis_core.retrieval import (
    KIND_PLAYER_SEASON,
    FastEmbedEmbeddings,
    Retriever,
    retrieve_context,
)
from app.analysis.client.base import Actor
from app.analysis.client.inprocess import InProcessAnalysisClient

SEASON = 2026          # current fantasy season in these tests
LATEST = SEASON - 1    # latest completed season


def match(meta, flt):
    """Evaluate a PGVector-style metadata filter ($and, $eq, $ne)."""
    if "$and" in flt:
        return all(match(meta, f) for f in flt["$and"])
    for field, cond in flt.items():
        (op, value), = cond.items()
        if op == "$eq" and meta.get(field) != value:
            return False
        if op == "$ne" and meta.get(field) == value:
            return False
    return True


class FakeStore:
    def __init__(self, docs, vectors):
        self.docs = docs
        self.vectors = vectors       # parallel to docs
        self.calls = []

    def _rank(self, vec, k, flt):
        hits = [(float(np.dot(vec, v)), d) for d, v in zip(self.docs, self.vectors)
                if match(d.metadata, flt)]
        hits.sort(key=lambda t: -t[0])
        return [d for _, d in hits[:k]]

    def similarity_search(self, query, k=4, filter=None):
        self.calls.append(("query", query, k, filter))
        return self._rank(FakeEmbeddings().embed_query(query), k, filter)

    def similarity_search_by_vector(self, embedding, k=4, filter=None):
        self.calls.append(("vector", tuple(embedding), k, filter))
        return self._rank(np.array(embedding), k, filter)


class FakeEmbeddings:
    """2-d vectors: axis 0 = 'injury-ness', axis 1 = target share."""

    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        injury = 1.0 if "injur" in text else 0.1
        share = float(text.split("share=")[1].split()[0]) if "share=" in text else 0.0
        return [injury, share]


def doc(pid, name, season, *, position="WR", injured=False, share=0.2, has_next=True):
    profile = f"{name} {position} {season} share={share}" + (" injury report" if injured else "")
    return Document(
        page_content=profile + (" next season: ..." if has_next else ""),
        metadata={"kind": KIND_PLAYER_SEASON, "player_id": pid, "name": name,
                  "name_key": name.lower(), "position": position, "season": season,
                  "has_next": has_next, "profile": profile},
    )


@pytest.fixture
def store():
    docs = [
        doc("A", "Alpha", LATEST - 2, injured=True, share=0.25),
        doc("A", "Alpha", LATEST - 1, share=0.28),
        doc("A", "Alpha", LATEST, share=0.30, has_next=False),
        doc("A", "Alpha", LATEST - 3, share=0.10),
        doc("B", "Bravo", LATEST - 1, share=0.31),               # closest comparable
        doc("C", "Charlie", LATEST - 2, share=0.05),
        doc("D", "Delta", LATEST - 1, position="RB", share=0.30),  # wrong position
        doc("E", "Echo", LATEST, share=0.30, has_next=False),     # no outcome yet
    ]
    emb = FakeEmbeddings()
    vecs = [np.array(emb._vec(d.metadata["profile"])) for d in docs]
    return FakeStore(docs, vecs), emb


def test_history_ranks_injury_seasons_then_sorts_newest_first(store):
    st, _ = store
    history = Retriever(st).retrieve("alpha")
    seasons = [d.metadata["season"] for d in history]
    assert len(seasons) == retrieval.HISTORY_K
    assert LATEST - 2 in seasons          # the injury season made the cut
    assert seasons == sorted(seasons, reverse=True)
    q = st.calls[0]
    assert q[0] == "query" and q[1] == retrieval.INJURY_QUERY
    assert q[3] == {"$and": [{"kind": {"$eq": KIND_PLAYER_SEASON}},
                             {"name_key": {"$eq": "alpha"}}]}
    assert all(d.metadata["name"] == "Alpha" for d in history)


def test_typed_name_variants_match(store):
    st, _ = store
    assert Retriever(st).retrieve("  ALPHA ")


def test_unknown_player_retrieves_nothing(store):
    st, _ = store
    assert Retriever(st).retrieve("Nobody") == []


def test_k_is_configurable(store):
    st, _ = store
    assert len(Retriever(st, k=1).retrieve("Alpha")) == 1


def test_context_text_lists_seasons():
    text = retrieval.to_context([Document(page_content="H1"), Document(page_content="H2")])
    assert text.splitlines() == [
        "Injury and availability history (NFL weekly injury reports; the 2 most "
        "injury-relevant seasons, newest first):",
        "- H1",
        "- H2",
    ]


# --------------------------------------------------------------------------- #
# retrieve_context: configuration and fallbacks
# --------------------------------------------------------------------------- #
def test_disabled_without_database_url():
    assert retrieval.get_retriever() is None
    assert retrieve_context("Alpha") is None


def test_returns_context_text(store, monkeypatch):
    st, emb = store
    monkeypatch.setattr(retrieval, "_build_retriever", lambda: Retriever(st))
    text = retrieve_context("Alpha")
    assert text.startswith("Injury and availability history")
    assert "Bravo" not in text


def test_empty_result_is_none(store, monkeypatch):
    st, emb = store
    monkeypatch.setattr(retrieval, "_build_retriever", lambda: Retriever(st))
    assert retrieve_context("Nobody") is None


def test_search_errors_are_swallowed(monkeypatch, caplog):
    class Broken:
        def retrieve(self, name):
            raise RuntimeError("db down")

    monkeypatch.setattr(retrieval, "_build_retriever", lambda: Broken())
    with caplog.at_level(logging.ERROR):
        assert retrieve_context("Alpha") is None
    assert "retrieval failed" in caplog.text


def test_retriever_is_built_once(monkeypatch):
    built = []
    monkeypatch.setattr(retrieval, "_build_retriever", lambda: built.append(1) or None)
    retrieval.get_retriever()
    retrieval.get_retriever()
    assert built == [1]


def test_missing_packages_disable_retrieval(monkeypatch, caplog):
    monkeypatch.setenv(retrieval.DATABASE_ENV, "postgresql://x/y")

    def no_pkg(*a, **k):
        raise ImportError("No module named 'langchain_postgres'")

    monkeypatch.setattr(retrieval, "open_store", no_pkg)
    with caplog.at_level(logging.WARNING):
        assert retrieval.get_retriever() is None
    assert "not installed" in caplog.text


def test_store_errors_disable_retrieval(monkeypatch, caplog):
    monkeypatch.setenv(retrieval.DATABASE_ENV, "postgresql://x/y")

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(retrieval, "open_store", boom)
    with caplog.at_level(logging.ERROR):
        assert retrieval.get_retriever() is None
    assert "could not open the retrieval store" in caplog.text


def test_builds_a_real_retriever_from_env(monkeypatch):
    monkeypatch.setenv(retrieval.DATABASE_ENV, "postgresql://x/y")
    monkeypatch.setattr(retrieval, "open_store", lambda url, emb: ("store", url))
    r = retrieval.get_retriever()
    assert isinstance(r, Retriever) and r._store == ("store", "postgresql://x/y")


# --------------------------------------------------------------------------- #
# Store construction and embeddings
# --------------------------------------------------------------------------- #
def test_open_store_configures_pgvector(monkeypatch):
    seen = {}

    class SpyPGVector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_postgres",
                        types.SimpleNamespace(PGVector=SpyPGVector))
    retrieval.open_store("postgresql://u:p@h/db", FakeEmbeddings())
    assert seen["connection"] == "postgresql+psycopg://u:p@h/db"
    assert seen["collection_name"] == retrieval.COLLECTION
    assert seen["embedding_length"] == retrieval.EMBEDDING_DIM
    assert seen["create_extension"] is False and seen["pre_delete_collection"] is False
    assert isinstance(seen["embeddings"], Embeddings)

    retrieval.open_store("postgresql://u:p@h/db", FakeEmbeddings(), create=True, pre_delete=True)
    assert seen["create_extension"] is True and seen["pre_delete_collection"] is True


def test_langchain_adapter_delegates():
    emb = retrieval.as_langchain_embeddings(FakeEmbeddings())
    assert emb.embed_query("injury") == [1.0, 0.0]
    assert emb.embed_documents(["share=0.5"]) == [[0.1, 0.5]]


def test_fastembed_wrapper_loads_model_once_and_uses_query_instruction(monkeypatch, tmp_path):
    loads = []

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            loads.append((model_name, cache_dir))

        def embed(self, texts):
            return (np.array([len(t), 0.0]) for t in texts)

        def query_embed(self, text):
            return iter([np.array([0.0, len(text)])])

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setenv(retrieval.MODEL_CACHE_ENV, str(tmp_path))
    emb = FastEmbedEmbeddings()
    assert emb.embed_documents(["ab", "abc"]) == [[2.0, 0.0], [3.0, 0.0]]
    assert emb.embed_query("abcd") == [0.0, 4.0]
    assert loads == [(retrieval.EMBEDDING_MODEL, str(tmp_path))]


# --------------------------------------------------------------------------- #
# Callers hand the context to the LLM as fenced retrieved content
# --------------------------------------------------------------------------- #
CLEAN = "RISK: 3/10\nVERDICT: Draft\nREASON: Fine value."


def test_inprocess_client_passes_retrieved_context(monkeypatch):
    monkeypatch.setattr(anthropic_client, "client", object())
    monkeypatch.setattr(retrieval, "retrieve_context", lambda name: f"history for {name}")
    seen = {}

    def fake(name, retrieved_context=None, projection=None):
        seen["context"] = retrieved_context
        return CLEAN

    monkeypatch.setattr(anthropic_client, "analyze_player", fake)
    InProcessAnalysisClient().analyze(player_name="Puka Nacua", actor=Actor(user_id="1"))
    assert seen["context"] == "history for Puka Nacua"


def test_retrieved_context_is_fenced_as_untrusted():
    content = anthropic_client._build_user_content("Puka Nacua", "Injury and availability history: ...")
    fence = content.index(anthropic_client.UNTRUSTED_OPEN)
    assert content.index("Injury and availability history") > fence
