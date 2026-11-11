"""Index and retrieve against a real Postgres with pgvector.

Skipped unless TEST_PGVECTOR_URL points at a database where the test may
create the ``vector`` extension, e.g.::

    docker run -d --name pgv -e POSTGRES_PASSWORD=pw -p 5433:5432 pgvector/pgvector:pg16
    TEST_PGVECTOR_URL=postgresql://postgres:pw@localhost:5433/postgres pytest tests/test_rag_pgvector.py

Uses small fake embeddings, so it exercises the store, metadata filters, and
cosine ranking, not the embedding model.
"""

import os

import pytest

from analysis_core import retrieval
from analysis_core.retrieval import Retriever, open_store
from ml.rag import index
from tests.test_rag_documents import RecordingEmbeddings, raw  # noqa: F401  (fixture)

URL = os.environ.get("TEST_PGVECTOR_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_PGVECTOR_URL not set")


class TinyEmbeddings(RecordingEmbeddings):
    """Unit vectors: role (WR vs RB) and whether the season had injuries."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return [0.0, 1.0] + [0.0] * (retrieval.EMBEDDING_DIM - 2)

    @staticmethod
    def _vec(t):
        injured = "listed for" in t
        v = [1.0, 1.0 if injured else 0.0] + [0.0] * (retrieval.EMBEDDING_DIM - 2)
        return v


def test_index_then_retrieve(raw, monkeypatch):  # noqa: F811
    emb = TinyEmbeddings()
    n = index.index(URL, raw, 2020, 2021, embeddings=emb)
    assert n == 3

    store = open_store(URL, emb)
    history = Retriever(store).retrieve("Alpha One")
    # Both of Alpha's seasons, newest first; never another player's.
    assert [d.metadata["season"] for d in history] == [2021, 2020]
    assert "Bravo" not in " ".join(d.page_content for d in history)
    # The injury season ranks first for the injury query.
    assert Retriever(store, k=1).retrieve("Alpha One")[0].metadata["season"] == 2020

    # Reindexing replaces the collection rather than duplicating it.
    index.index(URL, raw, 2020, 2021, embeddings=emb)
    again = Retriever(open_store(URL, emb)).retrieve("Alpha One")
    assert [d.metadata["season"] for d in again] == [2021, 2020]

    # The service path end to end, through the env var.
    monkeypatch.setenv(retrieval.DATABASE_ENV, URL)
    monkeypatch.setattr(retrieval, "FastEmbedEmbeddings", lambda: emb)
    retrieval.reset_retriever()
    text = retrieval.retrieve_context("alpha one")
    assert text.startswith("Injury and availability history")
    assert "Alpha One (WR, AAA, age 22) 2020 season" in text
