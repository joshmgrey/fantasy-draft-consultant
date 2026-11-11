"""Retrieval-augmented generation: player history from a pgvector store.

The corpus is one document per player-season (built and indexed offline by
``python -m ml.rag.index``): role, production, snap share, weekly injury
reports, and, for past seasons, what happened the next season. Each
document's vector embeds its *profile* (the season without its outcome),
using a Hugging Face embedding model run locally through fastembed.

For each analysis, the injury-focused query is embedded live and searched
against the player's own seasons in Postgres/pgvector through LangChain's
``PGVector`` store (metadata filter on the player, cosine ranking), so the
seasons with missed games and injury designations come first. Offline, this
ranks a player's most injury-affected season first 86% of the time versus
38% for "most recent season first" (``ml/reports/RAG_REPORT.md``).

A second retrieval, *comparables* (the most similar past player-seasons and
what happened to them next), was built and evaluated but is not served: on
the held-out seasons it predicted next-season scoring no better than a plain
position average, so it would add tokens without adding signal. The
evaluation stays in ``ml/rag/evaluate.py`` as the record of that decision.

The result is plain text handed to ``analyze_player(retrieved_context=...)``,
which fences it as untrusted retrieved content in the prompt.

Retrieval is optional by design. Without ``RAG_DATABASE_URL``, without the
RAG packages, or on any database or model error, :func:`retrieve_context`
returns ``None`` and the analysis runs exactly as before.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional, Protocol

from .db_url import sqlalchemy_url
from .projection import normalize_name

log = logging.getLogger(__name__)

DATABASE_ENV = "RAG_DATABASE_URL"
COLLECTION = "player_seasons"
KIND_PLAYER_SEASON = "player_season"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
MODEL_CACHE_ENV = "FASTEMBED_CACHE_PATH"

INJURY_QUERY = (
    "injury report history: games missed, listed out or questionable, "
    "injuries, missed practices"
)
HISTORY_K = 3


class _Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbeddings:
    """LangChain ``Embeddings`` backed by fastembed (ONNX, no PyTorch).

    ``embed_query`` uses the model's query instruction (BGE models are trained
    for asymmetric retrieval); ``embed_documents`` embeds passages as-is.
    The model loads on first use and is cached by fastembed under
    ``FASTEMBED_CACHE_PATH`` (the Docker images bake it in at build time).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.environ.get(MODEL_CACHE_ENV)
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._get().query_embed(text))).tolist()


def as_langchain_embeddings(embeddings: _Embeddings):
    """Wrap any object with embed_documents/embed_query as a LangChain
    ``Embeddings`` subclass (PGVector type-checks for one)."""
    from langchain_core.embeddings import Embeddings

    class _Adapter(Embeddings):
        def embed_documents(self, texts):
            return embeddings.embed_documents(texts)

        def embed_query(self, text):
            return embeddings.embed_query(text)

    return _Adapter()


def open_store(url: str, embeddings: _Embeddings, *, create: bool = False,
               pre_delete: bool = False, collection: str = COLLECTION):
    """LangChain ``PGVector`` store for the player-season collection.

    ``create=True`` (indexing only) creates the ``vector`` extension and
    tables; the services never do, so they can run as a non-owner role.
    PGVector resolves the collection by name on every query, so a running
    service picks up a re-index without a restart.
    """
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=as_langchain_embeddings(embeddings),
        connection=sqlalchemy_url(url),
        collection_name=collection,
        embedding_length=EMBEDDING_DIM,
        use_jsonb=True,
        create_extension=create,
        pre_delete_collection=pre_delete,
        engine_args={"pool_pre_ping": True, "connect_args": {"connect_timeout": 3}},
    )


def to_context(history: list) -> str:
    """Prompt text for the retrieved seasons, newest first."""
    lines = ["Injury and availability history (NFL weekly injury reports; the "
             f"{len(history)} most injury-relevant seasons, newest first):"]
    lines += [f"- {d.page_content}" for d in history]
    return "\n".join(lines)


class Retriever:
    def __init__(self, store, k: int = HISTORY_K):
        self._store = store
        self._k = k

    def retrieve(self, player_name: str) -> list:
        """The player's ``k`` most injury-relevant seasons (LangChain
        Documents), returned newest first."""
        history = self._store.similarity_search(
            INJURY_QUERY, k=self._k,
            filter={"$and": [
                {"kind": {"$eq": KIND_PLAYER_SEASON}},
                {"name_key": {"$eq": normalize_name(player_name)}},
            ]},
        )
        return sorted(history, key=lambda d: d.metadata.get("season", 0), reverse=True)


_lock = threading.Lock()
_loaded = False
_retriever: Optional[Retriever] = None


def get_retriever() -> Optional[Retriever]:
    """Process-wide retriever, built once on first use, or ``None`` if RAG
    isn't configured or its packages aren't installed."""
    global _loaded, _retriever
    if not _loaded:
        with _lock:
            if not _loaded:
                _retriever = _build_retriever()
                _loaded = True
    return _retriever


def _build_retriever() -> Optional[Retriever]:
    url = os.environ.get(DATABASE_ENV)
    if not url:
        log.info("%s is not set; retrieval disabled", DATABASE_ENV)
        return None
    try:
        return Retriever(open_store(url, FastEmbedEmbeddings()))
    except ImportError:
        log.warning("RAG packages are not installed; retrieval disabled")
    except Exception:
        log.exception("could not open the retrieval store; retrieval disabled")
    return None


def retrieve_context(player_name: str) -> Optional[str]:
    """Retrieved context for ``player_name``, or ``None``. Never raises:
    retrieval is an enhancement and must not break the analysis."""
    try:
        retriever = get_retriever()
        if retriever is None:
            return None
        history = retriever.retrieve(player_name)
        return to_context(history) if history else None
    except Exception:
        log.exception("retrieval failed for %r", player_name)
        return None


def reset_retriever() -> None:
    """Forget the cached retriever (tests, or after reindexing)."""
    global _loaded, _retriever
    with _lock:
        _loaded = False
        _retriever = None
