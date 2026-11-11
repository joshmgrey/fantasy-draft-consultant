"""Build the retrieval corpus and load it into Postgres/pgvector.

    python -m ml.download                                  # stats, injuries, snap counts
    RAG_DATABASE_URL=postgresql://... python -m ml.rag.index

Rebuilds the ``player_seasons`` collection each run: one document per
QB/RB/WR/TE season, embedded with the same fastembed model the services use
for queries. Each vector embeds the season's *profile* (no outcome); the
stored text includes the next-season outcome for the LLM.

The rebuild is **blue/green**: documents load into a staging collection
(several minutes of embedding), then one transaction deletes the live
collection and renames staging into its place. Running services keep
reading the old collection until that commit, and PGVector looks the
collection up by name on every query, so they pick up the new one without a
restart. Row IDs carry a per-build prefix because PGVector upserts on a
globally unique ID: reusing the live IDs would overwrite live rows instead
of staging new ones.

Creates the ``vector`` extension if needed, so run it as a role allowed to
(the services only read). Rerun each May after the season rolls over.
"""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from analysis_core.db_url import sqlalchemy_url
from analysis_core.retrieval import COLLECTION, DATABASE_ENV, FastEmbedEmbeddings, open_store
from analysis_core.season import current_season

from ..train import RAW_DIR
from .documents import build_documents

BATCH = 256
STAGING = f"{COLLECTION}__staging"


def index(url: str, raw_dir: Path, first: int, last: int, embeddings=None,
          batch: int = BATCH) -> int:
    docs = build_documents(raw_dir, first, last)
    embeddings = embeddings or FastEmbedEmbeddings()
    build = uuid.uuid4().hex[:8]
    # pre_delete clears any staging left behind by an interrupted run.
    store = open_store(url, embeddings, create=True, pre_delete=True, collection=STAGING)
    for start in range(0, len(docs), batch):
        chunk = docs[start:start + batch]
        store.add_embeddings(
            texts=[d.page_content for d in chunk],
            embeddings=embeddings.embed_documents([d.metadata["profile"] for d in chunk]),
            metadatas=[d.metadata for d in chunk],
            ids=[f"{build}:{d.metadata['player_id']}-{d.metadata['season']}" for d in chunk],
        )
    swap_in_staging(url)
    return len(docs)


def swap_in_staging(url: str) -> None:
    """Atomically replace the live collection with the staged one.

    Deleting the live collection row removes its embeddings through the
    ``ON DELETE CASCADE`` foreign key; readers see either the old or the new
    collection, never neither.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(sqlalchemy_url(url))
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM langchain_pg_collection WHERE name = :live"),
                         {"live": COLLECTION})
            renamed = conn.execute(
                text("UPDATE langchain_pg_collection SET name = :live WHERE name = :staging"),
                {"live": COLLECTION, "staging": STAGING},
            ).rowcount
            if renamed != 1:
                raise RuntimeError(f"expected one staged collection, found {renamed}")
    finally:
        engine.dispose()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get(DATABASE_ENV),
                        help=f"Postgres URL with pgvector (default: ${DATABASE_ENV})")
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--first", type=int, default=2012)
    parser.add_argument("--last", type=int, default=current_season() - 1)
    args = parser.parse_args(argv)
    if not args.database_url:
        parser.error(f"pass --database-url or set {DATABASE_ENV}")
    n = index(args.database_url, args.raw, args.first, args.last)
    print(f"indexed {n} player-season documents")


if __name__ == "__main__":
    main()
