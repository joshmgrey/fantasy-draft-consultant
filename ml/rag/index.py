"""Build the retrieval corpus and load it into Postgres/pgvector.

    python -m ml.download                                  # stats, injuries, snap counts
    RAG_DATABASE_URL=postgresql://... python -m ml.rag.index

Rebuilds the ``player_seasons`` collection from scratch each run: one
document per QB/RB/WR/TE season, embedded with the same fastembed model the
services use for queries. Each vector embeds the season's *profile* (no
outcome); the stored text includes the next-season outcome for the LLM.

Creates the ``vector`` extension if needed, so run it as a role allowed to
(the services only read). Retrain-style cadence: rerun each May after the
season rolls over.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from analysis_core.retrieval import DATABASE_ENV, FastEmbedEmbeddings, open_store
from analysis_core.season import current_season

from ..train import RAW_DIR
from .documents import build_documents

BATCH = 256


def index(url: str, raw_dir: Path, first: int, last: int, embeddings=None,
          batch: int = BATCH) -> int:
    docs = build_documents(raw_dir, first, last)
    embeddings = embeddings or FastEmbedEmbeddings()
    store = open_store(url, embeddings, create=True, pre_delete=True)
    for start in range(0, len(docs), batch):
        chunk = docs[start:start + batch]
        store.add_embeddings(
            texts=[d.page_content for d in chunk],
            embeddings=embeddings.embed_documents([d.metadata["profile"] for d in chunk]),
            metadatas=[d.metadata for d in chunk],
            ids=[f"{d.metadata['player_id']}-{d.metadata['season']}" for d in chunk],
        )
    return len(docs)


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
