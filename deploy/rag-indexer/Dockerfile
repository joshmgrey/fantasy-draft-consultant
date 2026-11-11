# One-off RAG indexer: downloads the nflverse files and (re)builds the
# player_seasons pgvector collection with `python -m ml.rag.index`.
# Not a service. infra/ runs it as an ECS task (see infra/README.md,
# "Building the RAG index"). Build context is the repo root.
#
# Same dependencies and embedding model as the analysis service, so the
# documents are embedded exactly as the service embeds its queries, plus
# pandas and the ml/ package, which the service image leaves out.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY analysis_service/requirements.txt .
RUN pip install -r requirements.txt "pandas>=2.2"

ENV FASTEMBED_CACHE_PATH=/app/.fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed')"

COPY analysis_core/ analysis_core/
COPY ml/ ml/

# RAG_DATABASE_URL must be set (the ECS task builds it from the RDS secret).
CMD ["sh", "-c", "python -m ml.download && python -m ml.rag.index"]
