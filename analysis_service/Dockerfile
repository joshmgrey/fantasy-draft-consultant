# Analysis service. Build context is the repo root so analysis_core comes along.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY analysis_service/requirements.txt .
RUN pip install -r requirements.txt

# Bake the retrieval embedding model into the image so the first request
# doesn't download it (~130 MB, Hugging Face via fastembed).
ENV FASTEMBED_CACHE_PATH=/app/.fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed')"

COPY analysis_core/ analysis_core/
COPY analysis_service/ analysis_service/

EXPOSE 8000
CMD ["gunicorn", "analysis_service.wsgi:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
