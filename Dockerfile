# Core service: identity + subscription + the analysis client.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY analysis_core/ analysis_core/
COPY app/ app/
COPY templates/ templates/
COPY run.py .

EXPOSE 8000
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
