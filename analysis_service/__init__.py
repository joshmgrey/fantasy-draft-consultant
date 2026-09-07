"""Standalone deployable that owns the analysis bounded context.

Run from the repo root so ``analysis_core`` is importable::

    gunicorn analysis_service.wsgi:app --bind 0.0.0.0:$PORT
"""
