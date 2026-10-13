"""Database URL normalization, shared by the core app and the analysis service.

Postgres URLs from Render, compose and k8s come as ``postgres://`` or
``postgresql://`` with no driver named. SQLAlchemy 2.0 resolves a bare
``postgresql://`` to psycopg2 and 2.1 to psycopg (3), so leaving the driver
implicit ties the app to whichever SQLAlchemy happens to be installed. Name
the driver we ship (``psycopg``, version 3) explicitly instead.
"""

_DRIVER = "postgresql+psycopg://"
_BARE_PREFIXES = ("postgres://", "postgresql://")


def sqlalchemy_url(url: str) -> str:
    """Return ``url`` with an explicit psycopg driver if it is a bare Postgres URL.

    URLs that already name a driver (``postgresql+...://``) and non-Postgres
    URLs (e.g. ``sqlite://``) are returned unchanged.
    """
    for prefix in _BARE_PREFIXES:
        if url.startswith(prefix):
            return _DRIVER + url[len(prefix):]
    return url
