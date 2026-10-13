import pytest

from analysis_core.db_url import sqlalchemy_url


@pytest.mark.parametrize("raw,expected", [
    # Render hands out postgres://; compose and k8s use postgresql://
    ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
    ("postgresql://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
    # An explicit driver is the operator's choice: leave it alone
    ("postgresql+psycopg2://u:p@host/db", "postgresql+psycopg2://u:p@host/db"),
    ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
    # Not Postgres
    ("sqlite:///app.db", "sqlite:///app.db"),
    ("sqlite:///:memory:", "sqlite:///:memory:"),
])
def test_sqlalchemy_url(raw, expected):
    assert sqlalchemy_url(raw) == expected


def test_only_the_scheme_is_rewritten():
    """A 'postgres://' later in the URL (e.g. inside a password) is not touched."""
    raw = "postgresql://u:postgres://x@host/db"
    assert sqlalchemy_url(raw) == "postgresql+psycopg://u:postgres://x@host/db"


def test_url_resolves_to_the_psycopg3_dialect():
    """Whatever SQLAlchemy version is installed, the driver is the one we ship."""
    from sqlalchemy.engine import make_url

    dialect = make_url(sqlalchemy_url("postgres://u:p@host/db")).get_dialect()
    assert dialect.driver == "psycopg"
