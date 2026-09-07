import os

from flask import Flask
from werkzeug.exceptions import HTTPException

from analysis_core.contract import ErrorCode

from .api import bp
from .persistence import db
from .schemas import error_response


def create_app(config=None):
    app = Flask(__name__)

    app.config["ANALYSIS_TOKEN_SECRET"] = os.environ.get("ANALYSIS_TOKEN_SECRET", "")
    app.config["TOKEN_LEEWAY_SECONDS"] = int(
        os.environ.get("ANALYSIS_TOKEN_LEEWAY_SECONDS", "5")
    )
    # The service's own database. Unset -> ephemeral in-memory (fine for tests
    # and a "you didn't configure it" local run); compose/prod set
    # ANALYSIS_DATABASE_URL to a dedicated Postgres.
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "ANALYSIS_DATABASE_URL", "sqlite:///:memory:"
    ).replace("postgres://", "postgresql://")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["PERSIST_ANALYSES"] = True

    if config:
        app.config.update(config)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    app.register_blueprint(bp)

    @app.errorhandler(404)
    def _not_found(_e):
        return error_response(ErrorCode.MALFORMED_REQUEST, "no such endpoint", status=404)

    @app.errorhandler(405)
    def _method_not_allowed(_e):
        return error_response(ErrorCode.MALFORMED_REQUEST, "method not allowed", status=405)

    @app.errorhandler(Exception)
    def _unhandled(e):
        if isinstance(e, HTTPException):
            return e
        # Never leak a traceback across the boundary.
        app.logger.exception("unhandled error in analysis service")
        return error_response(ErrorCode.UNAVAILABLE, "internal error", status=500)

    return app
