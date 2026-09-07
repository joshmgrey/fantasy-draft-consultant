import os

from flask import Flask
from werkzeug.exceptions import HTTPException

from analysis_core.contract import ErrorCode

from .api import bp
from .schemas import error_response


def create_app(config=None):
    app = Flask(__name__)

    app.config["ANALYSIS_TOKEN_SECRET"] = os.environ.get("ANALYSIS_TOKEN_SECRET", "")
    app.config["TOKEN_LEEWAY_SECONDS"] = int(
        os.environ.get("ANALYSIS_TOKEN_LEEWAY_SECONDS", "5")
    )

    if config:
        app.config.update(config)

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
