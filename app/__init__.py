import os
from flask import Flask, jsonify
from .extensions import db, bcrypt, login_manager, limiter


def create_app(config=None):
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    )

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
    _db_url = os.environ.get("DATABASE_URL", "sqlite:///app.db").replace("postgres://", "postgresql://")
    app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    if config:
        app.config.update(config)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "identity.login"
    login_manager.login_message = ""

    limiter.init_app(app)
    # Rate limiting only gets in the way of the test suite, which fires many
    # requests at the same endpoint from one client.
    if app.config.get("TESTING"):
        limiter.enabled = False

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return (
            jsonify(error="Too many requests. Please wait a moment and try again."),
            429,
        )

    from .identity.infrastructure.repository import get_by_id

    @login_manager.user_loader
    def load_user(user_id):
        return get_by_id(user_id)

    from .identity.presentation.routes import bp as identity_bp
    from .analysis.presentation.routes import bp as analysis_bp
    from .subscription.presentation.routes import bp as subscription_bp

    app.register_blueprint(identity_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(subscription_bp)

    with app.app_context():
        db.create_all()

    return app
