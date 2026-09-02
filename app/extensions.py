from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()


def user_or_ip_key():
    """Rate-limit key that identifies the *user*, not the connection.

    On Render the app runs behind a reverse proxy and gunicorn does not
    rewrite ``REMOTE_ADDR`` from ``X-Forwarded-For``, so keying on the client
    IP would put every request in one shared bucket. For authenticated routes
    we key on the user id instead; the IP is only a fallback for anonymous
    requests.
    """
    if current_user and current_user.is_authenticated:
        return f"user:{current_user.get_id()}"
    return f"ip:{get_remote_address()}"


# In-memory storage (storage_uri="memory://") is fine for a single Render
# instance. Switch to a shared store (e.g. Redis) if the app is ever scaled
# horizontally, otherwise each process would track limits independently.
limiter = Limiter(
    key_func=user_or_ip_key,
    storage_uri="memory://",
)
