from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()

# In-memory storage (storage_uri="memory://") is fine for a single Render
# instance. Switch to a shared store (e.g. Redis) if the app is ever scaled
# horizontally, otherwise each process would track limits independently.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://",
)
