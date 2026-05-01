from app.extensions import db
from .models import User


def create_user(email: str, password: str) -> User:
    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def authenticate_user(email: str, password: str):
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return user
    return None


def email_exists(email: str) -> bool:
    return User.query.filter_by(email=email).first() is not None
