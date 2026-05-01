from app.extensions import db
from app.identity.domain.models import User


def get_by_id(user_id: int):
    return db.session.get(User, int(user_id))


def get_by_email(email: str):
    return User.query.filter_by(email=email).first()
