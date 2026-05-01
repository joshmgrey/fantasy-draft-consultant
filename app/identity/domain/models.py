from flask_login import UserMixin
from app.extensions import db, bcrypt


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    plan = db.Column(db.String(20), default="free", nullable=False)
    plan_expires = db.Column(db.Date, nullable=True)
    queries_this_month = db.Column(db.Integer, default=0, nullable=False)
    query_month = db.Column(db.String(7), default="")
    stripe_customer_id = db.Column(db.String(100), nullable=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
