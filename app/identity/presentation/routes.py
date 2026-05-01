from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from app.identity.domain.services import create_user, authenticate_user, email_exists

bp = Blueprint("identity", __name__)


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("analysis.index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.")
            return render_template("signup.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return render_template("signup.html")
        if email_exists(email):
            flash("An account with that email already exists.")
            return render_template("signup.html")
        user = create_user(email, password)
        login_user(user)
        return redirect(url_for("analysis.index"))
    return render_template("signup.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("analysis.index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = authenticate_user(email, password)
        if not user:
            flash("Invalid email or password.")
            return render_template("login.html")
        login_user(user)
        return redirect(url_for("analysis.index"))
    return render_template("login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("identity.login"))
