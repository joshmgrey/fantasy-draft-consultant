"""
Fantasy Football Draft Consultant — 2026 PPR
Web app: python app.py  →  http://localhost:5000
"""

import os
import re
import sys
import anthropic
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

_api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=_api_key) if _api_key else None

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = ""

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z '\-\.]{0,48}$")
_MAX_NAME_LEN = 50

SYSTEM_PROMPT = """You are an elite fantasy football analyst specializing in 2026 PPR drafts.
Your edge is identifying VALUE OVER ADP — players whose projected output exceeds their draft cost.

When evaluating a player, you MUST:
1. Search for their 2026 PPR fantasy projections (points, position rank, ADP)
2. Search for recent news (trades, depth chart changes, target share, snap counts)
3. Search for injury history and current injury risk

Then deliver a concise verdict:
- RISK SCORE: 1–10 (1 = zero risk, 10 = extreme risk/injury-prone/situation unclear)
- RECOMMENDATION: "Draft" or "Pass"
- JUSTIFICATION: One sentence focusing on VALUE vs ADP (not just talent)

Format your final answer EXACTLY as:
RISK: <number>/10
VERDICT: <Draft|Pass>
REASON: <one sentence>

Be direct. No preamble. No lengthy analysis. Just the three lines above."""

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Email and password are required.")
            return render_template("signup.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return render_template("signup.html")
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.")
            return render_template("signup.html")
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email or password.")
            return render_template("login.html")
        login_user(user)
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# App routes
# ---------------------------------------------------------------------------

def _validate_player_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Player name cannot be empty.")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"Player name too long (max {_MAX_NAME_LEN} characters).")
    if not _NAME_RE.match(name):
        raise ValueError(
            "Player name contains invalid characters. "
            "Use letters, spaces, apostrophes, hyphens, or periods only."
        )
    return name


def _analyze_player(player_name: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze the player named <player>{player_name}</player> for a 2026 PPR fantasy football draft. "
                "Search for their 2026 projections, ADP, recent news, and injury risk, "
                "then give me your Risk Score, Verdict, and Reason."
            ),
        }
    ]

    iterations = 0
    while iterations < 10:
        iterations += 1

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason == "pause_turn":
            continue
        break

    final_text = ""
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            final_text = block.text
            break

    return _parse_verdict(player_name, final_text)


_TAG_RE = re.compile(r"<[^>]+>")


def _parse_verdict(player_name: str, text: str) -> dict:
    result = {
        "player": player_name,
        "risk_score": None,
        "verdict": None,
        "reason": None,
    }

    for line in text.strip().splitlines():
        line = _TAG_RE.sub("", line).strip()
        if line.upper().startswith("RISK:"):
            val = line.split(":", 1)[1].strip().split("/")[0].strip()
            try:
                result["risk_score"] = int(val)
            except ValueError:
                pass
        elif line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip()
            result["verdict"] = "Draft" if "draft" in val.lower() else "Pass"
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if not client:
        return jsonify({"error": "ANTHROPIC_API_KEY environment variable is not set."}), 500

    data = request.get_json(silent=True) or {}
    raw_name = data.get("player_name", "")

    try:
        player_name = _validate_player_name(raw_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = _analyze_player(player_name)
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error ({e.status_code}): {e.message}"}), 502
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not reach Anthropic API. Check your connection."}), 502

    return jsonify(result)


if __name__ == "__main__":
    if not _api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
