import anthropic as anthropic_sdk
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user

from app.analysis.domain.services import validate_player_name, parse_verdict
from app.analysis.infrastructure.anthropic_client import analyze_player, client
from app.subscription.domain.services import (
    can_query, increment_query, queries_remaining, has_season_access,
    current_season, FREE_QUERY_LIMIT,
)

bp = Blueprint("analysis", __name__)


@bp.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        has_access=has_season_access(current_user),
        season=current_season(),
        queries_remaining=queries_remaining(current_user),
        free_limit=FREE_QUERY_LIMIT,
        email=current_user.email,
    )


@bp.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if not client:
        return jsonify({"error": "ANTHROPIC_API_KEY environment variable is not set."}), 500

    if not can_query(current_user):
        return jsonify({"error": "free_limit_reached"}), 403

    data = request.get_json(silent=True) or {}
    raw_name = data.get("player_name", "")

    try:
        player_name = validate_player_name(raw_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        raw_text = analyze_player(player_name)
        result = parse_verdict(player_name, raw_text)
    except anthropic_sdk.APIStatusError as e:
        return jsonify({"error": f"API error ({e.status_code}): {e.message}"}), 502
    except anthropic_sdk.APIConnectionError:
        return jsonify({"error": "Could not reach Anthropic API. Check your connection."}), 502

    increment_query(current_user)
    return jsonify(result.to_dict())
