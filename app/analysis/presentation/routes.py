from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user

from app.extensions import limiter
from analysis_core.services import validate_player_name
from app.analysis.client.base import (
    Actor,
    AnalysisBadRequest,
    AnalysisNotConfigured,
    AnalysisRateLimited,
    AnalysisServiceDown,
    AnalysisUnavailable,
)
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
@limiter.limit("10 per minute", exempt_when=lambda: not current_user.is_authenticated)
def analyze():
    if not can_query(current_user):
        return jsonify({"error": "free_limit_reached"}), 403

    data = request.get_json(silent=True) or {}
    raw_name = data.get("player_name", "")

    try:
        player_name = validate_player_name(raw_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    analysis_client = current_app.extensions["analysis_client"]
    try:
        result = analysis_client.analyze(
            player_name=player_name,
            actor=Actor.from_user(current_user),
        )
    except AnalysisBadRequest as e:
        return jsonify({"error": str(e)}), 400
    except AnalysisNotConfigured as e:
        return jsonify({"error": str(e)}), 500
    except (AnalysisRateLimited, AnalysisServiceDown) as e:
        return jsonify({"error": str(e)}), 503
    except AnalysisUnavailable as e:
        return jsonify({"error": str(e)}), 502

    increment_query(current_user)
    return jsonify(result.to_dict())
