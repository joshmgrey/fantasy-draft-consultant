from datetime import datetime, date
from app.extensions import db
# The season calendar is shared with the analysis service (the AI researches
# the same season the pass is sold for), so it lives in analysis_core.
from analysis_core.season import current_season, season_expiry  # noqa: F401

FREE_QUERY_LIMIT = 10


def has_season_access(user) -> bool:
    return user.plan == "seasonal" and user.plan_expires and date.today() < user.plan_expires


def _sync_month(user) -> None:
    current_month = datetime.utcnow().strftime("%Y-%m")
    if user.query_month != current_month:
        user.queries_this_month = 0
        user.query_month = current_month
        db.session.commit()


def can_query(user) -> bool:
    if has_season_access(user):
        return True
    _sync_month(user)
    return user.queries_this_month < FREE_QUERY_LIMIT


def queries_remaining(user):
    if has_season_access(user):
        return None
    _sync_month(user)
    return max(0, FREE_QUERY_LIMIT - user.queries_this_month)


def increment_query(user) -> None:
    _sync_month(user)
    user.queries_this_month += 1
    db.session.commit()


def grant_season_access(user, stripe_customer_id: str) -> None:
    user.plan = "seasonal"
    user.plan_expires = season_expiry()
    user.stripe_customer_id = stripe_customer_id
    db.session.commit()
