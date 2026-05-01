import stripe
from flask import Blueprint, redirect, request
from flask_login import login_required, current_user

from app.identity.infrastructure.repository import get_by_email
from app.subscription.domain.services import grant_season_access
from app.subscription.infrastructure.stripe_client import create_checkout_session, verify_webhook

bp = Blueprint("subscription", __name__)


@bp.route("/subscribe")
@login_required
def subscribe():
    session = create_checkout_session(
        customer_email=current_user.email,
        success_url=request.host_url.rstrip("/") + "/?purchased=1",
        cancel_url=request.host_url.rstrip("/") + "/",
    )
    return redirect(session.url)


@bp.route("/webhook", methods=["POST"])
def webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = verify_webhook(payload, sig_header)
    except (ValueError, stripe.error.SignatureVerificationError):
        return "", 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user = get_by_email(session.get("customer_email"))
        if user:
            grant_season_access(user, session.get("customer"))

    return "", 200
