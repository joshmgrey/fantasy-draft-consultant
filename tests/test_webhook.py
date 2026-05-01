import json
import pytest
from unittest.mock import patch
from app import db, User


def post_webhook(client, payload, signature="valid"):
    return client.post(
        "/webhook",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"Stripe-Signature": signature},
    )


def test_webhook_invalid_signature(client):
    with patch("stripe.Webhook.construct_event", side_effect=ValueError("bad sig")):
        res = post_webhook(client, {})
    assert res.status_code == 400


def test_webhook_checkout_completed_grants_season_access(client, free_user, app):
    event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer_email": "free@test.com",
                "customer": "cus_abc123",
            }
        },
    }
    with patch("stripe.Webhook.construct_event", return_value=event):
        res = post_webhook(client, event)
    assert res.status_code == 200
    with app.app_context():
        user = User.query.filter_by(email="free@test.com").first()
        assert user.plan == "seasonal"
        assert user.plan_expires is not None
        assert user.stripe_customer_id == "cus_abc123"


def test_webhook_unknown_event_ignored(client):
    event = {"type": "invoice.paid", "data": {"object": {}}}
    with patch("stripe.Webhook.construct_event", return_value=event):
        res = post_webhook(client, event)
    assert res.status_code == 200


def test_webhook_checkout_unknown_email_ignored(client, app):
    event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer_email": "nobody@test.com",
                "customer": "cus_xyz",
                "subscription": "sub_xyz",
            }
        },
    }
    with patch("stripe.Webhook.construct_event", return_value=event):
        res = post_webhook(client, event)
    assert res.status_code == 200
