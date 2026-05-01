import os
import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
_PRICE_ID = os.environ.get("STRIPE_PRICE_ID")


def create_checkout_session(customer_email: str, success_url: str, cancel_url: str):
    return stripe.checkout.Session.create(
        customer_email=customer_email,
        payment_method_types=["card"],
        line_items=[{"price": _PRICE_ID, "quantity": 1}],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
    )


def verify_webhook(payload: bytes, sig_header: str):
    return stripe.Webhook.construct_event(payload, sig_header, _WEBHOOK_SECRET)
