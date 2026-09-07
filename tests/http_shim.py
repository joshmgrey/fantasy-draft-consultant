"""Adapt a Flask test client to the slice of ``requests.Session`` that
``HttpAnalysisClient`` uses, so the core app and the real analysis service can
be exercised together in-process — no sockets, no ports.
"""

from urllib.parse import urlparse

import requests


class StubResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


class AppSession:
    """Forwards ``.post(url, json=, headers=, timeout=)`` into ``flask_app``."""

    def __init__(self, flask_app):
        self._client = flask_app.test_client()
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        res = self._client.post(urlparse(url).path, json=json, headers=headers)
        return StubResponse(res.status_code, res.get_json(silent=True))


class DeadSession:
    """Simulates a service that cannot be reached — every call raises."""

    def __init__(self, exc=None):
        self._exc = exc or requests.ConnectionError("connection refused")
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        raise self._exc
