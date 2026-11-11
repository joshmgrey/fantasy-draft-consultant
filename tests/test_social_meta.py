"""Link-preview metadata (Open Graph + Twitter cards) and the preview image."""

import re
import struct

import pytest

PAGES = ["/login", "/signup"]


def meta(html: str, attr: str, name: str) -> str:
    m = re.search(rf'<meta {attr}="{re.escape(name)}" content="([^"]*)"', html)
    assert m, f"missing {attr}={name}"
    return m.group(1)


@pytest.mark.parametrize("path", PAGES)
def test_public_pages_have_link_preview_tags(client, path):
    html = client.get(path).get_data(as_text=True)
    assert meta(html, "property", "og:title") == "PPR Fantasy Football Draft Consultant"
    assert meta(html, "property", "og:url") == "https://fantasy-football-draft-consultant.com/"
    assert meta(html, "property", "og:image") == (
        "https://fantasy-football-draft-consultant.com/static/og-image.png"
    )
    assert meta(html, "property", "og:image:width") == "1200"
    assert meta(html, "property", "og:image:height") == "630"
    assert meta(html, "name", "twitter:card") == "summary_large_image"
    assert meta(html, "name", "twitter:image") == meta(html, "property", "og:image")
    assert 50 <= len(meta(html, "name", "description")) <= 200


def test_logged_in_page_has_the_same_tags(client, free_user):
    client.post("/login", data={"email": "free@test.com", "password": "password123"})
    html = client.get("/").get_data(as_text=True)
    assert meta(html, "property", "og:image").endswith("/static/og-image.png")


def test_public_url_is_configurable(app, client):
    app.config["PUBLIC_URL"] = "https://staging.example.com/"
    html = client.get("/login").get_data(as_text=True)
    assert meta(html, "property", "og:url") == "https://staging.example.com/"
    assert meta(html, "property", "og:image") == "https://staging.example.com/static/og-image.png"


def test_preview_image_is_served_at_1200x630(client):
    res = client.get("/static/og-image.png")
    assert res.status_code == 200
    assert res.mimetype == "image/png"
    data = res.get_data()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])   # IHDR
    assert (width, height) == (1200, 630)
    # Most platforms ignore preview images over ~5 MB.
    assert len(data) < 1_000_000
