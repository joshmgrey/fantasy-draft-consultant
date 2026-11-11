"""Generate the 1200x630 link-preview image (Open Graph / Twitter card).

    pip install pillow
    python scripts/og_image.py            # writes app/static/og-image.png

Drawn in the site's own palette (templates/*.html) with a sample result card,
so a shared link previews what the app does. Uses Segoe UI on Windows or
DejaVu Sans elsewhere; rerun after changing the copy below.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "og-image.png"
W, H = 1200, 630

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#3fb950"
GREEN_BG = "#0d2818"
TRACK = "#21262d"

TITLE = "PPR Fantasy Football\nDraft Consultant"
SUBTITLE = "AI research, a 1–10 risk score, and a\nDraft or Pass verdict in seconds."
DOMAIN = "fantasy-football-draft-consultant.com"

FONT_CANDIDATES = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[weight]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    sys.exit(f"no {weight} font found; edit FONT_CANDIDATES")


def draw() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Accent bar and the left column of copy.
    d.rectangle([0, 0, W, 8], fill=GREEN)
    d.text((72, 92), "AI DRAFT ASSISTANT", font=font("bold", 26), fill=GREEN)
    d.multiline_text((72, 140), TITLE, font=font("bold", 64), fill=TEXT, spacing=8)
    d.multiline_text((72, 318), SUBTITLE, font=font("regular", 32), fill=MUTED, spacing=10)
    d.text((72, 530), DOMAIN, font=font("bold", 28), fill=TEXT)

    # A sample result card, like the app's.
    x0, y0, x1, y1 = 760, 150, 1128, 480
    d.rounded_rectangle([x0, y0, x1, y1], radius=18, fill=PANEL, outline=BORDER, width=2)
    d.text((x0 + 32, y0 + 30), "Bijan Robinson", font=font("bold", 34), fill=TEXT)
    d.rounded_rectangle([x0 + 32, y0 + 92, x0 + 196, y0 + 146], radius=10, fill=GREEN_BG)
    d.text((x0 + 56, y0 + 98), "DRAFT", font=font("bold", 32), fill=GREEN)
    d.text((x0 + 32, y0 + 180), "RISK SCORE", font=font("bold", 20), fill=MUTED)
    track = [x0 + 32, y0 + 220, x1 - 100, y0 + 236]
    d.rounded_rectangle(track, radius=8, fill=TRACK)
    fill_to = track[0] + (track[2] - track[0]) * 3 // 10
    d.rounded_rectangle([track[0], track[1], fill_to, track[3]], radius=8, fill=GREEN)
    d.text((x1 - 84, y0 + 210), "3/10", font=font("bold", 28), fill=GREEN)
    d.text((x0 + 32, y0 + 266), "Value over ADP · PPR", font=font("regular", 24), fill=MUTED)
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw().save(OUT, optimize=True)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
