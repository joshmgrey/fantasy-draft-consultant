# 2026 PPR Fantasy Football Draft Consultant

An AI-powered draft assistant that researches players in real time and delivers instant Draft/Pass verdicts during your fantasy football draft.

Built with Claude Opus 4.7 and live web search — type a player name, get a risk score, recommendation, and one-sentence justification in seconds.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey) ![Claude](https://img.shields.io/badge/Claude-Opus%204.7-orange)

---

## How it works

1. You enter a player name in the web UI
2. The agent autonomously searches for their 2026 PPR projections, ADP, recent news, and injury history
3. Claude reasons over the results and returns a structured verdict:
   - **Risk Score** — 1–10 (1 = zero risk, 10 = extreme risk)
   - **Verdict** — Draft or Pass
   - **Reason** — One sentence focused on value over ADP

The agent uses Claude's server-side `web_search` tool, meaning Anthropic executes the searches automatically — no third-party search API needed.

---

## Stack

- **Backend:** Python, Flask
- **AI:** Anthropic Claude Opus 4.7 with adaptive thinking and live web search tool use
- **Frontend:** Vanilla HTML/CSS/JS (no framework dependencies)

---

## Setup

**1. Clone the repo**
```bash
git clone <repo-url>
cd Fantasy-Agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Get a key at [console.anthropic.com](https://console.anthropic.com).

**4. Run the app**
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## CLI mode

A standalone CLI version is also available if you prefer the terminal:

```bash
# Single player
python draft_consultant.py "Ja'Marr Chase"

# Interactive mode
python draft_consultant.py
```

---

## Project structure

```
Fantasy-Agent/
├── app.py                  # Flask web app + Anthropic agent logic
├── draft_consultant.py     # Standalone CLI version
├── requirements.txt
└── templates/
    └── index.html          # Single-page web UI
```

---

## Security

- Player name input is validated against a strict allowlist (letters, spaces, apostrophes, hyphens, periods only)
- Player names are wrapped in XML delimiters in the prompt to prevent injection attacks
- API errors surface cleanly in the UI — no raw tracebacks exposed

---

## Example output

```
─────────────────────────────────────────────
  JA'MARR CHASE
─────────────────────────────────────────────
  ✅  DRAFT
  RISK  [███░░░░░░░]  3/10
  WHY   Chase's target share and YAC upside
        make him a value at his current ADP
        given his elite floor in a pass-heavy
        offense.
─────────────────────────────────────────────
```
