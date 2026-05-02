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

## Architecture

Structured with **Domain-Driven Design (DDD)** and three bounded contexts:

| Context | Responsibility |
|---|---|
| `identity` | User registration, authentication, sessions |
| `analysis` | Player name validation, AI agent, verdict parsing |
| `subscription` | Season access control, usage limits, Stripe payments |

Each context is self-contained with `domain/`, `infrastructure/`, and `presentation/` layers. A Flask app factory (`create_app`) wires everything together.

---

## Stack

- **Backend:** Python, Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt
- **Architecture:** Domain-Driven Design (DDD) with bounded contexts
- **AI:** Anthropic Claude Opus 4.7 with adaptive thinking and live web search
- **Payments:** Stripe one-time checkout
- **Database:** SQLite (local), PostgreSQL (production)
- **Frontend:** Vanilla HTML/CSS/JS (no framework dependencies)
- **Deployment:** Render

---

## Features

- **User auth** — email/password sign-up and login
- **Free tier** — 10 queries per month
- **Season pass** — $19.99 one-time payment for unlimited queries through the draft season
- **Stripe integration** — secure checkout and webhook-based access grants
- **33 tests** covering auth, analysis, usage limits, and Stripe webhooks

---

## Local setup

**1. Clone the repo**
```bash
git clone https://github.com/joshmgrey/fantasy-draft-consultant.git
cd fantasy-draft-consultant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set environment variables**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SECRET_KEY=your-random-secret-key
export STRIPE_SECRET_KEY=sk_test_...
export STRIPE_PRICE_ID=price_...
export STRIPE_WEBHOOK_SECRET=whsec_...
```

**4. Run the app**
```bash
python run.py
```

Open [http://localhost:5000](http://localhost:5000).

---

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key — [console.anthropic.com](https://console.anthropic.com) |
| `SECRET_KEY` | Flask session secret — any long random string |
| `DATABASE_URL` | PostgreSQL URL — auto-provided by Render (SQLite used locally) |
| `STRIPE_SECRET_KEY` | Stripe secret key — Developers → API keys |
| `STRIPE_PRICE_ID` | Stripe **price** ID (`price_...`) for the $19.99 season pass |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret for `/webhook` endpoint |

---

## Running tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest tests/ -v
```

---

## Deployment (Render)

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com) and connect the repo
3. Set the start command to `gunicorn run:app --bind 0.0.0.0:$PORT`
4. Add a **PostgreSQL** database and copy the `DATABASE_URL`
5. Set all environment variables in the Render dashboard
6. Add a Stripe webhook pointing to `https://your-app.onrender.com/webhook` with the `checkout.session.completed` event

---

## Project structure

```
fantasy-draft-consultant/
├── run.py                      # Entry point
├── app/
│   ├── __init__.py             # App factory (create_app)
│   ├── extensions.py           # db, bcrypt, login_manager
│   ├── identity/               # Bounded context: auth & users
│   │   ├── domain/
│   │   │   ├── models.py       # User entity
│   │   │   └── services.py     # create_user, authenticate_user
│   │   ├── infrastructure/
│   │   │   └── repository.py   # DB queries for users
│   │   └── presentation/
│   │       └── routes.py       # /login, /signup, /logout
│   ├── analysis/               # Bounded context: player research
│   │   ├── domain/
│   │   │   ├── models.py       # PlayerVerdict value object
│   │   │   └── services.py     # validate_player_name, parse_verdict
│   │   ├── infrastructure/
│   │   │   └── anthropic_client.py  # Anthropic API adapter
│   │   └── presentation/
│   │       └── routes.py       # /, /analyze
│   └── subscription/           # Bounded context: billing & access
│       ├── domain/
│       │   └── services.py     # can_query, increment_query, grant_season_access
│       ├── infrastructure/
│       │   └── stripe_client.py  # Stripe adapter
│       └── presentation/
│           └── routes.py       # /subscribe, /webhook
├── templates/
│   ├── index.html              # Main app UI
│   ├── login.html              # Login page
│   └── signup.html             # Sign-up page
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_auth.py            # Auth route tests
│   ├── test_analyze.py         # Analysis and usage limit tests
│   └── test_webhook.py         # Stripe webhook tests
├── draft_consultant.py         # Standalone CLI version
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Test dependencies
└── Procfile                    # Render process config
```

---

## Security

- Player name input validated against a strict allowlist (letters, spaces, apostrophes, hyphens, periods only)
- Player names wrapped in XML delimiters in the prompt to prevent injection
- Passwords hashed with bcrypt
- Stripe webhook signature verified on every request
- API errors surface cleanly in the UI — no raw tracebacks exposed
