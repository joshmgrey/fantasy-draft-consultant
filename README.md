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

Structured with **Domain-Driven Design (DDD)**. Three bounded contexts, each
self-contained with `domain/`, `infrastructure/`, and `presentation/` layers:

| Context | Responsibility | Lives in |
|---|---|---|
| `identity` | User registration, authentication, sessions | core app |
| `subscription` | Season access control, usage limits, Stripe payments | core app |
| `analysis` | Player-name validation, the Claude agent, verdict parsing | `analysis_core` (shared library) |

The **analysis** context runs one of two ways, selected by `ANALYSIS_MODE`:

- **`inprocess`** (default) — the core app calls `analysis_core` directly. This
  is the monolith; `python run.py` needs nothing extra.
- **`http`** — the core app calls a **standalone analysis service** over HTTP.
  The two services have separate databases and talk only through a versioned
  JSON contract (`analysis_core/contract.py`). This is what `docker compose` runs.

Either way the core `/analyze` route goes through one `AnalysisClient` interface
(`app/analysis/client/`), so the request flow and the tests are identical for
both modes.

### The core ↔ analysis contract (`http` mode)

| | |
|---|---|
| `POST /v1/analyses` | `{player_name, request_id?}` → `{player, risk_score, verdict, reason, model, notice, analysis_id}` |
| `GET /v1/analyses?user_id=` | analysis history for a user (the service's audit log) |
| `GET /healthz` | liveness + whether `ANTHROPIC_API_KEY` is set |
| Auth | the core app signs a short-lived HMAC token (`ANALYSIS_TOKEN_SECRET`, shared) naming the user; the service verifies it and never imports the identity context |
| Errors | `invalid_player_name` 400 · `unauthorized` 401 · `malformed_request` 422 · `upstream_rate_limited` 429 · `upstream_error` 502 · `unavailable` 503 |

---

## Stack

- **Backend:** Python, Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt
- **Architecture:** Domain-Driven Design (DDD); optional core + analysis split over HTTP (`requests`, no RPC framework)
- **AI:** Anthropic Claude Opus 4.7 with adaptive thinking and live web search
- **Payments:** Stripe one-time checkout
- **Database:** SQLite (local), PostgreSQL (production); the analysis service has its own
- **Frontend:** Vanilla HTML/CSS/JS (no framework dependencies)
- **Deployment:** Render; `docker-compose.yml` for the full local split stack

---

## Features

- **User auth** — email/password sign-up and login
- **Free tier** — 10 queries per month
- **Season pass** — $19.99 one-time payment for unlimited queries through the draft season
- **Stripe integration** — secure checkout and webhook-based access grants
- **Rate limiting** — per-user cap on `/analyze` plus retry-with-backoff on Anthropic 429s
- **157 tests** (106 core + 51 analysis service) — auth, analysis, usage limits, Stripe webhooks, prompt-injection defenses, rate limiting, the HTTP client, the service contract, and the analysis-service-down failure mode

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

Open [http://localhost:5000](http://localhost:5000). This runs the monolith
(`ANALYSIS_MODE=inprocess`).

### Full split stack (core + analysis + a database each)

```bash
cp .env.example .env    # fill in ANTHROPIC_API_KEY and ANALYSIS_TOKEN_SECRET
docker compose up --build
```

Runs the core app in `http` mode against the standalone analysis service, each
with its own Postgres. Open [http://localhost:5000](http://localhost:5000). The
analysis service is not published to the host — only the core app reaches it.

---

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key — [console.anthropic.com](https://console.anthropic.com). In `http` mode only the **analysis service** needs it, not core. |
| `SECRET_KEY` | Flask session secret — any long random string |
| `DATABASE_URL` | Core PostgreSQL URL — auto-provided by Render (SQLite used locally) |
| `STRIPE_SECRET_KEY` | Stripe secret key — Developers → API keys |
| `STRIPE_PRICE_ID` | Stripe **price** ID (`price_...`) for the $19.99 season pass |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret for `/webhook` endpoint |
| `ANALYSIS_MODE` | `inprocess` (default) or `http` |
| `ANALYSIS_SERVICE_URL` | Base URL of the analysis service (`http` mode) |
| `ANALYSIS_TOKEN_SECRET` | Shared HMAC secret for the core→analysis actor token (`http` mode) — **must be identical on both services** |
| `ANALYSIS_CONNECT_TIMEOUT` / `ANALYSIS_READ_TIMEOUT` | Core→analysis HTTP timeouts in seconds (default `3` / `90`) |
| `ANALYSIS_DATABASE_URL` | PostgreSQL URL for the **analysis service's own** database |

---

## Running tests

```bash
pip install -r requirements.txt -r analysis_service/requirements.txt -r requirements-dev.txt
pytest                          # all 157 (core + analysis service)
pytest tests/                   # just the core app (106)
pytest analysis_service/tests/  # just the analysis service (51)
```

`pytest.ini` sets `pythonpath`, so no `PYTHONPATH=` prefix is needed.

---

## Deployment (Render)

### Single service (monolith)

1. Create a **Web Service** with start command `gunicorn run:app --bind 0.0.0.0:$PORT`
2. Add a **PostgreSQL** database → `DATABASE_URL`
3. Set `ANTHROPIC_API_KEY`, `SECRET_KEY`, and the Stripe vars (leave `ANALYSIS_MODE` unset)
4. Add a Stripe webhook to `https://your-app.onrender.com/webhook` for `checkout.session.completed`

### Two services (`http` mode)

Deploy both from this repo:

| | Core | Analysis |
|---|---|---|
| Start command | `gunicorn run:app --bind 0.0.0.0:$PORT` | `gunicorn analysis_service.wsgi:app --bind 0.0.0.0:$PORT` |
| Database | `DATABASE_URL` | `ANALYSIS_DATABASE_URL` |
| Env | `ANALYSIS_MODE=http`, `ANALYSIS_SERVICE_URL`, `ANALYSIS_TOKEN_SECRET`, `SECRET_KEY`, Stripe vars | `ANTHROPIC_API_KEY`, `ANALYSIS_TOKEN_SECRET` (same value) |

Keep the analysis service on private networking — only the core app should reach
it. The Stripe webhook still points at the **core** service.

---

## Project structure

```
fantasy-draft-consultant/
├── run.py                          # Core entry point (create_app)
├── docker-compose.yml              # core + analysis + a Postgres each
├── Dockerfile                      # core image
├── .env.example
│
├── analysis_core/                  # Analysis domain logic — imported by both services
│   ├── models.py                   # PlayerVerdict value object
│   ├── services.py                 # validate_player_name, parse_verdict
│   ├── anthropic_client.py         # the Claude agent (web_search, retry, prompt hardening)
│   └── contract.py                 # wire schemas + HMAC actor-token helpers
│
├── app/                            # Core service
│   ├── __init__.py                 # app factory; selects the analysis client
│   ├── extensions.py               # db, bcrypt, login_manager, limiter
│   ├── identity/                   #   domain/ infrastructure/ presentation/  (/login, /signup)
│   ├── subscription/               #   domain/ infrastructure/ presentation/  (/subscribe, /webhook)
│   └── analysis/
│       ├── client/
│       │   ├── base.py             # AnalysisClient protocol, error taxonomy, Actor, factory
│       │   ├── inprocess.py        # calls analysis_core directly (default)
│       │   └── http.py             # calls the analysis service
│       └── presentation/routes.py  # /, /analyze
│
├── analysis_service/               # Standalone analysis service (http mode)
│   ├── wsgi.py  Dockerfile  Procfile  requirements.txt
│   ├── service/
│   │   ├── __init__.py             # app factory
│   │   ├── api.py                  # POST/GET /v1/analyses, GET /healthz
│   │   ├── auth.py                 # verify the HMAC actor token (no identity import)
│   │   ├── schemas.py              # Flask <-> analysis_core.contract glue
│   │   └── persistence.py          # its own db + AnalysisRecord audit log
│   └── tests/
│
├── templates/                      # index.html, login.html, signup.html
├── tests/                          # core suite (+ tests/http_shim.py)
├── draft_consultant.py             # standalone CLI version
├── requirements.txt  requirements-dev.txt
├── pytest.ini                      # pythonpath + testpaths (both trees)
└── Procfile                        # core Render process config
```

---

## Security

- Player name input validated against a strict allowlist (letters, spaces, apostrophes, hyphens, periods only)
- Player names wrapped in XML delimiters in the prompt to prevent injection
- Passwords hashed with bcrypt
- Stripe webhook signature verified on every request
- API errors surface cleanly in the UI — no raw tracebacks exposed
- In `http` mode, core→analysis calls carry a short-lived HMAC-signed token; the analysis service verifies it, shares no database or models with the core app, and is never exposed publicly

---

## When the analysis service is unavailable (`http` mode)

If the analysis service is unreachable (connection refused, timeout) or returns
a 5xx, the core app's `/analyze`:

- returns **HTTP 503** with `{"error": "The analysis service is busy right now. Please try again in a minute."}` — no traceback
- **does not** count the request against the user's monthly quota
- logs the failure

An Anthropic `429` propagated through the service is treated the same (503,
retryable). An Anthropic backend error (`5xx` / connection) surfaces as `502`.
None of these consume a query. The browser UI shows the message and lets the
user retry.

Check the analysis service directly at `GET /healthz` (`anthropic_configured`
tells you whether its `ANTHROPIC_API_KEY` is set).
