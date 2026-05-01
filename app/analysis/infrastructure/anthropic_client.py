import os
import anthropic

_api_key = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=_api_key) if _api_key else None

SYSTEM_PROMPT = """You are an elite fantasy football analyst specializing in 2026 PPR drafts.
Your edge is identifying VALUE OVER ADP — players whose projected output exceeds their draft cost.

When evaluating a player, you MUST:
1. Search for their 2026 PPR fantasy projections (points, position rank, ADP)
2. Search for recent news (trades, depth chart changes, target share, snap counts)
3. Search for injury history and current injury risk

Then deliver a concise verdict:
- RISK SCORE: 1–10 (1 = zero risk, 10 = extreme risk/injury-prone/situation unclear)
- RECOMMENDATION: "Draft" or "Pass"
- JUSTIFICATION: One sentence focusing on VALUE vs ADP (not just talent)

Format your final answer EXACTLY as:
RISK: <number>/10
VERDICT: <Draft|Pass>
REASON: <one sentence>

Be direct. No preamble. No lengthy analysis. Just the three lines above."""

TOOLS = [{"type": "web_search_20260209", "name": "web_search"}]


def analyze_player(player_name: str) -> str:
    """Runs the AI agent and returns the raw text response."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze the player named <player>{player_name}</player> for a 2026 PPR fantasy football draft. "
                "Search for their 2026 projections, ADP, recent news, and injury risk, "
                "then give me your Risk Score, Verdict, and Reason."
            ),
        }
    ]

    iterations = 0
    while iterations < 10:
        iterations += 1

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason == "pause_turn":
            continue
        break

    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            return block.text

    return ""
