"""
Fantasy Football Draft Consultant — 2026 PPR
Usage: python draft_consultant.py "Player Name"
       python draft_consultant.py  (interactive mode)
"""

import os
import re
import sys
import anthropic

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    print("Error: ANTHROPIC_API_KEY environment variable is not set.")
    print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

client = anthropic.Anthropic(api_key=_api_key)

# Only letters, spaces, apostrophes, hyphens, and periods (e.g. "D.K. Metcalf", "Ja'Marr Chase")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z '\-\.]{0,48}$")
_MAX_NAME_LEN = 50


def _validate_player_name(name: str) -> str:
    """Return the cleaned name or raise ValueError."""
    name = name.strip()
    if not name:
        raise ValueError("Player name cannot be empty.")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"Player name too long (max {_MAX_NAME_LEN} characters).")
    if not _NAME_RE.match(name):
        raise ValueError(
            "Player name contains invalid characters. "
            "Use letters, spaces, apostrophes, hyphens, or periods only."
        )
    return name

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

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
]


def analyze_player(player_name: str) -> dict:
    """Run the draft consultant agent for a single player."""
    player_name = _validate_player_name(player_name)

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

    print(f"\n🔍  Researching {player_name}", end="", flush=True)

    # web_search_20260209 is a server-side tool — Anthropic runs it automatically.
    # The loop handles pause_turn (server hit its iteration limit) by continuing;
    # end_turn means Claude finished and wrote its verdict.
    iterations = 0
    while iterations < 10:
        iterations += 1
        print(".", end="", flush=True)

        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIStatusError as e:
            print(f"\n  API error ({e.status_code}): {e.message}")
            sys.exit(1)
        except anthropic.APIConnectionError:
            print("\n  Connection error: could not reach Anthropic API.")
            sys.exit(1)

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "pause_turn":
            # Server-side loop hit its cap — continue to let it finish.
            continue

        # Unexpected stop reason — bail out.
        break

    print(" done.\n")

    # Extract the final text response
    final_text = ""
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            final_text = block.text
            break

    return _parse_verdict(player_name, final_text)


def _parse_verdict(player_name: str, text: str) -> dict:
    """Parse the structured verdict from the model's response."""
    result = {
        "player": player_name,
        "risk_score": None,
        "verdict": None,
        "reason": None,
        "raw": text,
    }

    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("RISK:"):
            val = line.split(":", 1)[1].strip().split("/")[0].strip()
            try:
                result["risk_score"] = int(val)
            except ValueError:
                pass
        elif line.upper().startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip()
            result["verdict"] = "Draft" if "draft" in val.lower() else "Pass"
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result


def print_result(result: dict) -> None:
    """Pretty-print the draft verdict."""
    player = result["player"]
    risk = result["risk_score"]
    verdict = result["verdict"]
    reason = result["reason"]

    if not all([risk, verdict, reason]):
        print("─" * 55)
        print(f"  {player}")
        print("─" * 55)
        print(result["raw"])
        return

    # Color-coded verdict
    verdict_str = f"✅  DRAFT" if verdict == "Draft" else f"❌  PASS"
    risk_bar = "█" * risk + "░" * (10 - risk)

    print("─" * 55)
    print(f"  {player.upper()}")
    print("─" * 55)
    print(f"  {verdict_str}")
    print(f"  RISK  [{risk_bar}]  {risk}/10")
    print(f"  WHY   {reason}")
    print("─" * 55)


def main():
    if len(sys.argv) > 1:
        # Single player from CLI argument
        try:
            player_name = _validate_player_name(" ".join(sys.argv[1:]))
        except ValueError as e:
            print(f"  Error: {e}")
            sys.exit(1)
        result = analyze_player(player_name)
        print_result(result)
    else:
        # Interactive mode
        print("━" * 55)
        print("   🏈  2026 PPR DRAFT CONSULTANT")
        print("   Powered by Claude Opus 4.7 + Live Web Search")
        print("━" * 55)
        print("   Type a player name and press Enter.")
        print("   Type 'quit' to exit.\n")

        while True:
            try:
                player_name = input("  Player: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Bye!")
                break

            if not player_name:
                continue
            if player_name.lower() in ("quit", "exit", "q"):
                print("  Bye!")
                break

            try:
                player_name = _validate_player_name(player_name)
            except ValueError as e:
                print(f"  Error: {e}\n")
                continue

            result = analyze_player(player_name)
            print_result(result)
            print()


if __name__ == "__main__":
    main()
