"""The prompt tells the model how to judge a team defense (D/ST)."""

from analysis_core import anthropic_client as ac


def test_system_prompt_has_dst_branch():
    lowered = ac.SYSTEM_PROMPT.lower()
    assert "team defense (d/st)" in lowered
    # D/ST is judged on what drives its scoring, not one player's injury history.
    for marker in ("sacks", "turnover", "schedule", "streamed"):
        assert marker in lowered
    assert "the output format below is the same for a d/st" in lowered


def test_user_instruction_points_at_dst_branch():
    content = ac._build_user_content("Steelers D/ST", None, None)
    assert "<player>Steelers D/ST</player>" in content
    assert "team defense (D/ST)" in content


def test_dst_text_does_not_trip_the_leak_guard():
    # The leak guard looks for sentinel phrases; a normal D/ST answer that echoes
    # words like "streamable" or "schedule" must not be withheld.
    answer = "RISK: 4/10\nVERDICT: Pass\nREASON: Strong pass rush but streamable at its ADP."
    assert not ac._looks_like_prompt_leak(answer)
