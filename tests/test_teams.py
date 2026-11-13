"""Recognizing a team defense (D/ST) in a typed name."""

import pytest

from analysis_core.teams import FRANCHISE_MOVES, TEAMS, current_abbr, dst_display_name, resolve_dst


@pytest.mark.parametrize("name, abbr", [
    ("Steelers D/ST", "PIT"),
    ("steelers d / st", "PIT"),
    ("Pittsburgh Steelers", "PIT"),
    ("Steelers", "PIT"),
    ("PIT DST", "PIT"),
    ("Steelers Defense", "PIT"),
    ("49ers D/ST", "SF"),
    ("SF DEF", "SF"),
    ("Bills DEF", "BUF"),
    ("Kansas City Chiefs Defense", "KC"),
    ("Kansas City", "KC"),
    ("Rams D/ST", "LA"),
    ("LAR DST", "LA"),
    ("Chargers D/ST", "LAC"),
    ("NY Jets DST", "NYJ"),
    ("Washington Redskins", "WAS"),
    ("Oakland Raiders D/ST", "LV"),
    ("OAK DST", "LV"),
    ("Seahawks D/ST", "SEA"),
])
def test_resolves_team_defenses(name, abbr):
    assert resolve_dst(name) == abbr


@pytest.mark.parametrize("name", [
    "Ja'Marr Chase",
    "Chase Brown",
    "Josh Allen",
    "PIT",              # a bare abbreviation needs a D/ST marker
    "New York D/ST",    # two teams share the city
    "Los Angeles DST",
    "D/ST",
    "",
])
def test_ignores_players_and_ambiguous_names(name):
    assert resolve_dst(name) is None


def test_every_team_resolves_by_nickname_and_display_name():
    for abbr in TEAMS:
        assert resolve_dst(dst_display_name(abbr)) == abbr
        assert resolve_dst(f"{TEAMS[abbr][1]} D/ST") == abbr


def test_moved_franchises_map_to_current_abbreviation():
    for old, new in FRANCHISE_MOVES.items():
        assert current_abbr(old) == new and new in TEAMS
    assert current_abbr("PIT") == "PIT"
