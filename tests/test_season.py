from datetime import date

import pytest

from analysis_core.season import current_season, season_expiry
from tests.conftest import login


# ---------------------------------------------------------------------------
# Unit: the season calendar (rolls over May 1, just after the NFL Draft)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("today,season", [
    (date(2026, 1, 15), 2025),   # playoffs / off-season: still last season
    (date(2026, 4, 30), 2025),   # NFL Draft week
    (date(2026, 5, 1), 2026),    # rollover
    (date(2026, 8, 25), 2026),   # fantasy draft season
    (date(2026, 10, 12), 2026),  # games underway: not next year yet
    (date(2026, 12, 31), 2026),
])
def test_current_season(today, season):
    assert current_season(today) == season


@pytest.mark.parametrize("today,expiry", [
    (date(2026, 2, 1), date(2026, 5, 1)),    # bought for 2025: ends at the 2026 rollover
    (date(2026, 5, 1), date(2027, 5, 1)),
    (date(2026, 8, 25), date(2027, 5, 1)),   # bought for 2026: covers drafts and games
    (date(2026, 10, 12), date(2027, 5, 1)),
])
def test_season_expiry_is_the_next_rollover(today, expiry):
    assert season_expiry(today) == expiry


def test_a_pass_never_spans_two_seasons():
    for month in range(1, 13):
        today = date(2026, month, 15)
        assert current_season(season_expiry(today)) == current_season(today) + 1


# ---------------------------------------------------------------------------
# Pages: the title and the pass wording name the same season
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/login", "/signup"])
def test_auth_pages_show_current_season(client, path):
    html = client.get(path).get_data(as_text=True)
    assert f"{current_season()} PPR" in html


def test_index_title_and_pass_name_the_same_season(client, free_user):
    login(client, "free@test.com")
    html = client.get("/").get_data(as_text=True)
    season = current_season()
    assert f"<title>{season} PPR Draft Consultant</title>" in html
    assert f"Get {season} Season" in html
    for other in (season - 1, season + 1):
        assert f"{other} PPR" not in html and f"{other} Season" not in html
