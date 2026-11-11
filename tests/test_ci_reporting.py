"""CI badge data (ml/ci_badges.py) and the generated README metrics block
(ml/readme_metrics.py)."""

import json

import pytest

from ml import ci_badges, readme_metrics

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>"""
COVERAGE = '<?xml version="1.0"?><coverage line-rate="{rate}"></coverage>'


def write(tmp_path, tests=290, failures=0, errors=0, skipped=0, rate="0.9812"):
    j, c = tmp_path / "junit.xml", tmp_path / "coverage.xml"
    j.write_text(JUNIT.format(tests=tests, failures=failures, errors=errors, skipped=skipped))
    c.write_text(COVERAGE.format(rate=rate))
    return j, c


# --------------------------------------------------------------------------- #
# ci_badges
# --------------------------------------------------------------------------- #
def test_passing_run_badges(tmp_path):
    j, c = write(tmp_path)
    ci_badges.main([str(j), str(c), "--out", str(tmp_path / "out")])
    tests = json.loads((tmp_path / "out" / "tests.json").read_text())
    cov = json.loads((tmp_path / "out" / "coverage.json").read_text())
    assert tests == {"schemaVersion": 1, "label": "tests", "message": "290 passed",
                     "color": "brightgreen"}
    assert cov == {"schemaVersion": 1, "label": "coverage", "message": "98.1%",
                   "color": "brightgreen"}


def test_failures_and_skips_show_and_turn_red(tmp_path):
    r = ci_badges.read(*write(tmp_path, tests=10, failures=1, errors=1, skipped=2, rate="0.8"))
    assert (r.passed, r.ok) == (6, False)
    b = ci_badges.badges(r)
    assert b["tests.json"]["message"] == "2 failed, 6 passed, 2 skipped"
    assert b["tests.json"]["color"] == "red"
    assert b["coverage.json"]["color"] == "yellow"


@pytest.mark.parametrize("pct, color", [(99, "brightgreen"), (95, "brightgreen"),
                                        (90, "green"), (80, "yellow"), (50, "red")])
def test_coverage_colors(pct, color):
    assert ci_badges.coverage_color(pct) == color


def test_single_testsuite_root_is_read(tmp_path):
    j = tmp_path / "junit.xml"
    j.write_text('<testsuite tests="3" failures="0" errors="0" skipped="1"/>')
    c = tmp_path / "coverage.xml"
    c.write_text(COVERAGE.format(rate="1"))
    r = ci_badges.read(j, c)
    assert (r.tests, r.skipped, r.coverage) == (3, 1, 100.0)


def test_summary_markdown(tmp_path, capsys):
    j, c = write(tmp_path, skipped=1)
    ci_badges.main([str(j), str(c), "--summary"])
    out = capsys.readouterr().out
    assert "| ✅ all passing | 289 | 0 | 0 | 1 | 98.1% |" in out


# --------------------------------------------------------------------------- #
# readme_metrics
# --------------------------------------------------------------------------- #
def test_shield_escapes_dashes_underscores_and_spaces():
    url = readme_metrics.shield("RAG hit@1", "86% vs 38%")
    assert url == "https://img.shields.io/badge/RAG_hit%401-86%25_vs_38%25-blue"
    assert readme_metrics.shield("a-b_c", "x").startswith("https://img.shields.io/badge/a--b__c-x")


def test_render_uses_the_report_numbers():
    model, rag = readme_metrics.load_reports()
    block = readme_metrics.render(model, rag)
    m = model["test"]["model"]["overall"]["mae"]
    b = model["test"]["baseline"]["overall"]["mae"]
    assert f"**{m}**" in block and f"| {b} (same as last season)" in block
    assert f"{readme_metrics.pct_lower(m, b)}% lower" in block
    assert block.startswith(readme_metrics.START) and block.endswith(readme_metrics.END)


def test_readme_metrics_are_current():
    """Fails when a retrain or re-evaluation changed ml/reports but the README
    block wasn't regenerated (python -m ml.readme_metrics)."""
    assert readme_metrics.main(["--check"]) == 0


def test_check_detects_drift_and_rewrite_fixes_it(tmp_path, capsys):
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{readme_metrics.START}\nold\n{readme_metrics.END}\n", encoding="utf-8")
    assert readme_metrics.main(["--check"], readme=readme) == 1
    assert "stale" in capsys.readouterr().err
    assert readme_metrics.main([], readme=readme) == 0
    assert "updated" in capsys.readouterr().out
    assert readme_metrics.main(["--check"], readme=readme) == 0
    assert readme_metrics.main([], readme=readme) == 0     # already current: no rewrite


def test_missing_markers_is_an_error(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no block\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metrics:start"):
        readme_metrics.main([], readme=readme)
