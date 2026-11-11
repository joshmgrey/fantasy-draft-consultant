"""Turn CI test results into README badge data and a job summary.

    python -m ml.ci_badges junit.xml coverage.xml --out badges-out   # badge JSON
    python -m ml.ci_badges junit.xml coverage.xml --summary          # markdown

Reads pytest's JUnit XML and coverage.py's Cobertura XML. Badge files use
the shields.io endpoint schema (https://shields.io/badges/endpoint-badge),
so the README can render live numbers from the ``badges`` branch.

Stdlib only: CI runs it before (or without) installing the app.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Results:
    tests: int
    failures: int
    errors: int
    skipped: int
    coverage: float  # percent

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped

    @property
    def ok(self) -> bool:
        return self.failures == 0 and self.errors == 0


def read(junit: Path, coverage: Path) -> Results:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    rate = float(ET.parse(coverage).getroot().get("line-rate", 0))
    return Results(coverage=round(100 * rate, 1), **total)


def coverage_color(pct: float) -> str:
    if pct >= 95:
        return "brightgreen"
    if pct >= 85:
        return "green"
    if pct >= 75:
        return "yellow"
    return "red"


def badges(r: Results) -> dict[str, dict]:
    tests = (f"{r.passed} passed" if r.ok
             else f"{r.failures + r.errors} failed, {r.passed} passed")
    if r.skipped:
        tests += f", {r.skipped} skipped"
    return {
        "tests.json": {"schemaVersion": 1, "label": "tests", "message": tests,
                       "color": "brightgreen" if r.ok else "red"},
        "coverage.json": {"schemaVersion": 1, "label": "coverage",
                          "message": f"{r.coverage:g}%", "color": coverage_color(r.coverage)},
    }


def summary(r: Results) -> str:
    status = "✅ all passing" if r.ok else "❌ failures"
    return "\n".join([
        "### Test results",
        "",
        "| Result | Passed | Failed | Errors | Skipped | Line coverage |",
        "|---|---|---|---|---|---|",
        f"| {status} | {r.passed} | {r.failures} | {r.errors} | {r.skipped} | {r.coverage:g}% |",
        "",
    ])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("--out", type=Path, help="directory to write badge JSON into")
    parser.add_argument("--summary", action="store_true", help="print a markdown summary")
    args = parser.parse_args(argv)
    r = read(args.junit, args.coverage)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, body in badges(r).items():
            (args.out / name).write_text(json.dumps(body) + "\n", encoding="utf-8")
    if args.summary:
        # GitHub's step summary is UTF-8; a Windows console may not be.
        sys.stdout.reconfigure(encoding="utf-8")
        print(summary(r))


if __name__ == "__main__":
    main()
