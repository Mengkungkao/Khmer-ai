"""Loading authored benchmark cases from disk (README section 17).

`benchmark.py` deliberately ships no comprehension/knowledge/translation
cases, because auto-generated ones would measure nothing. Real cases have
to be written by a Khmer speaker. This module is the bridge: it loads
authored cases from a JSONL file so contributing them requires no code
changes.

Format - one JSON object per line:

    {"category": "knowledge", "input": "...", "expected": "...", "note": "..."}

`note` is optional. `category` groups the scores in the report, so use it
to separate e.g. `knowledge` from `translation`.

The loader validates rather than trusting the file: a benchmark with
malformed or empty cases silently reports misleading scores, which is
worse than refusing to load.
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark import BenchmarkCase

DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "cases.jsonl"

REQUIRED_FIELDS = ("category", "input", "expected")


class BenchmarkCaseError(ValueError):
    """Raised when a case file is malformed."""


def parse_case(payload: dict, line_number: int) -> BenchmarkCase:
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise BenchmarkCaseError(f"line {line_number}: missing field(s) {', '.join(missing)}")

    for field in REQUIRED_FIELDS:
        if not isinstance(payload[field], str):
            raise BenchmarkCaseError(f"line {line_number}: '{field}' must be a string")

    if not payload["category"].strip():
        raise BenchmarkCaseError(f"line {line_number}: 'category' must not be empty")
    if not payload["expected"].strip():
        raise BenchmarkCaseError(
            f"line {line_number}: 'expected' must not be empty - a case with no "
            "expected answer cannot be scored"
        )

    return BenchmarkCase(
        category=payload["category"].strip(),
        input=payload["input"],
        expected=payload["expected"],
        note=payload.get("note", ""),
    )


def load_cases(path: str | Path | None = None) -> list[BenchmarkCase]:
    """Load authored benchmark cases from a JSONL file.

    Blank lines and lines beginning with `#` are ignored, so the file can
    carry comments explaining what a group of cases is testing.
    """
    path = Path(path) if path is not None else DEFAULT_CASES_PATH
    if not path.exists():
        return []

    cases: list[BenchmarkCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkCaseError(f"line {line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(payload, dict):
                raise BenchmarkCaseError(f"line {line_number}: expected a JSON object")
            cases.append(parse_case(payload, line_number))
    return cases


def case_counts(cases: list[BenchmarkCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
    return counts
