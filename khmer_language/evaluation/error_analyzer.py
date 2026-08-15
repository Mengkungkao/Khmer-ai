"""Khmer error analyzer (README section 18).

Section 18 asks for a per-output report like:

    Unicode:      PASS
    Spelling:     PASS
    Grammar:      PASS
    Meaning:      PASS
    Naturalness:  0.91

Only some of those are honestly computable today, and this module says so
explicitly rather than inventing numbers. A fabricated "Naturalness: 0.91"
would be worse than useless - it would make the model look evaluated when
it is not, and that is exactly the failure mode section 18 exists to
prevent.

Currently measurable (real checks, from Project 1's engine):
  - Unicode structure: orphan combining marks, malformed COENG sequences,
    unassigned code points
  - Script composition: how much of the output is even Khmer
  - Repetition: degenerate loops, the classic undertrained-LM failure

Not yet measurable, reported as UNAVAILABLE with the reason:
  - Spelling      - needs the Khmer dictionary from README section 8
  - Grammar       - needs a parser or a trained grammar model
  - Meaning       - needs reference answers or a judge model
  - Naturalness   - needs a reference LM or human ratings
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..unicode.character_types import CharacterType, classify_codepoint
from ..unicode.grapheme import grapheme_strings
from ..unicode.validator import ValidationIssue, validate

PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"

_KHMER_TYPES = frozenset(
    {
        CharacterType.CONSONANT,
        CharacterType.SUBSCRIPT_CONSONANT,
        CharacterType.INDEPENDENT_VOWEL,
        CharacterType.DEPENDENT_VOWEL,
        CharacterType.INHERENT_VOWEL,
        CharacterType.REGISTER_SHIFTER,
        CharacterType.COENG,
        CharacterType.DIACRITIC,
        CharacterType.DIGIT,
        CharacterType.LEK_ATTAK,
        CharacterType.PUNCTUATION,
        CharacterType.CURRENCY,
        CharacterType.OTHER_KHMER,
    }
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS / FAIL / UNAVAILABLE
    detail: str = ""
    score: float | None = None


@dataclass(frozen=True)
class ErrorReport:
    text: str
    checks: tuple[Check, ...]
    issues: tuple[ValidationIssue, ...] = field(default=())

    @property
    def passed(self) -> bool:
        """True when no *implemented* check failed. Unavailable checks are
        not counted as passes - see `coverage`."""
        return all(c.status != FAIL for c in self.checks)

    @property
    def coverage(self) -> str:
        implemented = sum(1 for c in self.checks if c.status != UNAVAILABLE)
        return f"{implemented}/{len(self.checks)} checks implemented"


def _khmer_ratio(text: str) -> float:
    considered = [c for c in text if not c.isspace()]
    if not considered:
        return 0.0
    khmer = sum(1 for c in considered if classify_codepoint(ord(c)) in _KHMER_TYPES)
    return khmer / len(considered)


def _max_repetition(text: str) -> int:
    """Longest run of a single repeated grapheme cluster."""
    units = grapheme_strings(text)
    if not units:
        return 0
    longest = current = 1
    for a, b in zip(units, units[1:]):
        current = current + 1 if a == b else 1
        longest = max(longest, current)
    return longest


def analyze_output(
    text: str,
    *,
    min_khmer_ratio: float = 0.5,
    max_repetition: int = 5,
) -> ErrorReport:
    """Analyze one model output. See the module docstring for scope."""
    issues = tuple(validate(text))
    errors = [i for i in issues if i.severity == "error"]

    unicode_check = Check(
        name="Unicode",
        status=PASS if not errors else FAIL,
        detail="no structural errors" if not errors else "; ".join(i.code for i in errors),
    )

    ratio = _khmer_ratio(text)
    script_check = Check(
        name="Khmer script",
        status=PASS if ratio >= min_khmer_ratio else FAIL,
        detail=f"{ratio:.0%} of non-space characters are Khmer",
        score=ratio,
    )

    repetition = _max_repetition(text)
    repetition_check = Check(
        name="Repetition",
        status=PASS if repetition <= max_repetition else FAIL,
        detail=f"longest repeated grapheme run: {repetition}",
        score=float(repetition),
    )

    unavailable = (
        Check("Spelling", UNAVAILABLE, "needs the Khmer dictionary (README section 8)"),
        Check("Grammar", UNAVAILABLE, "needs a parser or trained grammar model"),
        Check("Meaning", UNAVAILABLE, "needs reference answers or a judge model"),
        Check("Naturalness", UNAVAILABLE, "needs a reference LM or human ratings"),
    )

    return ErrorReport(
        text=text,
        checks=(unicode_check, script_check, repetition_check) + unavailable,
        issues=issues,
    )


def format_report(report: ErrorReport) -> str:
    lines = [f"Output: {report.text}", ""]
    width = max(len(c.name) for c in report.checks) + 2
    for check in report.checks:
        score = f"  ({check.score:.2f})" if check.score is not None else ""
        lines.append(f"  {check.name + ':':<{width}} {check.status:<12}{check.detail}{score}")
    lines.append("")
    lines.append(f"  {report.coverage}")
    return "\n".join(lines)
