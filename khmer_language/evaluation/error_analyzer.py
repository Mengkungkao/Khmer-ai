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
  - Spelling: likely misspellings against the Khmer dictionary - but only
    once that dictionary has been built, so this one check reports either
    a real score or UNAVAILABLE depending on the machine it runs on

Not yet measurable, reported as UNAVAILABLE with the reason:
  - Grammar       - needs a parser or a trained grammar model
  - Meaning       - needs reference answers or a judge model
  - Naturalness   - needs a reference LM or human ratings
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..corpus.language_id import khmer_script_ratio
from ..unicode.grapheme import grapheme_strings
from ..unicode.validator import ValidationIssue, validate

PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"

NO_LEXICON = "needs the Khmer dictionary - build it with scripts/build_lexicon.py"


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
    """Share of script-bearing characters that are Khmer.

    Delegates to `corpus.language_id`, which owns the canonical
    definition. This module previously had its own copy and the two
    disagreed - notably on zero-width word separators and on Khmer
    combining marks - so there is deliberately only one implementation
    now, with a test asserting the two entry points agree.
    """
    ratio, _, _ = khmer_script_ratio(text)
    return ratio


def _spelling_check(text: str, min_score: float = 0.9) -> Check:
    """Spelling, if a lexicon has been built.

    Always returns a Check. When no lexicon exists - or the output has
    no Khmer words in it - that Check is UNAVAILABLE, so the row stays
    in the report and says why. Dropping it instead would quietly shrink
    the report from seven checks to six, and a reader comparing two runs
    would see spelling simply absent rather than unmeasured.

    Only *likely misspellings* count against the score - words a single
    edit away from a real dictionary word. Words simply absent from the
    dictionary do not, because 24% of Khmer Wikipedia words are outside
    this lexicon and virtually all of them are correctly spelled proper
    nouns and technical terms. Treating those as errors would make the
    check worse than useless.
    """
    try:
        from ..lexicon import KhmerLexicon
        from ..lexicon.spelling import SpellChecker
    except ImportError:  # pragma: no cover
        return Check("Spelling", UNAVAILABLE, NO_LEXICON)

    global _SPELL_CHECKER
    if _SPELL_CHECKER is None:
        try:
            _SPELL_CHECKER = SpellChecker(KhmerLexicon.load())
        except FileNotFoundError:
            return Check("Spelling", UNAVAILABLE, NO_LEXICON)

    report = _SPELL_CHECKER.check(text)
    if not report.checks:
        return Check("Spelling", UNAVAILABLE, "no Khmer words in the output to check")

    detail = f"{len(report.misspelled)} likely misspelling(s)"
    if report.unknown:
        detail += f", {len(report.unknown)} not in dictionary (not counted as errors)"
    if report.misspelled:
        first = report.misspelled[0]
        if first.suggestions:
            detail += f"; e.g. {first.word} -> {first.suggestions[0]}"

    return Check(
        name="Spelling",
        status=PASS if report.score >= min_score else FAIL,
        detail=detail,
        score=report.score,
    )


# Built lazily and reused: the deletion index costs a second to construct
# and the analyzer is called per model output.
_SPELL_CHECKER = None


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

    spelling_check = _spelling_check(text)

    checks = (
        unicode_check,
        script_check,
        repetition_check,
        spelling_check,
        Check("Grammar", UNAVAILABLE, "needs a parser or trained grammar model"),
        Check("Meaning", UNAVAILABLE, "needs reference answers or a judge model"),
        Check("Naturalness", UNAVAILABLE, "needs a reference LM or human ratings"),
    )

    return ErrorReport(text=text, checks=checks, issues=issues)


def format_report(report: ErrorReport) -> str:
    lines = [f"Output: {report.text}", ""]
    width = max(len(c.name) for c in report.checks) + 2
    for check in report.checks:
        score = f"  ({check.score:.2f})" if check.score is not None else ""
        lines.append(f"  {check.name + ':':<{width}} {check.status:<12}{check.detail}{score}")
    lines.append("")
    lines.append(f"  {report.coverage}")
    return "\n".join(lines)
