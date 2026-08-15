"""Structural validation of Khmer Unicode text.

Scope, deliberately: this checks Unicode *structural* well-formedness of
Khmer sequences (does a COENG have a consonant after it, does a cluster
start with a mark that needs a base, etc). It does not, and cannot without
a dictionary, check spelling or grammar - "is this a real Khmer word" is
out of scope for this module. See README.md section 17 (evaluation
system) for where that belongs once a corpus/dictionary exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import codepoints as cp_db
from .character_types import (
    ZWJ,
    ZWNJ,
    CharacterType,
    classify_codepoint,
)
from .grapheme import segment_graphemes

# Unicode chart footnotes describe these independent-vowel code points as
# rare/legacy in modern orthography (see codepoints.py notes on QAQ/QAA).
_LEGACY_INDEPENDENT_VOWELS = {0x17A3, 0x17A4}

_STARTS_CLUSTER_INVALID = frozenset(
    {
        CharacterType.DEPENDENT_VOWEL,
        CharacterType.INHERENT_VOWEL,
        CharacterType.DIACRITIC,
        CharacterType.REGISTER_SHIFTER,
        CharacterType.COENG,
    }
)


@dataclass(frozen=True)
class ValidationIssue:
    index: int
    severity: str  # "error" or "warning"
    code: str
    message: str
    char: str = ""


def _check_coeng_sequences(text: str) -> list[ValidationIssue]:
    issues = []
    n = len(text)
    for i, ch in enumerate(text):
        if ord(ch) != cp_db.COENG_CODEPOINT:
            continue
        j = i + 1
        while j < n and ord(text[j]) in (ZWNJ, ZWJ):
            j += 1
        if j >= n or classify_codepoint(ord(text[j])) != CharacterType.CONSONANT:
            issues.append(
                ValidationIssue(
                    index=i,
                    severity="error",
                    code="coeng-no-consonant",
                    message="COENG (subscript sign) is not followed by a consonant",
                    char=ch,
                )
            )
    return issues


def _check_unassigned(text: str) -> list[ValidationIssue]:
    issues = []
    for i, ch in enumerate(text):
        if classify_codepoint(ord(ch)) is CharacterType.UNASSIGNED:
            issues.append(
                ValidationIssue(
                    index=i,
                    severity="error",
                    code="unassigned-codepoint",
                    message=f"U+{ord(ch):04X} is an unassigned code point in the Khmer block",
                    char=ch,
                )
            )
    return issues


def _check_clusters(text: str) -> list[ValidationIssue]:
    issues = []
    for grapheme in segment_graphemes(text):
        char, base_type = grapheme.char_types[0]

        if base_type in _STARTS_CLUSTER_INVALID:
            issues.append(
                ValidationIssue(
                    index=grapheme.start,
                    severity="error",
                    code="orphan-combining-mark",
                    message=f"cluster {grapheme.text!r} starts with {base_type.value}, which needs a preceding base character",
                    char=char,
                )
            )

        vowel_count = sum(1 for _, t in grapheme.char_types if t is CharacterType.DEPENDENT_VOWEL)
        if vowel_count > 1:
            issues.append(
                ValidationIssue(
                    index=grapheme.start,
                    severity="warning",
                    code="multiple-vowel-signs",
                    message=f"cluster {grapheme.text!r} has {vowel_count} dependent vowel signs (usually at most one)",
                    char=grapheme.text,
                )
            )

        shifter_count = sum(1 for _, t in grapheme.char_types if t is CharacterType.REGISTER_SHIFTER)
        if shifter_count > 1:
            issues.append(
                ValidationIssue(
                    index=grapheme.start,
                    severity="warning",
                    code="multiple-register-shifters",
                    message=f"cluster {grapheme.text!r} has {shifter_count} register shifters (usually at most one)",
                    char=grapheme.text,
                )
            )

        base_consonant = cp_db.CONSONANTS_BY_CHAR.get(char)
        if base_consonant is not None and base_consonant.obsolete:
            issues.append(
                ValidationIssue(
                    index=grapheme.start,
                    severity="warning",
                    code="obsolete-consonant",
                    message=f"{char!r} ({base_consonant.name}) is an obsolete consonant, rarely used outside Sanskrit transliteration",
                    char=char,
                )
            )
        if ord(char) in _LEGACY_INDEPENDENT_VOWELS:
            issues.append(
                ValidationIssue(
                    index=grapheme.start,
                    severity="warning",
                    code="legacy-independent-vowel",
                    message=f"{char!r} is a legacy/rare independent vowel code point in modern Khmer text",
                    char=char,
                )
            )

    return issues


def validate(text: str) -> list[ValidationIssue]:
    """Return all structural issues found in `text`, ordered by position."""
    issues = _check_coeng_sequences(text) + _check_unassigned(text) + _check_clusters(text)
    return sorted(issues, key=lambda issue: issue.index)


def is_valid(text: str) -> bool:
    return not any(issue.severity == "error" for issue in validate(text))
