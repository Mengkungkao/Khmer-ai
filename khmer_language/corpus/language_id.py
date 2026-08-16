"""Khmer language identification (README section 6, "Language Identification").

Khmer is one of the cases where script-based identification is genuinely
reliable rather than a crude approximation: the Khmer Unicode block
(U+1780-U+17FF) is used by Khmer essentially exclusively. Contrast with
distinguishing Spanish from Portuguese, or Hindi from Marathi, where a
shared script forces a statistical model.

So this measures the *proportion* of Khmer script, which answers the two
questions the corpus pipeline actually needs:

  - "is this Khmer at all?"                  -> khmer_ratio high
  - "is this mixed-language garbage?"        -> khmer_ratio middling

What it deliberately does NOT claim to do is detect *which* language the
non-Khmer portion is, or catch Khmer-script text that is actually Pali or
Sanskrit transliteration. Those need a real language model.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..unicode.character_types import CharacterType, classify_codepoint

# Every character the Khmer block assigns counts as Khmer script - including
# combining marks (COENG, vowel signs, diacritics) and Khmer-specific
# punctuation such as ។, which is itself strong evidence of Khmer.
KHMER_SCRIPT_TYPES = frozenset(
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

# Skipped entirely rather than counted for either side. Whitespace and
# zero-width characters carry no script (ZWSP is the standard Khmer word
# boundary, so counting it against Khmer would be actively wrong), and
# ASCII digits/punctuation appear in perfectly good Khmer text.
_NEUTRAL_TYPES = frozenset({CharacterType.WHITESPACE, CharacterType.ZERO_WIDTH})


@dataclass(frozen=True)
class LanguageScore:
    khmer_ratio: float
    khmer_chars: int
    other_chars: int

    @property
    def is_khmer(self) -> bool:
        return self.khmer_ratio >= 0.5


def khmer_script_ratio(text: str) -> tuple[float, int, int]:
    """Canonical Khmer-script measurement: (ratio, khmer_chars, other_chars).

    This lives in one place because two modules previously computed it
    separately and disagreed. Classification order matters and is the
    subtlety worth stating: a character is checked against the Khmer
    script FIRST, before any neutrality test.

    Testing neutrality first is what caused the original bug. Khmer
    combining marks (COENG, vowel signs) are Unicode categories Mn/Mc, so
    `str.isalpha()` is False for them, and a `not ch.isalpha()` test meant
    to skip ASCII punctuation silently skipped them too - undercounting
    "កម្ពុជា" as 4 Khmer characters instead of 7, and under-rating
    Khmer-heavy mixed-language documents in the corpus filter.
    """
    khmer = other = 0
    for ch in text:
        char_type = classify_codepoint(ord(ch))

        if char_type in KHMER_SCRIPT_TYPES:
            khmer += 1
            continue
        if char_type in _NEUTRAL_TYPES or ch.isspace():
            continue
        # Non-Khmer: script-bearing letters count against, while ASCII
        # digits and punctuation are neutral.
        if ch.isalpha():
            other += 1

    total = khmer + other
    return (khmer / total if total else 0.0), khmer, other


def identify(text: str) -> LanguageScore:
    """Measure how much of `text` is Khmer script."""
    ratio, khmer, other = khmer_script_ratio(text)
    return LanguageScore(khmer_ratio=ratio, khmer_chars=khmer, other_chars=other)


def is_khmer(text: str, threshold: float = 0.5) -> bool:
    return identify(text).khmer_ratio >= threshold
