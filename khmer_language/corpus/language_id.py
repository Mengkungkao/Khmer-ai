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
        CharacterType.CURRENCY,
        CharacterType.OTHER_KHMER,
    }
)

# Punctuation and digits are script-neutral: ASCII digits and Latin
# punctuation appear in perfectly good Khmer text, so counting them as
# "not Khmer" would unfairly penalise it.
_NEUTRAL_TYPES = frozenset({CharacterType.WHITESPACE, CharacterType.ZERO_WIDTH})


@dataclass(frozen=True)
class LanguageScore:
    khmer_ratio: float
    khmer_chars: int
    other_chars: int

    @property
    def is_khmer(self) -> bool:
        return self.khmer_ratio >= 0.5


def _is_neutral(ch: str) -> bool:
    if classify_codepoint(ord(ch)) in _NEUTRAL_TYPES:
        return True
    # Script-neutral characters: ASCII digits and common punctuation.
    return ch.isdigit() or (not ch.isalpha() and not ch.isspace())


def identify(text: str) -> LanguageScore:
    """Measure how much of `text` is Khmer script."""
    khmer = other = 0
    for ch in text:
        if _is_neutral(ch):
            continue
        if classify_codepoint(ord(ch)) in _KHMER_TYPES:
            khmer += 1
        else:
            other += 1

    total = khmer + other
    ratio = khmer / total if total else 0.0
    return LanguageScore(khmer_ratio=ratio, khmer_chars=khmer, other_chars=other)


def is_khmer(text: str, threshold: float = 0.5) -> bool:
    return identify(text).khmer_ratio >= threshold
