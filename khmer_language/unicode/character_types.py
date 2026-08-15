"""Classification of characters into Khmer linguistic categories.

Two entry points:

- `classify_codepoint(cp)` - context-free classification. Cannot distinguish
  a consonant used as a normal initial from a consonant used as a subscript,
  because that distinction depends on whether a COENG sign precedes it.
- `classify(text, index)` - context-aware classification. This is what
  `analyze()` and the grapheme engine use.
"""

from __future__ import annotations

import unicodedata
from enum import Enum

from . import codepoints as cp_db

KHMER_BLOCK_START = 0x1780
KHMER_BLOCK_END = 0x17FF

ZWSP = 0x200B  # zero width space - commonly used as a Khmer word-boundary hint
ZWNJ = 0x200C  # zero width non-joiner - forces a coeng cluster to render un-joined
ZWJ = 0x200D  # zero width joiner - forces a conjunct/ligature rendering


class CharacterType(str, Enum):
    CONSONANT = "CONSONANT"
    SUBSCRIPT_CONSONANT = "SUBSCRIPT_CONSONANT"
    INDEPENDENT_VOWEL = "INDEPENDENT_VOWEL"
    DEPENDENT_VOWEL = "DEPENDENT_VOWEL"
    INHERENT_VOWEL = "INHERENT_VOWEL"
    REGISTER_SHIFTER = "REGISTER_SHIFTER"
    COENG = "COENG"
    DIACRITIC = "DIACRITIC"
    DIGIT = "DIGIT"
    LEK_ATTAK = "LEK_ATTAK"
    PUNCTUATION = "PUNCTUATION"
    CURRENCY = "CURRENCY"
    OTHER_KHMER = "OTHER_KHMER"
    UNASSIGNED = "UNASSIGNED"
    ZERO_WIDTH = "ZERO_WIDTH"
    WHITESPACE = "WHITESPACE"
    NON_KHMER = "NON_KHMER"


_SIGN_KIND_TO_TYPE = {
    "SIGN": CharacterType.DIACRITIC,
    "REGISTER_SHIFTER": CharacterType.REGISTER_SHIFTER,
    "COENG": CharacterType.COENG,
    "PUNCTUATION": CharacterType.PUNCTUATION,
    "CURRENCY": CharacterType.CURRENCY,
    "OTHER": CharacterType.OTHER_KHMER,
}

# Characters that "attach" to whatever came before them: combining marks
# (consonants after COENG are handled separately, contextually) plus the
# zero-width joiner/non-joiner, which are typographic control characters
# rather than base characters in their own right.
_ATTACHING_TYPES = frozenset(
    {
        CharacterType.DEPENDENT_VOWEL,
        CharacterType.INHERENT_VOWEL,
        CharacterType.REGISTER_SHIFTER,
        CharacterType.DIACRITIC,
        CharacterType.COENG,
        CharacterType.SUBSCRIPT_CONSONANT,
    }
)


def classify_codepoint(cp: int) -> CharacterType:
    """Context-free classification of a single Unicode code point."""
    if cp == ZWSP or cp == ZWNJ or cp == ZWJ:
        return CharacterType.ZERO_WIDTH

    if cp in cp_db.CONSONANTS_BY_CODEPOINT:
        return CharacterType.CONSONANT
    if cp in cp_db.INDEPENDENT_VOWELS_BY_CODEPOINT:
        return CharacterType.INDEPENDENT_VOWEL
    if cp in cp_db.INHERENT_VOWEL_CODEPOINTS:
        return CharacterType.INHERENT_VOWEL
    if cp in cp_db.DEPENDENT_VOWELS_BY_CODEPOINT:
        return CharacterType.DEPENDENT_VOWEL
    if cp in cp_db.SIGNS_BY_CODEPOINT:
        return _SIGN_KIND_TO_TYPE[cp_db.SIGNS_BY_CODEPOINT[cp].kind]
    if cp in cp_db.DIGITS_BY_CODEPOINT:
        return CharacterType.DIGIT
    if cp in cp_db.LEK_ATTAK_BY_CODEPOINT:
        return CharacterType.LEK_ATTAK

    if KHMER_BLOCK_START <= cp <= KHMER_BLOCK_END:
        return CharacterType.UNASSIGNED

    ch = chr(cp)
    if ch.isspace():
        return CharacterType.WHITESPACE

    return CharacterType.NON_KHMER


def is_attaching(char_type: CharacterType) -> bool:
    """Whether a character of this type attaches to the preceding grapheme
    (used by the grapheme clustering algorithm in `grapheme.py`)."""
    return char_type in _ATTACHING_TYPES


def classify(text: str, index: int) -> CharacterType:
    """Context-aware classification of `text[index]`.

    The only context-dependent distinction is CONSONANT vs
    SUBSCRIPT_CONSONANT: the same code point is a normal initial consonant
    unless it is immediately preceded by COENG (U+17D2), optionally with a
    ZWNJ/ZWJ formatting character in between (real-world text sometimes
    inserts ZWNJ between COENG and the consonant to force an "un-joined"
    subscript rendering, e.g. in dictionaries).
    """
    base_type = classify_codepoint(ord(text[index]))
    if base_type is not CharacterType.CONSONANT:
        return base_type

    j = index - 1
    while j >= 0 and ord(text[j]) in (ZWNJ, ZWJ):
        j -= 1
    if j >= 0 and ord(text[j]) == cp_db.COENG_CODEPOINT:
        return CharacterType.SUBSCRIPT_CONSONANT
    return base_type


def get_consonant_series(text: str, index: int) -> str | None:
    """Series ('a' or 'o') of the consonant/independent-vowel at `index`,
    or None if that position is not a consonant or independent vowel.

    Independent vowels do not have a "series" property themselves, but
    conventionally pattern with the a-series for any following signs;
    they are reported as 'a' here for that reason.
    """
    ch = text[index]
    cp = ord(ch)
    consonant = cp_db.CONSONANTS_BY_CODEPOINT.get(cp)
    if consonant is not None:
        return consonant.series
    if cp in cp_db.INDEPENDENT_VOWELS_BY_CODEPOINT:
        return "a"
    return None


def unicode_name(ch: str) -> str:
    return unicodedata.name(ch, "UNKNOWN")
