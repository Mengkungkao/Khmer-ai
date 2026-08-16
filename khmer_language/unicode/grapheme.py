"""Khmer grapheme cluster segmentation ("Khmer Character Cluster" / KCC).

This is the project's `KhmerGraphemeTokenizer` (README.md section 5). It
deliberately does not rely on Python's default Unicode grapheme handling:
the standard UAX #29 extended-grapheme-cluster algorithm does not fully
cluster Khmer COENG (subscript) sequences into their base syllable, because
the consonant following COENG is an ordinary spacing letter as far as the
default algorithm is concerned. Khmer needs a script-specific rule, so this
module implements one directly instead of depending on `regex`'s `\\X` or
an ICU binding.

A Khmer grapheme cluster is, informally:

    base (consonant | independent vowel | anything else, as a singleton)
    + (COENG consonant)*      zero or more subscript consonants
    + (register shifter)?
    + (dependent vowel sign)?
    + (diacritic sign)*

Two Unicode formatting characters get special treatment because they are
invisible and do not behave like ordinary base characters:

- ZWSP (U+200B) never attaches - it is commonly inserted in Khmer digital
  text as an invisible word-boundary hint (Khmer does not use spaces
  between words), so it is always its own cluster.
- ZWNJ/ZWJ (U+200C/U+200D) always attach to the current cluster - they
  only ever appear as joining/anti-joining hints between a COENG and its
  consonant (e.g. to force an "unjoined" subscript rendering in
  dictionaries), never as a base in their own right.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from . import codepoints as cp_db
from .character_types import (
    ZWJ,
    ZWNJ,
    CharacterType,
    classify,
    classify_codepoint,
    is_attaching,
)


@dataclass(frozen=True)
class Grapheme:
    text: str
    start: int
    end: int
    char_types: tuple[tuple[str, CharacterType], ...]

    @property
    def base_type(self) -> CharacterType:
        return self.char_types[0][1]

    def __str__(self) -> str:
        return self.text


def _attaches(text: str, i: int) -> bool:
    ch = text[i]
    if ord(ch) in (ZWNJ, ZWJ):
        return True
    char_type = classify(text, i)
    if is_attaching(char_type):
        return True
    # Generic Unicode fallback for non-Khmer combining marks (e.g. Latin
    # text with a detached combining accent) so mixed-script input doesn't
    # get needlessly fragmented.
    return char_type is CharacterType.NON_KHMER and unicodedata.combining(ch) != 0


def _classify_all(text: str) -> list[CharacterType]:
    """Classify every character in one pass.

    `classify()` is context-aware only for the CONSONANT ->
    SUBSCRIPT_CONSONANT distinction, which depends on a preceding COENG.
    Resolving that in a single left-to-right sweep avoids re-deriving it
    per character, and lets the caller reuse one classification list for
    both boundary decisions and cluster contents - previously every
    character was classified twice.
    """
    types = [classify_codepoint(ord(ch)) for ch in text]
    coeng = cp_db.COENG_CODEPOINT

    for i, char_type in enumerate(types):
        if char_type is not CharacterType.CONSONANT:
            continue
        j = i - 1
        while j >= 0 and ord(text[j]) in (ZWNJ, ZWJ):
            j -= 1
        if j >= 0 and ord(text[j]) == coeng:
            types[i] = CharacterType.SUBSCRIPT_CONSONANT
    return types


def _attaches_precomputed(ch: str, char_type: CharacterType) -> bool:
    if ord(ch) in (ZWNJ, ZWJ):
        return True
    if is_attaching(char_type):
        return True
    return char_type is CharacterType.NON_KHMER and unicodedata.combining(ch) != 0


@lru_cache(maxsize=32)
def _segment_cached(text: str) -> tuple[Grapheme, ...]:
    """Segmentation memoized by exact input string.

    A corpus pipeline segments the same document several times over
    (dedup shingling, length check, unicode scoring, validation,
    repetition scoring). Profiling a Wikipedia run showed segmentation
    at 77% of total time with ~4.3 calls per document, nearly all
    redundant. Caching is safe because `Grapheme` is a frozen dataclass,
    so the returned objects cannot be mutated by one caller and observed
    by another.

    The cache is deliberately small: calls cluster tightly around whatever
    document is being processed, so a handful of entries captures almost
    every repeat without holding a whole corpus in memory.
    """
    clusters: list[Grapheme] = []
    types = _classify_all(text)
    n = len(text)
    i = 0
    while i < n:
        start = i
        i += 1
        while i < n and _attaches_precomputed(text[i], types[i]):
            i += 1
        char_types = tuple((text[k], types[k]) for k in range(start, i))
        clusters.append(Grapheme(text=text[start:i], start=start, end=i, char_types=char_types))
    return tuple(clusters)


def segment_graphemes(text: str) -> list[Grapheme]:
    """Segment `text` into Khmer grapheme clusters.

    This performs no validity checking - it clusters whatever it is given,
    including malformed sequences (e.g. a stray dependent vowel with no
    base). Use `validator.py` to check well-formedness.
    """
    # A fresh list per call, so callers may mutate it freely; the frozen
    # Grapheme objects inside are shared and must not be mutated.
    return list(_segment_cached(text))


def grapheme_strings(text: str) -> list[str]:
    """Convenience wrapper returning just the substrings."""
    return [g.text for g in _segment_cached(text)]
