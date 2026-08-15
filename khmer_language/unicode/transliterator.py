"""Best-effort Khmer -> Latin transliteration.

This is explicitly NOT a phonological engine. Real Khmer pronunciation
has many exceptions this module does not model: allophonic changes on
final consonants, irregular readings for specific consonant+vowel
combinations, vowel length shortening (BANTOC), and more. What it does
implement, faithfully, is the regular part of the system that is
well-documented and mechanical:

- each consonant/independent vowel's citation romanization (codepoints.py)
- the fact that most dependent vowel signs read differently depending on
  the a-series/o-series register of the base consonant
- register shifters (MUUSIKATOAN/TRIISAP) flipping that register
- the inherent vowel (â for a-series, ô for o-series) when no dependent
  vowel sign is written
- TOANDAKHIAT ("asat") silencing the inherent vowel
- NIKAHIT/REAHMUK as trailing nasal/aspiration markers

Treat the output as a readable approximation for non-Khmer readers, not
a citable IPA transcription.
"""

from __future__ import annotations

from . import codepoints as cp_db
from .character_types import CharacterType
from .cluster import analyze_cluster
from .grapheme import segment_graphemes

_INHERENT_VOWEL_ROMANIZATION = {"a": "â", "o": "ô"}

_MUUSIKATOAN = 0x17C9
_TRIISAP = 0x17CA
_TOANDAKHIAT = 0x17CD
_NIKAHIT = 0x17C6
_REAHMUK = 0x17C7


def _effective_series(base_series: str | None, register_shifter: str | None) -> str | None:
    if register_shifter is None or base_series is None:
        return base_series
    cp = ord(register_shifter)
    if cp == _MUUSIKATOAN:
        return "o"
    if cp == _TRIISAP:
        return "a"
    return base_series


def _vowel_romanization(vowel_char: str, series: str | None) -> str:
    entry = cp_db.DEPENDENT_VOWELS_BY_CHAR.get(vowel_char)
    if entry is None:
        return vowel_char
    if series == "o" and entry.o_series_romanization:
        return entry.o_series_romanization
    if entry.a_series_romanization:
        return entry.a_series_romanization
    return entry.o_series_romanization or vowel_char


def _transliterate_cluster(grapheme) -> str:
    char, base_type = grapheme.char_types[0]

    if base_type in (CharacterType.CONSONANT,):
        cluster = analyze_cluster(grapheme)
        consonant = cp_db.CONSONANTS_BY_CHAR[char]
        series = _effective_series(consonant.series, cluster.register_shifter)

        parts = [consonant.romanization]
        for sub_char in cluster.subscripts:
            sub = cp_db.CONSONANTS_BY_CHAR.get(sub_char)
            parts.append(sub.romanization if sub else sub_char)

        diacritic_cps = {ord(d) for d in cluster.diacritics}
        if cluster.vowel is not None:
            parts.append(_vowel_romanization(cluster.vowel, series))
        elif _TOANDAKHIAT not in diacritic_cps:
            parts.append(_INHERENT_VOWEL_ROMANIZATION.get(series, ""))

        if _NIKAHIT in diacritic_cps:
            parts.append("ṃ")  # ṃ
        if _REAHMUK in diacritic_cps:
            parts.append("ḥ")  # ḥ

        return "".join(parts)

    if base_type is CharacterType.INDEPENDENT_VOWEL:
        entry = cp_db.INDEPENDENT_VOWELS_BY_CHAR.get(char)
        if entry and entry.romanization:
            return entry.romanization
        return char

    if base_type is CharacterType.DIGIT:
        entry = cp_db.DIGITS_BY_CHAR.get(char)
        return str(entry.value) if entry else char

    return grapheme.text


def transliterate(text: str) -> str:
    """Transliterate Khmer text to a readable Latin approximation."""
    return "".join(_transliterate_cluster(g) for g in segment_graphemes(text))
