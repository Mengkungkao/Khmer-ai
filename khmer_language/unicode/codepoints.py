"""Machine-readable database of the Khmer Unicode block (U+1780-U+17FF).

This is the project's source of truth for Khmer character data (Project 1,
see README.md section "Phase 0 - Khmer Language Foundation"). Every table
below was built from the two authorities that matter for this kind of data:

  1. `unicodedata` (stdlib) for codepoint <-> name <-> general category.
     This is exact and version-pinned to the Python install, so glyphs are
     always derived with ``chr(codepoint)`` rather than typed by hand -
     hand-copied Khmer glyphs are a real source of transcription bugs.
  2. Published Khmer orthography references for the linguistic metadata
     that Unicode itself does not encode: consonant register ("series"),
     romanization and IPA. Two consonants are well-known exceptions to the
     otherwise regular "a,a,o,o,o" per-row register pattern: NNO (na) and
     LA. Where a source could not be confidently cross-checked (a handful
     of rare independent vowels), the field is left as ``None`` rather
     than guessed - see ``notes`` on those entries.

Consonant "series" (register) matters because most dependent vowel signs
have two different readings depending on whether they attach to an
a-series or o-series consonant - see DEPENDENT_VOWELS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Consonant:
    codepoint: int
    name: str
    series: str  # "a" or "o"
    romanization: str
    ipa: str
    obsolete: bool = False
    note: str = ""

    @property
    def char(self) -> str:
        return chr(self.codepoint)

    @property
    def subscript_form(self) -> str:
        """This consonant written as a subscript: COENG + consonant."""
        return chr(COENG_CODEPOINT) + self.char


@dataclass(frozen=True)
class IndependentVowel:
    codepoint: int
    name: str
    romanization: Optional[str]
    ipa: Optional[str]
    note: str = ""

    @property
    def char(self) -> str:
        return chr(self.codepoint)


@dataclass(frozen=True)
class DependentVowel:
    codepoint: int
    name: str
    visual_position: str
    a_series_ipa: Optional[str]
    a_series_romanization: Optional[str]
    o_series_ipa: Optional[str]
    o_series_romanization: Optional[str]

    @property
    def char(self) -> str:
        return chr(self.codepoint)


@dataclass(frozen=True)
class Sign:
    codepoint: int
    name: str
    kind: str  # SIGN, REGISTER_SHIFTER, COENG, PUNCTUATION, CURRENCY, OTHER
    note: str = ""

    @property
    def char(self) -> str:
        return chr(self.codepoint)


@dataclass(frozen=True)
class NumeralSymbol:
    codepoint: int
    name: str
    value: int
    note: str = ""

    @property
    def char(self) -> str:
        return chr(self.codepoint)


COENG_CODEPOINT = 0x17D2

# ---------------------------------------------------------------------------
# Consonants: U+1780-U+17A2 (35 codepoints; 33 in modern use, 2 obsolete)
# Row pattern for the 5 consonant "series" blocks (velar/palatal/retroflex/
# dental/labial) is normally a,a,o,o,o. NNO (retroflex n) and LA are the
# two well-documented exceptions kept here rather than "corrected" away.
# ---------------------------------------------------------------------------
CONSONANTS: tuple[Consonant, ...] = (
    Consonant(0x1780, "KA", "a", "k", "k"),
    Consonant(0x1781, "KHA", "a", "kh", "kʰ"),
    Consonant(0x1782, "KO", "o", "k", "k"),
    Consonant(0x1783, "KHO", "o", "kh", "kʰ"),
    Consonant(0x1784, "NGO", "o", "ng", "ŋ"),
    Consonant(0x1785, "CA", "a", "ch", "c"),
    Consonant(0x1786, "CHA", "a", "chh", "cʰ"),
    Consonant(0x1787, "CO", "o", "ch", "c"),
    Consonant(0x1788, "CHO", "o", "chh", "cʰ"),
    Consonant(0x1789, "NYO", "o", "nh", "ɲ"),
    Consonant(0x178A, "DA", "a", "d", "ɗ"),
    Consonant(0x178B, "TTHA", "a", "th", "tʰ", note="rare, mostly Pali/Sanskrit loanwords"),
    Consonant(0x178C, "DO", "o", "d", "ɗ"),
    Consonant(0x178D, "TTHO", "o", "th", "tʰ", note="rare, mostly Pali/Sanskrit loanwords"),
    Consonant(0x178E, "NNO", "a", "n", "n", note="exception: a-series despite being the row-5 nasal"),
    Consonant(0x178F, "TA", "a", "t", "t"),
    Consonant(0x1790, "THA", "a", "th", "tʰ"),
    Consonant(0x1791, "TO", "o", "t", "t"),
    Consonant(0x1792, "THO", "o", "th", "tʰ"),
    Consonant(0x1793, "NO", "o", "n", "n"),
    Consonant(0x1794, "BA", "a", "b/p", "ɓ/p", note="pronounced p in final position"),
    Consonant(0x1795, "PHA", "a", "ph", "pʰ"),
    Consonant(0x1796, "PO", "o", "p", "p"),
    Consonant(0x1797, "PHO", "o", "ph", "pʰ"),
    Consonant(0x1798, "MO", "o", "m", "m"),
    Consonant(0x1799, "YO", "o", "y", "j"),
    Consonant(0x179A, "RO", "o", "r", "r"),
    Consonant(0x179B, "LO", "o", "l", "l"),
    Consonant(0x179C, "VO", "o", "v", "ʋ"),
    Consonant(0x179D, "SHA", "a", "ś", "ɕ", obsolete=True, note="obsolete, Sanskrit transliteration only"),
    Consonant(0x179E, "SSO", "a", "ṣ", "ʂ", obsolete=True, note="obsolete, Sanskrit transliteration only"),
    Consonant(0x179F, "SA", "a", "s", "s"),
    Consonant(0x17A0, "HA", "a", "h", "h"),
    Consonant(0x17A1, "LA", "o", "l", "l", note="exception: o-series; has no subscript form"),
    Consonant(0x17A2, "QA", "a", "'", "ʔ", note="glottal stop; also used as a null/vowel-carrying initial"),
)

# ---------------------------------------------------------------------------
# Independent vowels: U+17A3-U+17B3 (17 codepoints)
# QAQ/QAA are legacy compatibility codepoints, rare in modern text. QUK/QUU
# romanization could not be confidently cross-checked against a second
# source, so they are left as None rather than guessed.
# ---------------------------------------------------------------------------
INDEPENDENT_VOWELS: tuple[IndependentVowel, ...] = (
    IndependentVowel(0x17A3, "QAQ", None, None, note="rare/legacy in modern Khmer orthography"),
    IndependentVowel(0x17A4, "QAA", None, None, note="rare/legacy in modern Khmer orthography"),
    IndependentVowel(0x17A5, "QI", "ĕ", "ʔə~ʔɨ~ʔəj"),
    IndependentVowel(0x17A6, "QII", "ei", "ʔəj"),
    IndependentVowel(0x17A7, "QU", "ŏ/ŭ", "ʔo~ʔu~ʔao"),
    IndependentVowel(0x17A8, "QUK", None, None, note="uncommon; verify against a dedicated reference"),
    IndependentVowel(0x17A9, "QUU", None, None, note="uncommon; verify against a dedicated reference"),
    IndependentVowel(0x17AA, "QUUV", "âu", "ʔəw", note="moderate confidence"),
    IndependentVowel(0x17AB, "RY", "rœ̆", "rɨ"),
    IndependentVowel(0x17AC, "RYY", "rœ", "rɨː"),
    IndependentVowel(0x17AD, "LY", "lœ̆", "lɨ"),
    IndependentVowel(0x17AE, "LYY", "lœ", "lɨː"),
    IndependentVowel(0x17AF, "QE", "ê", "ʔae~ʔɛː~ʔeː"),
    IndependentVowel(0x17B0, "QAI", "ai", "ʔaj"),
    IndependentVowel(0x17B1, "QOO TYPE ONE", "aô", "ʔao"),
    IndependentVowel(0x17B2, "QOO TYPE TWO", "aô", "ʔao", note="stylistic variant of QOO TYPE ONE"),
    IndependentVowel(0x17B3, "QAU", "au", "ʔaw"),
)

# ---------------------------------------------------------------------------
# Inherent vowels: U+17B4-U+17B5. Normally invisible/unused in ordinary
# text (every consonant already carries an implicit inherent vowel); the
# Unicode Standard notes these should not be used in typical modern text.
# ---------------------------------------------------------------------------
INHERENT_VOWEL_CODEPOINTS: tuple[int, ...] = (0x17B4, 0x17B5)

# ---------------------------------------------------------------------------
# Dependent (diacritical) vowel signs: U+17B6-U+17C5 (16 codepoints).
# Pronunciation depends on the series of the base consonant they attach to.
# `visual_position` is supplementary typographic metadata (best effort) -
# the actual Unicode *storage* order is always AFTER the base consonant,
# regardless of where the glyph renders. That storage-order fact is what
# grapheme.py relies on; visual_position is not load-bearing for parsing.
# ---------------------------------------------------------------------------
DEPENDENT_VOWELS: tuple[DependentVowel, ...] = (
    DependentVowel(0x17B6, "AA", "right", "aː", "a", "iːə", "éa"),
    DependentVowel(0x17B7, "I", "above", "ə", "ĕ", "ɨ~i", "ĭ"),
    DependentVowel(0x17B8, "II", "above", "ej", "ei", "iː", "i"),
    DependentVowel(0x17B9, "Y", "above", "ə", "œ̆", "ɨ", "ẏ"),
    DependentVowel(0x17BA, "YY", "above", "əː", "œ", "ɨː", "ȳ"),
    DependentVowel(0x17BB, "U", "below", "o", "ŏ", "u", "ŭ"),
    DependentVowel(0x17BC, "UU", "below", "oː", "o", "uː", "u"),
    DependentVowel(0x17BD, "UA", "below-right", "uə", "uŏ", None, None),
    DependentVowel(0x17BE, "OE", "right", "aə", "aeu", "əː", "eu"),
    DependentVowel(0x17BF, "YA", "right", "ɨə", "œă", None, None),
    DependentVowel(0x17C0, "IE", "right", "iə", "iĕ", None, None),
    DependentVowel(0x17C1, "E", "left", "eː", "é", None, None),
    DependentVowel(0x17C2, "AE", "left", "ae", "ê", "ɛː", "ae"),
    DependentVowel(0x17C3, "AI", "left", "ej", "ai", "aj", "ey"),
    DependentVowel(0x17C4, "OO", "left-right", "ao", "aô", "oː", "oŭ"),
    DependentVowel(0x17C5, "AU", "right", "aɨ", "au", "əɨ", "ŏu"),
)

# ---------------------------------------------------------------------------
# Signs, register shifters, coeng, punctuation, currency, other.
# ---------------------------------------------------------------------------
SIGNS: tuple[Sign, ...] = (
    Sign(0x17C6, "NIKAHIT", "SIGN", "final nasalization; also used as a vowel component in some words"),
    Sign(0x17C7, "REAHMUK", "SIGN", "adds a final aspiration-like -h quality"),
    Sign(0x17C8, "YUUKALEAPINTU", "SIGN", "rare sign marking a specific vowel-length reading"),
    Sign(0x17C9, "MUUSIKATOAN", "REGISTER_SHIFTER", "shifts an a-series consonant to read with o-series vowel quality"),
    Sign(0x17CA, "TRIISAP", "REGISTER_SHIFTER", "shifts an o-series consonant to read with a-series vowel quality"),
    Sign(0x17CB, "BANTOC", "SIGN", "shortens the vowel of the syllable"),
    Sign(0x17CC, "ROBAT", "SIGN", "marks a historical preceding r-, written above the consonant"),
    Sign(0x17CD, "TOANDAKHIAT", "SIGN", "commonly called 'asat'; cancels/silences the inherent vowel of the consonant it attaches to"),
    Sign(0x17CE, "KAKABAT", "SIGN", "rare repetition/emphasis mark"),
    Sign(0x17CF, "AHSDA", "SIGN", "rare sign"),
    Sign(0x17D0, "SAMYOK SANNYA", "SIGN", "rare ligature-like sign used in a few fixed words"),
    Sign(0x17D1, "VIRIAM", "SIGN", "rare; marks a consonant as unpronounced, largely unused in modern Khmer"),
    Sign(0x17D2, "COENG", "COENG", "the subscript-forming sign: COENG + consonant renders that consonant as a subscript"),
    Sign(0x17D3, "BATHAMASAT", "SIGN", "rare diacritic"),
    Sign(0x17D4, "KHAN", "PUNCTUATION", "clause/sentence-final punctuation, similar role to a period"),
    Sign(0x17D5, "BARIYOOSAN", "PUNCTUATION", "section/paragraph end marker"),
    Sign(0x17D6, "CAMNUC PII KUUH", "PUNCTUATION", "repetition sign: repeat the preceding word"),
    Sign(0x17D7, "LEK TOO", "PUNCTUATION", "rare sign, historically used as a multiplier in numeral notation"),
    Sign(0x17D8, "BEYYAL", "PUNCTUATION", "enumeration mark, historically similar in use to 'etc.'"),
    Sign(0x17D9, "PHNAEK MUAN", "PUNCTUATION", "decorative section marker used in older/ceremonial texts"),
    Sign(0x17DA, "KOOMUUT", "PUNCTUATION", "decorative text-end marker used in older/literary texts"),
    Sign(0x17DB, "RIEL SIGN", "CURRENCY", "Cambodian riel currency symbol"),
    Sign(0x17DC, "AVAKRAHASANYA", "OTHER", "rare letter-like sign, lengthens an initial vowel in a few words"),
    Sign(0x17DD, "ATTHACAN", "SIGN", "rare diacritic, drawn as a small circle above"),
)

DIGITS: tuple[NumeralSymbol, ...] = tuple(
    NumeralSymbol(0x17E0 + i, name, i)
    for i, name in enumerate(
        ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    )
)

LEK_ATTAK: tuple[NumeralSymbol, ...] = tuple(
    NumeralSymbol(0x17F0 + i, f"LEK ATTAK {name}", i, note="alternate/formal numeral symbol set")
    for i, name in enumerate(
        ["SON", "MUOY", "PII", "BEI", "BUON", "PRAM", "PRAM-MUOY", "PRAM-PII", "PRAM-BEI", "PRAM-BUON"]
    )
)

# ---------------------------------------------------------------------------
# Lookup indexes
# ---------------------------------------------------------------------------
CONSONANTS_BY_CODEPOINT = {c.codepoint: c for c in CONSONANTS}
INDEPENDENT_VOWELS_BY_CODEPOINT = {v.codepoint: v for v in INDEPENDENT_VOWELS}
DEPENDENT_VOWELS_BY_CODEPOINT = {v.codepoint: v for v in DEPENDENT_VOWELS}
SIGNS_BY_CODEPOINT = {s.codepoint: s for s in SIGNS}
DIGITS_BY_CODEPOINT = {d.codepoint: d for d in DIGITS}
LEK_ATTAK_BY_CODEPOINT = {d.codepoint: d for d in LEK_ATTAK}

CONSONANTS_BY_CHAR = {c.char: c for c in CONSONANTS}
INDEPENDENT_VOWELS_BY_CHAR = {v.char: v for v in INDEPENDENT_VOWELS}
DEPENDENT_VOWELS_BY_CHAR = {v.char: v for v in DEPENDENT_VOWELS}
SIGNS_BY_CHAR = {s.char: s for s in SIGNS}
DIGITS_BY_CHAR = {d.char: d for d in DIGITS}
LEK_ATTAK_BY_CHAR = {d.char: d for d in LEK_ATTAK}


def _record(obj, **extra) -> dict:
    d = asdict(obj)
    d["codepoint_hex"] = f"U+{d['codepoint']:04X}"
    d.update(extra)
    return d


def all_records() -> dict:
    """Return the full database as plain dict/list data (JSON-serializable)."""
    return {
        "consonants": [_record(c, char=c.char, subscript_form=c.subscript_form) for c in CONSONANTS],
        "independent_vowels": [_record(v, char=v.char) for v in INDEPENDENT_VOWELS],
        "inherent_vowels": [
            {"codepoint": cp, "codepoint_hex": f"U+{cp:04X}", "char": chr(cp)}
            for cp in INHERENT_VOWEL_CODEPOINTS
        ],
        "dependent_vowels": [_record(v, char=v.char) for v in DEPENDENT_VOWELS],
        "signs": [_record(s, char=s.char) for s in SIGNS],
        "digits": [_record(d, char=d.char) for d in DIGITS],
        "lek_attak": [_record(d, char=d.char) for d in LEK_ATTAK],
    }


def export_json(path: str | Path) -> None:
    """Write the full database to a JSON file (the "machine-readable Khmer
    linguistic database" called for in the project README, section 3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_records(), f, ensure_ascii=False, indent=2)


DEFAULT_EXPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "khmer_characters.json"
