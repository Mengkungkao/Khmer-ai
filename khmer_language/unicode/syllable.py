"""Khmer syllable segmentation (`KhmerSyllableTokenizer`).

Current approximation: a Khmer orthographic syllable == one grapheme
cluster (KCC). This holds for the large majority of real text, since a
base letter plus its stacked subscripts, register shifter, vowel and
signs is exactly what a syllable is. It is a documented simplification,
not a guarantee: genuinely disambiguating rarer cases (e.g. certain
consonant sequences without an explicit COENG that are nonetheless read
as two syllables) needs a pronunciation dictionary or statistical model,
which does not exist yet in this project (see README.md Project 3/4:
corpus + tokenizer lab). Treat this module as a placeholder that will be
sharpened once that data exists, not as a finished syllabifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grapheme import Grapheme, segment_graphemes


@dataclass(frozen=True)
class Syllable:
    text: str
    start: int
    end: int
    grapheme: Grapheme


def segment_syllables(text: str) -> list[Syllable]:
    return [
        Syllable(text=g.text, start=g.start, end=g.end, grapheme=g)
        for g in segment_graphemes(text)
    ]


def syllable_strings(text: str) -> list[str]:
    return [s.text for s in segment_syllables(text)]
