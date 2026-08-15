"""Khmer word segmentation (`KhmerWordSegmenter`) - v1, boundary-hint only.

Khmer does not use spaces between words, so real word segmentation needs
a dictionary or a statistical model trained on a segmented corpus -
neither exists yet in this project (see README.md Project 3/4: corpus +
tokenizer lab). This v1 only splits on boundary signals that are already
unambiguous without one:

- ZWSP (U+200B), the standard invisible word-boundary hint used in a lot
  of real Khmer digital text and corpora
- whitespace
- punctuation (including the KHAN/BARIYOOSAN sentence marks)

A run of Khmer grapheme clusters containing none of those hints is
returned as a single "word" even though it may linguistically contain
several - that is the documented limitation of this v1, not a bug. It
will get sharper once Project 3/4 exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from .character_types import CharacterType
from .grapheme import segment_graphemes

_BOUNDARY_TYPES = frozenset(
    {CharacterType.WHITESPACE, CharacterType.PUNCTUATION, CharacterType.ZERO_WIDTH}
)


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int


def segment_words(text: str) -> list[Word]:
    words: list[Word] = []
    current_start: int | None = None

    graphemes = segment_graphemes(text)
    for g in graphemes:
        if g.base_type in _BOUNDARY_TYPES:
            if current_start is not None:
                words.append(Word(text=text[current_start:g.start], start=current_start, end=g.start))
                current_start = None
        elif current_start is None:
            current_start = g.start

    if current_start is not None:
        words.append(Word(text=text[current_start:], start=current_start, end=len(text)))

    return words


def word_strings(text: str) -> list[str]:
    return [w.text for w in segment_words(text)]
