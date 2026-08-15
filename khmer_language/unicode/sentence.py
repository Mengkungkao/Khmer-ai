"""Khmer sentence segmentation (`KhmerSentenceSegmenter`).

Khmer marks sentence/clause boundaries primarily with KHAN (។) and
BARIYOOSAN (៕), rather than a period. ASCII `.!?` are also honored so
mixed-language text splits sensibly. This is boundary detection only,
not clause-level grammatical analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

_SENTENCE_END_CODEPOINTS = {0x17D4, 0x17D5} | {ord(c) for c in ".!?"}


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def segment_sentences(text: str) -> list[Sentence]:
    """Split `text` into sentences. `start`/`end` span the original text
    (including trailing punctuation/whitespace); `.text` is trimmed.
    """
    sentences: list[Sentence] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        if ord(text[i]) in _SENTENCE_END_CODEPOINTS:
            end = i + 1
            # Absorb immediately repeated/mixed terminators (e.g. "។។",
            # "?!") into the same boundary instead of emitting an empty
            # punctuation-only "sentence" between them.
            while end < n and ord(text[end]) in _SENTENCE_END_CODEPOINTS:
                end += 1
            chunk = text[start:end].strip()
            if chunk:
                sentences.append(Sentence(text=chunk, start=start, end=end))
            start = end
            i = end
        else:
            i += 1

    tail = text[start:n].strip()
    if tail:
        sentences.append(Sentence(text=tail, start=start, end=n))

    return sentences


def sentence_strings(text: str) -> list[str]:
    return [s.text for s in segment_sentences(text)]
