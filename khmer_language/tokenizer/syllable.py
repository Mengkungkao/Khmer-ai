"""Syllable-level Khmer tokenizer (Tokenizer Lab).

Currently identical in output to `GraphemeTokenizer`, since
`khmer_language.unicode.syllable` approximates a syllable as one
grapheme cluster (see that module's docstring for the known limits).
Kept as a separate tokenizer class so callers/comparisons name the unit
they mean, and so this can diverge from the grapheme tokenizer later
without changing call sites.
"""

from __future__ import annotations

from ..unicode.syllable import syllable_strings
from .base import BaseTokenizer


class SyllableTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        return syllable_strings(text)
