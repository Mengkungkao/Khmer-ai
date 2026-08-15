"""Grapheme-cluster-level Khmer tokenizer (Tokenizer Lab, Version B).

Each token is one Khmer grapheme cluster from
`khmer_language.unicode.grapheme` (a base letter plus any stacked
subscripts/vowel/signs), so a COENG+consonant subscript is never split
from its base the way the character tokenizer splits it.
"""

from __future__ import annotations

from ..unicode.grapheme import grapheme_strings
from .base import BaseTokenizer


class GraphemeTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        return grapheme_strings(text)
