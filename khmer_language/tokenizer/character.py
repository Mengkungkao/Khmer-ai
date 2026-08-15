"""Character-level Khmer tokenizer (Tokenizer Lab, Version A).

Splits on raw Unicode code points. Simple and never produces an unknown
token for single characters seen during training, but a COENG subscript
sign and its consonant become two separate tokens, and sequences get
long. See `grapheme.py` for the linguistically-aware alternative.
"""

from __future__ import annotations

from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> list[str]:
        return list(text)
