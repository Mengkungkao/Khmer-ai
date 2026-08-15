"""Khmer Tokenizer Lab (README.md Project 4 / section 10): character,
grapheme, syllable, and grapheme-aware BPE tokenizers sharing a common
`BaseTokenizer` interface, plus a comparison harness.
"""

from .base import BaseTokenizer, Vocabulary
from .bpe import BPETokenizer
from .character import CharacterTokenizer
from .compare import SAMPLE_CORPUS, TokenizerStats, compare, format_comparison
from .grapheme import GraphemeTokenizer
from .syllable import SyllableTokenizer

__all__ = [
    "BaseTokenizer",
    "Vocabulary",
    "BPETokenizer",
    "CharacterTokenizer",
    "GraphemeTokenizer",
    "SyllableTokenizer",
    "SAMPLE_CORPUS",
    "TokenizerStats",
    "compare",
    "format_comparison",
]
