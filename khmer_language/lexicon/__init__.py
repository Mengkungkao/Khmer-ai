"""Khmer lexicon and dictionary-based word segmentation (README section 8).

Khmer is written without spaces, so word boundaries need a dictionary.
This package holds the lexicon (headwords, parts of speech, corpus
frequencies) and a Viterbi segmenter that uses it.

Lexical data is derived from Wiktionary via the Kaikki.org export, used
under CC BY-SA - the same licence as the Khmer Wikipedia corpus, so the
project gains no new restrictions.
"""

from .segmenter import (
    DEFAULT_LEXICON,
    KhmerLexicon,
    LexiconEntry,
    Word,
    WordSegmenter,
)

__all__ = [
    "DEFAULT_LEXICON",
    "KhmerLexicon",
    "LexiconEntry",
    "Word",
    "WordSegmenter",
]
