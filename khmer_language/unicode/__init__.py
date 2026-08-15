"""Khmer Unicode analysis engine (Project 1).

Character classification, grapheme/syllable segmentation, structural
validation and best-effort transliteration for Khmer text, built directly
on `unicodedata` with no third-party dependencies.
"""

from .character_types import CharacterType, classify, classify_codepoint
from .cluster import ConsonantCluster, analyze_cluster
from .grapheme import Grapheme, segment_graphemes
from .normalizer import normalize
from .syllable import Syllable, segment_syllables
from .transliterator import transliterate
from .validator import ValidationIssue, is_valid, validate

__all__ = [
    "CharacterType",
    "classify",
    "classify_codepoint",
    "ConsonantCluster",
    "analyze_cluster",
    "Grapheme",
    "segment_graphemes",
    "normalize",
    "Syllable",
    "segment_syllables",
    "transliterate",
    "ValidationIssue",
    "is_valid",
    "validate",
]
