"""Khmer grammar engine (README.md sections 8 and 24).

The rule-based half of the stack: a closed-class function-word database
plus a marker-detecting analyzer. Khmer verbs never inflect, so tense,
aspect, negation and mood appear as separate visible words rather than
morphology - which makes them detectable without a trained model.
"""

from .analyzer import (
    Match,
    SentenceAnalysis,
    analyze_sentence,
    analyze_text,
    format_analysis,
)
from .function_words import (
    ALL_FUNCTION_WORDS,
    ASPECT_TENSE,
    MODAL,
    NEGATION,
    PRONOUN,
    QUESTION,
    STRUCTURAL,
    FunctionWord,
    WordClass,
    by_class,
    lookup,
)

__all__ = [
    "Match",
    "SentenceAnalysis",
    "analyze_sentence",
    "analyze_text",
    "format_analysis",
    "ALL_FUNCTION_WORDS",
    "ASPECT_TENSE",
    "MODAL",
    "NEGATION",
    "PRONOUN",
    "QUESTION",
    "STRUCTURAL",
    "FunctionWord",
    "WordClass",
    "by_class",
    "lookup",
]
