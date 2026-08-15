"""Top-level `analyze(text)` entry point tying the Unicode engine together.

This is the "Khmer Unicode Explorer" from README.md Project 1: given any
Khmer text, explain its internal structure end to end - per-character
classification, grapheme clusters, syllables, structural validity and a
best-effort transliteration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .unicode.character_types import CharacterType, classify, unicode_name
from .unicode.cluster import ConsonantCluster, analyze_cluster
from .unicode.grapheme import Grapheme, segment_graphemes
from .unicode.normalizer import normalize as normalize_text
from .unicode.sentence import Sentence, segment_sentences
from .unicode.syllable import Syllable, segment_syllables
from .unicode.transliterator import transliterate
from .unicode.validator import ValidationIssue, validate
from .unicode.word import Word, segment_words


@dataclass(frozen=True)
class CharacterInfo:
    char: str
    codepoint: str
    unicode_name: str
    type: CharacterType


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    characters: tuple[CharacterInfo, ...]
    graphemes: tuple[Grapheme, ...]
    syllables: tuple[Syllable, ...]
    words: tuple[Word, ...]
    sentences: tuple[Sentence, ...]
    clusters: tuple[ConsonantCluster, ...]
    issues: tuple[ValidationIssue, ...]
    transliteration: str

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def analyze(text: str, *, normalize: bool = True) -> AnalysisResult:
    if normalize:
        text = normalize_text(text)

    characters = tuple(
        CharacterInfo(
            char=ch,
            codepoint=f"U+{ord(ch):04X}",
            unicode_name=unicode_name(ch),
            type=classify(text, i),
        )
        for i, ch in enumerate(text)
    )
    graphemes = tuple(segment_graphemes(text))
    syllables = tuple(segment_syllables(text))
    words = tuple(segment_words(text))
    sentences = tuple(segment_sentences(text))
    clusters = tuple(analyze_cluster(g) for g in graphemes)
    issues = tuple(validate(text))

    return AnalysisResult(
        text=text,
        characters=characters,
        graphemes=graphemes,
        syllables=syllables,
        words=words,
        sentences=sentences,
        clusters=clusters,
        issues=issues,
        transliteration=transliterate(text),
    )


def format_analysis(result: AnalysisResult) -> str:
    lines = [f"Input: {result.text}", ""]

    lines.append("Characters:")
    for info in result.characters:
        lines.append(f"  {info.char}   {info.codepoint:<8} {info.type.value:<20} {info.unicode_name}")

    lines.append("")
    lines.append(f"Graphemes ({len(result.graphemes)}): " + " | ".join(g.text for g in result.graphemes))
    lines.append(f"Syllables ({len(result.syllables)}): " + " | ".join(s.text for s in result.syllables))
    lines.append(f"Words ({len(result.words)}): " + " | ".join(w.text for w in result.words))
    lines.append(f"Sentences ({len(result.sentences)}): " + " | ".join(s.text for s in result.sentences))
    lines.append(f"Transliteration: {result.transliteration}")

    lines.append("")
    lines.append(f"Validity: {'PASS' if result.is_valid else 'FAIL'}")
    for issue in result.issues:
        lines.append(f"  [{issue.severity.upper()}] {issue.code} @ {issue.index}: {issue.message}")

    return "\n".join(lines)
