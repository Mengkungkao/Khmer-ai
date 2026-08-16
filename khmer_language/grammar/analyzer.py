"""Rule-based Khmer grammatical analysis.

Detects tense, aspect, negation, modality and sentence type by finding
function words, which works precisely because Khmer verbs never inflect:
the grammar is carried by separate words in fixed positions rather than
by morphology, so it is visible without a morphological analyzer.

Scope, stated plainly: this identifies grammatical MARKERS, not
grammatical structure. It reports "this clause is negated and marked
future"; it does not parse constituents, resolve scope, or judge whether
a sentence is well formed. Real grammaticality checking needs a parser or
a trained model, which is why `evaluation/error_analyzer.py` still
reports grammar as UNAVAILABLE - this module deliberately does not
pretend to close that gap.

Detection is done on grapheme-aware substring matching rather than word
segmentation, because Khmer is written without spaces and the project has
no dictionary-based segmenter (see `unicode/word.py`). That means a
function word occurring inside a longer word can be matched spuriously;
`SentenceAnalysis.confidence` reflects that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..unicode.sentence import sentence_strings
from .function_words import (
    ALL_FUNCTION_WORDS,
    CONTENT_WORDS_CONTAINING_FUNCTION_WORDS,
    FunctionWord,
    WordClass,
)


@dataclass(frozen=True)
class Match:
    word: FunctionWord
    index: int


@dataclass(frozen=True)
class SentenceAnalysis:
    text: str
    matches: tuple[Match, ...] = field(default=())

    def of_class(self, word_class: WordClass) -> tuple[Match, ...]:
        return tuple(m for m in self.matches if m.word.word_class is word_class)

    @property
    def is_negated(self) -> bool:
        """Khmer negation is a circumfix: មិន before the verb, ទេ closing
        the clause. Requiring a pre-verbal marker avoids counting every
        clause-final ទេ as negation, since that particle also forms polar
        questions."""
        preverbal = {"មិន", "ឥត", "ពុំ"}
        return any(m.word.word in preverbal for m in self.matches)

    @property
    def is_question(self) -> bool:
        """Khmer does not invert word order for questions, so the marker
        is lexical: an opener, a question word, or a final particle."""
        if self.text.rstrip().endswith("?"):
            return True
        if any(m.word.word_class is WordClass.QUESTION for m in self.matches):
            return True
        # A clause-final ទេ with no pre-verbal negator is a polar question.
        return self.text.rstrip().rstrip("។៕").endswith("ទេ") and not self.is_negated

    @property
    def ambiguous(self) -> tuple[str, ...]:
        """Markers whose reading this analyzer cannot settle without a parser."""
        return tuple(
            f"{m.word.word}: {m.word.gloss} vs {m.word.ambiguous_with}"
            for m in self.matches
            if m.word.ambiguous_with
        )

    @property
    def tense(self) -> str:
        """Tense from markers, hedged where a marker is ambiguous.

        នឹង marks future BEFORE a verb but is the preposition "with"
        after one - and "ខ្ញុំមិនយល់ស្របនឹង..." ("I do not agree with...")
        was being reported as future tense on exactly that confusion.
        Resolving it needs to know where the verb is, which needs a
        parser, so an unresolved case is reported as ambiguous instead of
        being asserted either way.
        """
        words = {m.word.word for m in self.matches}
        if "នឹង" in words:
            return "future?" if self.ambiguous else "future"
        if "បាន" in words:
            return "past"
        return "unmarked"

    @property
    def aspect(self) -> str:
        words = {m.word.word for m in self.matches}
        if "កំពុង" in words:
            return "continuous"
        if "ហើយ" in words:
            return "perfective"
        if "ធ្លាប់" in words:
            return "experiential"
        return "unmarked"

    @property
    def modals(self) -> tuple[str, ...]:
        return tuple(m.word.word for m in self.of_class(WordClass.MODAL))

    @property
    def confidence(self) -> float:
        """How much to trust these matches.

        Khmer is written without spaces and this project has no
        dictionary-based segmenter, so a function word can be matched
        inside a longer unrelated word. Short markers are far more prone
        to that than long ones, so confidence falls as the matched words
        get shorter.
        """
        if not self.matches:
            return 1.0
        mean_length = sum(len(m.word.word) for m in self.matches) / len(self.matches)
        return min(1.0, mean_length / 4.0)


_CLAUSE_ENDINGS = "។៕?!\n"


def _is_clause_final(text: str, start: int, length: int) -> bool:
    """Whether a match sits at the end of a clause.

    Everything after it must be clause-ending punctuation or whitespace.
    """
    return text[start + length :].strip(_CLAUSE_ENDINGS + " \t").strip() == ""


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by content words that merely CONTAIN a
    function word, so matches inside them can be ignored."""
    spans: list[tuple[int, int]] = []
    for content_word in CONTENT_WORDS_CONTAINING_FUNCTION_WORDS:
        start = text.find(content_word)
        while start != -1:
            spans.append((start, start + len(content_word)))
            start = text.find(content_word, start + 1)
    return spans


def analyze_sentence(text: str) -> SentenceAnalysis:
    matches: list[Match] = []
    masked = _masked_spans(text)

    # Longest first, so ហេតុអ្វី is not reported as អ្វី.
    for word in sorted(ALL_FUNCTION_WORDS, key=lambda w: -len(w.word)):
        start = text.find(word.word)
        while start != -1:
            end = start + len(word.word)
            overlaps = any(m.index <= start < m.index + len(m.word.word) for m in matches)
            positioned = not word.clause_final_only or _is_clause_final(text, start, len(word.word))
            # Ignore a match that lies wholly inside a longer content word
            # (ជា inside កម្ពុជា), unless the content word IS this word.
            buried = any(
                s <= start and end <= e and (e - s) > len(word.word) for s, e in masked
            )
            if not overlaps and positioned and not buried:
                matches.append(Match(word=word, index=start))
            start = text.find(word.word, start + 1)

    return SentenceAnalysis(text=text, matches=tuple(sorted(matches, key=lambda m: m.index)))


def analyze_text(text: str) -> list[SentenceAnalysis]:
    return [analyze_sentence(s) for s in sentence_strings(text)]


def format_analysis(analysis: SentenceAnalysis) -> str:
    lines = [f"{analysis.text}"]
    lines.append(
        f"  type: {'question' if analysis.is_question else 'statement'}"
        f"{' (negated)' if analysis.is_negated else ''}"
    )
    lines.append(f"  tense: {analysis.tense}   aspect: {analysis.aspect}")
    if analysis.modals:
        lines.append(f"  modals: {', '.join(analysis.modals)}")
    if analysis.matches:
        found = ", ".join(f"{m.word.word} ({m.word.gloss})" for m in analysis.matches[:8])
        lines.append(f"  markers: {found}")
    lines.append(f"  confidence: {analysis.confidence:.2f}")
    return "\n".join(lines)
