"""Khmer spelling checking against the lexicon.

`evaluation/error_analyzer.py` has reported Spelling as UNAVAILABLE
because it "needs the Khmer dictionary (README section 8)". The lexicon
now exists, so this implements it.

The central design decision is that **absence from the dictionary is not
evidence of misspelling.** Measured on this project's own data, 24% of
Wikipedia words are outside the lexicon - overwhelmingly transliterated
foreign names and technical terms, all correctly spelled. Flagging those
as errors would make the checker useless and, worse, confidently wrong.

So unknown words are separated into two kinds:

  MISSPELLED  a small edit away from a real dictionary word. That is
              positive evidence: the writer nearly wrote a known word,
              and a concrete correction can be offered.
  UNKNOWN     not close to anything in the dictionary. Most likely a
              proper noun, loanword or genuinely absent entry - reported
              as unknown, never as an error.

Candidate lookup uses the deletion-index trick (SymSpell): precompute
every one-deletion variant of each dictionary word, then compare against
the one-deletion variants of the query. Two words are within edit
distance 1 exactly when their deletion sets intersect. This matters for
Khmer specifically - the script has roughly 2,600 grapheme clusters, so
generating substitution and insertion candidates directly would mean
thousands of probes per word, while deletions cost only the word's
length.

Edits are over grapheme clusters, not code points, so a single mistyped
Khmer cluster counts as one error rather than three or four.

**Known limitation, inherent to unsegmented scripts.** Checking a word in
isolation is reliable; checking one inside running text is not, because
the segmenter may absorb the error. Dropping a grapheme from ប្រទេស gives
ប្រទេ, which segments cleanly into ប្រ + ទេ - two real words - so no
unknown word ever reaches the checker and the typo passes silently.
Detection therefore has high precision and limited recall on running
text. Closing that gap needs the segmenter to score how *plausible* a
segmentation is, not just whether every piece is a word, which in turn
wants a language model over words rather than a dictionary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from ..unicode.grapheme import grapheme_strings
from .segmenter import KhmerLexicon, WordSegmenter, _is_khmer_word


class Verdict(str, Enum):
    CORRECT = "CORRECT"
    MISSPELLED = "MISSPELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class WordCheck:
    word: str
    verdict: Verdict
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpellingReport:
    checks: tuple[WordCheck, ...]

    @property
    def misspelled(self) -> tuple[WordCheck, ...]:
        return tuple(c for c in self.checks if c.verdict is Verdict.MISSPELLED)

    @property
    def unknown(self) -> tuple[WordCheck, ...]:
        return tuple(c for c in self.checks if c.verdict is Verdict.UNKNOWN)

    @property
    def score(self) -> float:
        """Fraction of Khmer words that are not likely misspellings.

        Unknown-but-plausible words count as correct, because treating a
        proper noun as an error is a worse failure than missing a typo.
        """
        if not self.checks:
            return 1.0
        return 1.0 - len(self.misspelled) / len(self.checks)


def _deletions(units: list[str]) -> set[str]:
    """Every string formed by deleting exactly one grapheme."""
    return {"".join(units[:i] + units[i + 1 :]) for i in range(len(units))}


class SpellChecker:
    def __init__(self, lexicon: KhmerLexicon, max_suggestions: int = 3):
        self.lexicon = lexicon
        self.segmenter = WordSegmenter(lexicon)
        self.max_suggestions = max_suggestions

        # word-with-one-deletion -> the dictionary words producing it
        self._index: dict[str, list[str]] = defaultdict(list)
        for word in lexicon.entries:
            units = grapheme_strings(word)
            if len(units) < 2:
                continue
            for variant in _deletions(units):
                self._index[variant].append(word)

    def _candidates(self, word: str) -> list[str]:
        units = grapheme_strings(word)
        if len(units) < 2:
            return []

        found: set[str] = set()
        # A deleted-by-one form of the query that IS a dictionary word
        # means the query has one extra grapheme.
        for variant in _deletions(units):
            if variant in self.lexicon:
                found.add(variant)
            found.update(self._index.get(variant, ()))
        # The query itself matching a dictionary word's deletion means the
        # query is missing one grapheme.
        found.update(self._index.get(word, ()))
        found.discard(word)

        return self._rank(word, found)

    def _rank(self, query: str, candidates: set[str]) -> list[str]:
        """Order corrections by how plausibly they were the intended word.

        Frequency alone is a poor guide and was actively misleading here:
        for a truncated កម្ពុជា it proposed ក and កម, both common short
        words and both technically one edit away, while the obvious
        correction ranked lower. Short frequent words are edit-1
        neighbours of a great many things.

        Shared prefix is the stronger signal, because Khmer typing and
        truncation errors overwhelmingly affect the END of a word - the
        writer got most of it right. Frequency then breaks ties.
        """
        query_units = grapheme_strings(query)

        def shared_prefix(candidate: str) -> int:
            candidate_units = grapheme_strings(candidate)
            shared = 0
            for a, b in zip(query_units, candidate_units):
                if a != b:
                    break
                shared += 1
            return shared

        return sorted(
            candidates,
            key=lambda w: (-shared_prefix(w), -self.lexicon.entries[w].count),
        )[: self.max_suggestions]

    def check_word(self, word: str) -> WordCheck:
        if word in self.lexicon:
            return WordCheck(word, Verdict.CORRECT)

        suggestions = self._candidates(word)
        if suggestions:
            return WordCheck(word, Verdict.MISSPELLED, tuple(suggestions))
        return WordCheck(word, Verdict.UNKNOWN)

    def check(self, text: str) -> SpellingReport:
        words = [w.text for w in self.segmenter.segment(text) if _is_khmer_word(w.text)]
        return SpellingReport(checks=tuple(self.check_word(w) for w in words))
