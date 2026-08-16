"""Dictionary-based Khmer word segmentation.

Khmer is written without spaces, so finding word boundaries needs a
lexicon. `unicode/word.py` could only split on explicit hints (ZWSP,
punctuation) and returned a boundary-free run as a single "word"; this
module replaces that with real segmentation.

Two things make it work beyond simple dictionary lookup.

**Ambiguity is the normal case.** A run of Khmer usually admits many
valid splits, so the segmenter cannot just take the first or the longest
match. Longest-match (greedy) is the classic baseline and is genuinely
bad here: committing to a long word early can strand the remainder into
garbage, and the algorithm has no way to reconsider. Instead this uses
Viterbi over the whole run, maximizing total log-probability, so a
locally attractive long word loses if it wrecks the rest of the split.

**Frequency decides between valid splits.** A dictionary says which
strings are words; it does not say which reading is likely. Word counts
from the corpus supply that, with Laplace smoothing so a dictionary word
never seen in the corpus is still usable rather than impossible.

Unknown text remains segmentable: any span with no dictionary match falls
back to a single grapheme with a heavy penalty, so out-of-vocabulary
names and loanwords degrade to character-level rather than failing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..unicode.grapheme import grapheme_strings

DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "khmer_lexicon.jsonl"

# Log-probability charged for a span with no dictionary entry. Large
# enough that the segmenter prefers any real word, small enough that
# unknown text still segments instead of failing.
UNKNOWN_LOGPROB = -25.0

_KHMER_BLOCK = range(0x1780, 0x1800)


def _is_khmer_word(text: str) -> bool:
    """Whether a segment is Khmer script at all.

    Used to keep whitespace, punctuation and embedded Latin out of
    lexicon-coverage statistics.
    """
    return any(ord(c) in _KHMER_BLOCK for c in text)


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    pos: str
    count: int


class KhmerLexicon:
    def __init__(self, entries: list[LexiconEntry]):
        self.entries = {e.word: e for e in entries}
        self.max_word_graphemes = max(
            (len(grapheme_strings(e.word)) for e in entries), default=1
        )

        # Laplace-smoothed unigram probabilities: every dictionary word
        # gets non-zero probability even with a corpus count of zero, so
        # vocabulary the corpus happens not to contain is still usable.
        total = sum(e.count for e in entries) + len(entries)
        self._logprob = {
            e.word: math.log((e.count + 1) / total) for e in entries
        }

    def __contains__(self, word: str) -> bool:
        return word in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def logprob(self, word: str) -> float:
        return self._logprob.get(word, UNKNOWN_LOGPROB)

    def pos(self, word: str) -> str | None:
        entry = self.entries.get(word)
        return entry.pos if entry else None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "KhmerLexicon":
        path = Path(path) if path else DEFAULT_LEXICON
        if not path.exists():
            raise FileNotFoundError(
                f"no lexicon at {path}; build it with scripts/build_lexicon.py"
            )
        entries: list[LexiconEntry] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "word" not in record:  # the attribution header
                    continue
                entries.append(
                    LexiconEntry(
                        word=record["word"],
                        pos=record.get("pos", "unknown"),
                        count=int(record.get("count", 0)),
                    )
                )
        return cls(entries)


@dataclass(frozen=True)
class Word:
    text: str
    pos: str | None
    in_lexicon: bool


class WordSegmenter:
    def __init__(self, lexicon: KhmerLexicon):
        self.lexicon = lexicon

    def segment(self, text: str) -> list[Word]:
        """Split `text` into words, maximizing total log-probability."""
        units = grapheme_strings(text)
        n = len(units)
        if n == 0:
            return []

        best = [-math.inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        span = min(self.lexicon.max_word_graphemes, n)

        for end in range(1, n + 1):
            for start in range(max(0, end - span), end):
                if best[start] == -math.inf:
                    continue
                candidate = "".join(units[start:end])
                score = self.lexicon.logprob(candidate)
                if candidate not in self.lexicon:
                    # Only single graphemes may be unknown; allowing long
                    # unknown spans would let the segmenter "explain" any
                    # text as one huge non-word.
                    if end - start > 1:
                        continue
                total = best[start] + score
                if total > best[end]:
                    best[end] = total
                    back[end] = start

        words: list[Word] = []
        position = n
        while position > 0:
            start = back[position]
            surface = "".join(units[start:position])
            words.append(
                Word(
                    text=surface,
                    pos=self.lexicon.pos(surface),
                    in_lexicon=surface in self.lexicon,
                )
            )
            position = start
        return list(reversed(words))

    def segment_strings(self, text: str) -> list[str]:
        return [w.text for w in self.segment(text)]

    def coverage(self, text: str) -> float:
        """Share of KHMER words found in the lexicon.

        Whitespace, punctuation and Latin characters are excluded from
        the denominator. Counting them made the metric say what it did
        not mean: a Khmer technical article scored 65% purely because it
        contained spaces and embedded English terms like "EV", each
        counted as an out-of-vocabulary word. Those are legitimately
        absent from a Khmer dictionary and are not segmentation failures.

        Read the result as a health check on the lexicon's domain
        coverage: a low score means the dictionary does not know this
        subject matter, not that the text is wrong.
        """
        khmer_words = [w for w in self.segment(text) if _is_khmer_word(w.text)]
        if not khmer_words:
            return 0.0
        return sum(1 for w in khmer_words if w.in_lexicon) / len(khmer_words)

    def unknown_words(self, text: str) -> list[str]:
        """Khmer words the lexicon does not contain - the useful signal
        for deciding what a domain-specific lexicon still needs."""
        return [
            w.text
            for w in self.segment(text)
            if _is_khmer_word(w.text) and not w.in_lexicon
        ]
