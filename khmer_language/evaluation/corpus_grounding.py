"""Grade generated Khmer against a real corpus, without a native speaker.

Machine translation cannot do this - see `reference_translation.py`,
where real Khmer, model output and pure nonsense all scored identically,
because MT is built to always produce plausible output.

This module measures something an algorithm genuinely can decide: **how
much of the generated text actually occurs in real Khmer.** Take grapheme
n-grams of the output and ask what fraction appear anywhere in a
reference corpus. Fluent Khmer reuses real sequences heavily; invented
character soup does not, however Khmer-shaped it looks.

Why n-grams rather than words: Khmer has no spaces, so "is this a real
word" needs a dictionary the project does not have. Grapheme n-grams
sidestep that entirely while still capturing whether letters combine the
way Khmer actually combines them.

The reference corpus is what defines "real", so this measures agreement
with that corpus, not correctness in the abstract. Perfectly good Khmer
using vocabulary absent from the reference will score low. That makes it
a strong *relative* measure - comparing models, checkpoints or sampling
settings against a fixed reference - and a weak absolute one. It cannot
tell you the output is meaningful, only that its building blocks are
attested; grammatical nonsense assembled from real fragments still
scores well.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..unicode.grapheme import grapheme_strings


@dataclass(frozen=True)
class GroundingScore:
    n: int
    total: int
    attested: int

    @property
    def ratio(self) -> float:
        return self.attested / self.total if self.total else 0.0


class CorpusGrounding:
    """Reusable index of the n-grams that occur in a reference corpus."""

    def __init__(self, corpus: list[str], orders: tuple[int, ...] = (2, 3, 4)):
        self.orders = orders
        self._index: dict[int, set[tuple[str, ...]]] = {n: set() for n in orders}
        for text in corpus:
            units = grapheme_strings(text)
            for n in orders:
                index = self._index[n]
                for i in range(len(units) - n + 1):
                    index.add(tuple(units[i : i + n]))

    def sizes(self) -> dict[int, int]:
        return {n: len(index) for n, index in self._index.items()}

    def score(self, text: str, n: int) -> GroundingScore:
        """Fraction of the text's n-grams that occur in the reference."""
        if n not in self._index:
            raise ValueError(f"grounding was not built for n={n}; have {sorted(self._index)}")

        units = grapheme_strings(text)
        if len(units) < n:
            return GroundingScore(n=n, total=0, attested=0)

        index = self._index[n]
        total = len(units) - n + 1
        attested = sum(
            1 for i in range(total) if tuple(units[i : i + n]) in index
        )
        return GroundingScore(n=n, total=total, attested=attested)

    def score_all(self, text: str) -> dict[int, GroundingScore]:
        return {n: self.score(text, n) for n in self.orders}

    def summary(self, text: str) -> str:
        parts = [f"{n}-gram {s.ratio:.0%}" for n, s in sorted(self.score_all(text).items())]
        return "  ".join(parts)


def distinct_n(text: str, n: int = 2) -> float:
    """Fraction of the text's n-grams that are distinct (Li et al. 2016).

    The necessary counterweight to grounding. Grounding rewards output
    made of attested sequences, so on its own it is maximized by the most
    conservative decoding possible - and greedy decoding on this model
    scores near-perfectly while emitting one repeated zero-width space,
    a distinct-token ratio of 10%.

    Any decoding setting therefore has to be judged on both: grounded
    enough to be real Khmer, diverse enough to be worth reading. Neither
    number alone identifies a good sampler.
    """
    units = grapheme_strings(text)
    if len(units) < n:
        return 0.0
    ngrams = [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]
    return len(set(ngrams)) / len(ngrams)
