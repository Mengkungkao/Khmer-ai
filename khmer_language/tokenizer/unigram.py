"""Unigram language-model tokenizer (Kudo 2018), from scratch.

The last of README section 10's tokenizer variants, and worth having
because it is fundamentally *different* from BPE rather than a variation
on it:

    BPE      starts small and GROWS, greedily merging the most frequent
             adjacent pair. Segmentation is whatever the merge sequence
             happens to produce.
    Unigram  starts with a large candidate vocabulary and PRUNES it down,
             keeping the pieces that best explain the corpus under a
             unigram language model. Segmentation is the globally
             highest-probability split, found with Viterbi.

So Unigram optimizes an explicit objective (corpus likelihood), while BPE
follows a greedy heuristic. Comparing the two on Khmer is exactly the
experiment section 10 asks for.

As with `bpe.py`, candidate pieces are built from whole Khmer grapheme
clusters, so no learned piece can ever split a COENG subscript from its
base consonant.

Training loop, per iteration:
  1. **EM** fits the piece probabilities. The E-step uses forward-backward
     over all possible segmentations (not just the best one), so a piece
     gets credit in proportion to how much probability mass flows through
     it. Done in log space, because the probability of one segmentation of
     a long sentence underflows float64 quickly.
  2. **Prune** the pieces contributing least to total corpus likelihood.

Single grapheme clusters are never pruned, which guarantees every input
remains segmentable - without that floor, Viterbi can hit a substring it
cannot cover and fail outright.

Simplification worth flagging: pruning ranks pieces by their contribution
to the likelihood (expected count x log-probability) rather than by
SentencePiece's exact leave-one-out loss, which re-segments the corpus
with each candidate removed. The exact version costs an extra Viterbi
pass per candidate; this proxy is the standard cheap approximation and
picks the same pieces in the common case.
"""

from __future__ import annotations

import math
from collections import Counter

from ..unicode.grapheme import grapheme_strings
from .base import BaseTokenizer, Vocabulary

NEG_INF = float("-inf")
# Probability floor for a grapheme never seen in training, so an unknown
# character still gets a finite (very bad) score instead of breaking Viterbi.
_UNSEEN_LOGPROB = -30.0


def _logsumexp(values: list[float]) -> float:
    finite = [v for v in values if v != NEG_INF]
    if not finite:
        return NEG_INF
    top = max(finite)
    return top + math.log(sum(math.exp(v - top) for v in finite))


class UnigramTokenizer(BaseTokenizer):
    def __init__(self, max_piece_graphemes: int = 4):
        super().__init__()
        self.max_piece_graphemes = max_piece_graphemes
        self.piece_logprobs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def _candidate_pieces(self, corpus: list[str], seed_size: int) -> dict[str, float]:
        """All grapheme-aligned substrings up to `max_piece_graphemes`,
        kept by frequency, plus every single grapheme (never prunable)."""
        counts: Counter[str] = Counter()
        singles: set[str] = set()

        for text in corpus:
            units = grapheme_strings(text)
            singles.update(units)
            for i in range(len(units)):
                for length in range(1, min(self.max_piece_graphemes, len(units) - i) + 1):
                    counts["".join(units[i : i + length])] += 1

        for single in singles:
            counts.setdefault(single, 1)

        kept = dict(counts.most_common(seed_size))
        for single in singles:  # re-add any single squeezed out by the cap
            kept.setdefault(single, counts[single])

        total = sum(kept.values())
        return {piece: math.log(c / total) for piece, c in kept.items()}

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _span_logprob(self, piece: str, span_graphemes: int) -> float:
        """Score a candidate piece covering `span_graphemes` clusters.

        A piece absent from the vocabulary must score NEG_INF, not a
        finite floor - otherwise probability mass still flows through
        pruned pieces and the next EM step resurrects every one of them,
        so pruning never converges to the target vocabulary size.

        The single exception is a *single* grapheme cluster never seen in
        training: that needs a finite (very bad) score so unknown input
        stays segmentable instead of breaking Viterbi outright.
        """
        known = self.piece_logprobs.get(piece)
        if known is not None:
            return known
        return _UNSEEN_LOGPROB if span_graphemes == 1 else NEG_INF

    def _viterbi(self, units: list[str]) -> list[str]:
        """Highest-probability segmentation of `units`."""
        n = len(units)
        if n == 0:
            return []

        best = [NEG_INF] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0

        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_piece_graphemes), end):
                if best[start] == NEG_INF:
                    continue
                span = self._span_logprob("".join(units[start:end]), end - start)
                if span == NEG_INF:
                    continue
                score = best[start] + span
                if score > best[end]:
                    best[end] = score
                    back[end] = start

        pieces = []
        position = n
        while position > 0:
            start = back[position]
            pieces.append("".join(units[start:position]))
            position = start
        return list(reversed(pieces))

    # ------------------------------------------------------------------
    # EM
    # ------------------------------------------------------------------
    def _forward_backward(self, units: list[str]) -> tuple[dict[str, float], float]:
        """Expected piece counts over ALL segmentations, plus the sentence
        log-likelihood. Computed in log space to avoid underflow."""
        n = len(units)
        if n == 0:
            return {}, 0.0

        alpha = [NEG_INF] * (n + 1)
        alpha[0] = 0.0
        for end in range(1, n + 1):
            terms = [
                alpha[start] + self._span_logprob("".join(units[start:end]), end - start)
                for start in range(max(0, end - self.max_piece_graphemes), end)
                if alpha[start] != NEG_INF
            ]
            alpha[end] = _logsumexp(terms)

        total = alpha[n]
        if total == NEG_INF:
            return {}, NEG_INF

        beta = [NEG_INF] * (n + 1)
        beta[n] = 0.0
        for start in range(n - 1, -1, -1):
            terms = [
                self._span_logprob("".join(units[start:end]), end - start) + beta[end]
                for end in range(start + 1, min(n, start + self.max_piece_graphemes) + 1)
                if beta[end] != NEG_INF
            ]
            beta[start] = _logsumexp(terms)

        counts: dict[str, float] = {}
        for start in range(n):
            for end in range(start + 1, min(n, start + self.max_piece_graphemes) + 1):
                piece = "".join(units[start:end])
                span = self._span_logprob(piece, end - start)
                if span == NEG_INF or alpha[start] == NEG_INF or beta[end] == NEG_INF:
                    continue
                counts[piece] = counts.get(piece, 0.0) + math.exp(
                    alpha[start] + span + beta[end] - total
                )
        return counts, total

    def _em_step(self, tokenized: list[list[str]]) -> float:
        """One EM iteration. Returns total corpus log-likelihood BEFORE
        the update, which must increase monotonically across iterations."""
        expected: dict[str, float] = {}
        log_likelihood = 0.0

        for units in tokenized:
            counts, total = self._forward_backward(units)
            if total == NEG_INF:
                continue
            log_likelihood += total
            for piece, count in counts.items():
                expected[piece] = expected.get(piece, 0.0) + count

        total_count = sum(expected.values())
        if total_count > 0:
            # Filter on the ratio, not the raw count: a piece can have a
            # tiny-but-positive expected count that underflows to exactly
            # 0.0 once divided, and log(0.0) raises.
            updated = {}
            for piece, count in expected.items():
                probability = count / total_count
                if probability > 0.0:
                    updated[piece] = math.log(probability)
            if updated:
                self.piece_logprobs = updated
        return log_likelihood

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: list[str],
        vocab_size: int = 300,
        seed_multiplier: int = 10,
        em_iterations: int = 2,
        prune_fraction: float = 0.2,
        max_rounds: int = 20,
    ) -> None:
        corpus = [t for t in corpus if t]
        tokenized = [grapheme_strings(t) for t in corpus]
        singles = {u for units in tokenized for u in units}

        self.piece_logprobs = self._candidate_pieces(corpus, vocab_size * seed_multiplier)

        for _ in range(max_rounds):
            for _ in range(em_iterations):
                self._em_step(tokenized)

            if len(self.piece_logprobs) <= vocab_size:
                break

            # Contribution to corpus likelihood; single graphemes are
            # protected so every input stays segmentable.
            expected: dict[str, float] = {}
            for units in tokenized:
                counts, total = self._forward_backward(units)
                if total == NEG_INF:
                    continue
                for piece, count in counts.items():
                    expected[piece] = expected.get(piece, 0.0) + count

            prunable = [p for p in self.piece_logprobs if p not in singles]
            prunable.sort(key=lambda p: expected.get(p, 0.0) * abs(self.piece_logprobs[p]))

            target_removals = max(1, int(len(prunable) * prune_fraction))
            allowed = max(0, len(self.piece_logprobs) - vocab_size)
            for piece in prunable[: min(target_removals, allowed)]:
                del self.piece_logprobs[piece]

            if not allowed:
                break

        self._em_step(tokenized)
        self.vocab = Vocabulary(sorted(self.piece_logprobs))

    def tokenize(self, text: str) -> list[str]:
        if not self.piece_logprobs:
            # Untrained: fall back to grapheme clusters rather than
            # silently returning nothing.
            return grapheme_strings(text)
        return self._viterbi(grapheme_strings(text))

    def segmentation_logprob(self, text: str) -> float:
        """Log-probability of this text under the fitted unigram model."""
        from ..unicode.grapheme import grapheme_strings as _graphemes

        return sum(
            self._span_logprob(piece, len(_graphemes(piece))) for piece in self.tokenize(text)
        )
