"""Khmer Byte-Pair Encoding tokenizer (Tokenizer Lab, Version D).

Standard BPE (Sennrich et al. 2016) - repeatedly merge the most frequent
adjacent symbol pair - but "modified for Khmer" as README.md section 10
asks: the starting symbols are Khmer grapheme clusters
(`khmer_language.unicode.grapheme`), not raw Unicode code points or
bytes. Starting from graphemes rather than code points guarantees every
merge result is a concatenation of whole grapheme clusters, so a learned
subword token can never end mid-cluster (e.g. splitting a COENG off from
its subscript consonant) the way naive byte-level BPE could.

Unlike textbook BPE (English GPT-2 style), pair statistics are NOT
gathered per whitespace-split word: Khmer text mostly has no whitespace
between words in the first place (see `word.py`), so imposing that split
would be assuming exactly the thing this project doesn't have yet. Each
corpus entry (typically one line/sentence) is instead treated as a
single symbol sequence, whitespace graphemes included. This also keeps
`decode(encode(text)) == text` exactly, the same guarantee the
character/grapheme/syllable tokenizers give.
"""

from __future__ import annotations

from collections import Counter

from ..unicode.grapheme import grapheme_strings
from .base import BaseTokenizer

Symbols = tuple[str, ...]


def _apply_merge(symbols: Symbols, pair: tuple[str, str], merged: str) -> Symbols:
    left, right = pair
    result: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == left and symbols[i + 1] == right:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


class BPETokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.merges: list[tuple[str, str]] = []
        self.alphabet_exceeds_vocab = False

    def train(self, corpus: list[str], vocab_size: int = 1000, min_frequency: int = 2) -> None:
        seq_freqs: Counter[Symbols] = Counter(
            tuple(grapheme_strings(text)) for text in corpus if text
        )

        alphabet = sorted({symbol for seq in seq_freqs for symbol in seq})
        self.vocab = type(self.vocab)(alphabet)
        self.merges = []

        # Khmer's grapheme inventory is enormous next to an alphabet:
        # ~2,600 distinct clusters in real text, against roughly 100
        # characters for English. If vocab_size does not exceed that
        # inventory there is no room for a single merge, and BPE silently
        # degrades into the grapheme tokenizer - identical output, with
        # nothing to indicate the setting was meaningless.
        if vocab_size <= len(self.vocab):
            self.alphabet_exceeds_vocab = True
            import warnings

            warnings.warn(
                f"vocab_size={vocab_size} is not larger than the {len(self.vocab)} base "
                "grapheme clusters in this corpus, so no merges can be learned and this "
                "behaves exactly like GraphemeTokenizer. Khmer needs a substantially "
                "larger vocabulary than English for subword merging to begin.",
                stacklevel=2,
            )
        else:
            self.alphabet_exceeds_vocab = False

        # Incremental pair counting. The obvious implementation recounts
        # every pair in the whole corpus after each merge, which is
        # O(merges x corpus) and does not finish on real data - training
        # to vocab 4,000 on 60 Wikipedia articles ran over 12 minutes
        # without completing a single configuration.
        #
        # Merging a pair only changes the sequences that actually contain
        # it, so instead: keep a running pair count plus an index from
        # pair -> the sequences containing it, and on each merge touch
        # only those sequences, subtracting the pairs they lose and adding
        # the ones they gain. Same merges, same order, vastly less work.
        sequences: list[list[str]] = [list(seq) for seq in seq_freqs]
        frequencies: list[int] = [seq_freqs[seq] for seq in seq_freqs]

        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_locations: dict[tuple[str, str], set[int]] = {}

        def index_sequence(i: int, sign: int) -> None:
            symbols, freq = sequences[i], frequencies[i]
            for a, b in zip(symbols, symbols[1:]):
                pair = (a, b)
                pair_counts[pair] += sign * freq
                if sign > 0:
                    pair_locations.setdefault(pair, set()).add(i)

        for i in range(len(sequences)):
            index_sequence(i, +1)

        while len(self.vocab) < vocab_size:
            # Ties are the common case, not an edge case: on the first
            # iteration of the sample corpus, two distinct pairs share the
            # top count. Both the original and the incremental version
            # resolved them by dict insertion order, which is an artifact
            # of how the counts were accumulated rather than a decision -
            # and the two orders differ, so the same corpus produced
            # different tokenizers.
            #
            # Ties are therefore broken lexicographically on the pair.
            # The choice is arbitrary; being explicit and independent of
            # dict ordering is what matters, so training is reproducible.
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            best_count = pair_counts[best_pair]

            if best_count < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            self.vocab.add(merged)
            self.merges.append(best_pair)

            affected = pair_locations.pop(best_pair, set())
            for i in affected:
                index_sequence(i, -1)  # remove this sequence's old pairs
                sequences[i] = list(_apply_merge(tuple(sequences[i]), best_pair, merged))
                index_sequence(i, +1)  # and add its new ones

            # Counts can reach zero; drop them so the scan above stays small.
            for pair in [p for p, c in pair_counts.items() if c <= 0]:
                del pair_counts[pair]
                pair_locations.pop(pair, None)

    def tokenize(self, text: str) -> list[str]:
        symbols = tuple(grapheme_strings(text))
        for pair in self.merges:
            if len(symbols) < 2:
                break
            symbols = _apply_merge(symbols, pair, pair[0] + pair[1])
        return list(symbols)
