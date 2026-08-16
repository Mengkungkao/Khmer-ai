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

        while len(self.vocab) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for seq, freq in seq_freqs.items():
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += freq

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            self.vocab.add(merged)
            self.merges.append(best_pair)
            seq_freqs = Counter(
                {_apply_merge(seq, best_pair, merged): freq for seq, freq in seq_freqs.items()}
            )

    def tokenize(self, text: str) -> list[str]:
        symbols = tuple(grapheme_strings(text))
        for pair in self.merges:
            if len(symbols) < 2:
                break
            symbols = _apply_merge(symbols, pair, pair[0] + pair[1])
        return list(symbols)
