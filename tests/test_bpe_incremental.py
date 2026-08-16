"""The incremental pair-counting optimization must not change behaviour.

Recounting every pair after each merge is O(merges x corpus) and does not
finish on real Khmer. The incremental version updates only the sequences
affected by a merge. These tests pin that it selects the same merges in
the same order, since a faster tokenizer that segments differently would
silently invalidate every trained checkpoint.
"""

from collections import Counter

import pytest

from khmer_language.tokenizer.bpe import BPETokenizer, _apply_merge
from khmer_language.tokenizer.compare import SAMPLE_CORPUS
from khmer_language.unicode.grapheme import grapheme_strings

CORPUS = list(SAMPLE_CORPUS)


def _reference_merges(corpus, vocab_size, min_frequency=2):
    """Recount-everything BPE, kept as an oracle for the optimized version.

    Uses the same explicit lexicographic tie-break as the implementation.
    Without that both would fall back on dict insertion order, which
    differs between them - so this would be comparing two arbitrary
    choices rather than checking the optimization is equivalent.
    """
    seq_freqs = Counter(tuple(grapheme_strings(t)) for t in corpus if t)
    alphabet = sorted({s for seq in seq_freqs for s in seq})
    vocab_len = len(alphabet) + 4  # four special tokens
    merges = []

    while vocab_len < vocab_size:
        counts: Counter = Counter()
        for seq, freq in seq_freqs.items():
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] += freq
        if not counts:
            break
        best = max(counts, key=lambda pair: (counts[pair], pair))
        best_count = counts[best]
        if best_count < min_frequency:
            break
        merged = best[0] + best[1]
        merges.append(best)
        vocab_len += 1
        seq_freqs = Counter(
            {_apply_merge(seq, best, merged): freq for seq, freq in seq_freqs.items()}
        )
    return merges


@pytest.mark.parametrize("vocab_size", [100, 150, 250])
def test_incremental_selects_the_same_merges_as_recounting(vocab_size):
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=vocab_size)
    assert tokenizer.merges == _reference_merges(CORPUS, vocab_size)


def test_merge_order_is_preserved():
    """Order matters: tokenize() replays merges in sequence, so a
    different order produces a different segmentation."""
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=200)
    reference = _reference_merges(CORPUS, 200)
    assert tokenizer.merges[:10] == reference[:10]


def test_round_trip_still_exact():
    tokenizer = BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=200)
    for text in CORPUS:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_min_frequency_still_respected():
    tokenizer = BPETokenizer()
    tokenizer.train(["ab"], vocab_size=100, min_frequency=5)
    assert tokenizer.merges == []


def test_repeated_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(CORPUS, vocab_size=200)
    b.train(CORPUS, vocab_size=200)
    assert a.merges == b.merges


def test_merges_never_exceed_the_vocab_budget():
    for vocab_size in (100, 200, 400):
        tokenizer = BPETokenizer()
        tokenizer.train(CORPUS, vocab_size=vocab_size)
        assert len(tokenizer.vocab) <= vocab_size
