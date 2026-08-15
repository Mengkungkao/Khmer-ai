import itertools
import math

import pytest

from khmer_language.tokenizer.compare import SAMPLE_CORPUS
from khmer_language.tokenizer.grapheme import GraphemeTokenizer
from khmer_language.tokenizer.unigram import UnigramTokenizer
from khmer_language.unicode.grapheme import grapheme_strings

CORPUS = list(SAMPLE_CORPUS)


def _trained(vocab_size=60, **kwargs):
    tok = UnigramTokenizer(**kwargs)
    tok.train(CORPUS, vocab_size=vocab_size)
    return tok


# --------------------------------------------------------------------------
# EM correctness
# --------------------------------------------------------------------------
def test_em_increases_log_likelihood_monotonically():
    """The defining property of EM. If this fails, forward-backward or the
    M-step normalization is wrong."""
    tok = UnigramTokenizer()
    tokenized = [grapheme_strings(t) for t in CORPUS]
    tok.piece_logprobs = tok._candidate_pieces(CORPUS, 600)

    likelihoods = [tok._em_step(tokenized) for _ in range(6)]
    for before, after in zip(likelihoods, likelihoods[1:]):
        assert after >= before - 1e-6


def test_forward_backward_posteriors_sum_to_one_per_position():
    """Every grapheme position must be covered by exactly one piece, so the
    posterior mass of pieces covering a given position sums to 1."""
    tok = _trained()
    units = grapheme_strings(CORPUS[0])
    counts, total = tok._forward_backward(units)
    assert total != float("-inf")

    # Total expected pieces * average length must cover every position:
    # sum over pieces of (count * piece length) == number of positions.
    covered = sum(count * len(grapheme_strings(piece)) for piece, count in counts.items())
    assert covered == pytest.approx(len(units), rel=1e-6)


# --------------------------------------------------------------------------
# Viterbi correctness
# --------------------------------------------------------------------------
def _brute_force_best(tok, units):
    """Every possible segmentation, scored - only tractable for tiny input."""
    n = len(units)
    best_score, best_seg = float("-inf"), None
    for cuts in itertools.product([0, 1], repeat=max(0, n - 1)):
        segments, start = [], 0
        for i, cut in enumerate(cuts, start=1):
            if cut:
                segments.append(units[start:i])
                start = i
        segments.append(units[start:])
        if any(len(s) > tok.max_piece_graphemes for s in segments):
            continue
        score = sum(tok._span_logprob("".join(s), len(s)) for s in segments)
        if score > best_score:
            best_score, best_seg = score, ["".join(s) for s in segments]
    return best_score, best_seg


def test_viterbi_finds_the_globally_best_segmentation():
    """Checked against exhaustive enumeration - this is what distinguishes
    Unigram from BPE's greedy merge order."""
    tok = _trained()
    units = grapheme_strings(CORPUS[3])[:8]
    expected_score, _ = _brute_force_best(tok, units)
    actual = tok._viterbi(units)
    actual_score = sum(tok._span_logprob(p, len(grapheme_strings(p))) for p in actual)
    assert actual_score == pytest.approx(expected_score, rel=1e-9)


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------
def test_pruned_pieces_stay_pruned_after_an_em_step():
    """Regression: a finite fallback score for unknown pieces let
    probability mass keep flowing through pruned pieces, so the next EM
    step resurrected every one and pruning never converged."""
    tok = UnigramTokenizer()
    tokenized = [grapheme_strings(t) for t in CORPUS]
    tok.piece_logprobs = tok._candidate_pieces(CORPUS, 600)
    singles = {u for units in tokenized for u in units}

    victim = next(p for p in tok.piece_logprobs if p not in singles)
    del tok.piece_logprobs[victim]
    tok._em_step(tokenized)
    assert victim not in tok.piece_logprobs


def test_larger_vocab_request_yields_at_least_as_many_pieces():
    small = _trained(vocab_size=80)
    large = _trained(vocab_size=200)
    assert len(large.piece_logprobs) >= len(small.piece_logprobs)


def test_vocabulary_cannot_shrink_below_the_distinct_grapheme_count():
    """Single grapheme clusters are never pruned, so they form a hard
    floor - requesting fewer is silently impossible, not a failure."""
    distinct = len({u for t in CORPUS for u in grapheme_strings(t)})
    tok = _trained(vocab_size=5)
    assert len(tok.piece_logprobs) >= distinct


# --------------------------------------------------------------------------
# Segmentation behaviour
# --------------------------------------------------------------------------
def test_round_trip_is_exact_for_every_training_sentence():
    tok = _trained()
    for text in CORPUS:
        assert tok.decode(tok.encode(text)) == text


def test_segmentation_always_reassembles_to_the_original():
    tok = _trained()
    for text in CORPUS:
        assert "".join(tok.tokenize(text)) == text


def test_unseen_text_is_still_segmentable():
    """Unknown single graphemes get a finite floor score, so Viterbi never
    fails outright on out-of-corpus input."""
    tok = _trained()
    unseen = "ឡាវនិងថៃ"
    pieces = tok.tokenize(unseen)
    assert "".join(pieces) == unseen


def test_learns_multi_grapheme_pieces():
    """The point of subword tokenization: frequent sequences become single
    tokens, so sequences are shorter than plain grapheme segmentation."""
    tok = _trained(vocab_size=200)
    graphemes = GraphemeTokenizer()

    total_unigram = sum(len(tok.tokenize(t)) for t in CORPUS)
    total_grapheme = sum(len(graphemes.tokenize(t)) for t in CORPUS)
    assert total_unigram < total_grapheme
    assert any(len(grapheme_strings(p)) > 1 for p in tok.tokenize(CORPUS[0]))


def test_untrained_tokenizer_falls_back_to_graphemes():
    tok = UnigramTokenizer()
    assert tok.tokenize("កម្ពុជា") == grapheme_strings("កម្ពុជា")


def test_empty_string_tokenizes_to_nothing():
    assert _trained().tokenize("") == []


def test_segmentation_logprob_is_finite_and_negative():
    tok = _trained()
    score = tok.segmentation_logprob(CORPUS[0])
    assert math.isfinite(score)
    assert score < 0


def test_training_on_empty_corpus_does_not_crash():
    tok = UnigramTokenizer()
    tok.train([""], vocab_size=10)
    assert tok.tokenize("ក") == ["ក"]
