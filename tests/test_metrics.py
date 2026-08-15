import numpy as np
import pytest

from khmer_language.evaluation.metrics import (
    character_error_rate,
    count_validation_errors,
    edit_distance,
    exact_match,
    grapheme_error_rate,
    perplexity,
    unicode_validity_rate,
)
from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.unicode.grapheme import grapheme_strings


# --------------------------------------------------------------------------
# edit distance
# --------------------------------------------------------------------------
def test_edit_distance_of_identical_sequences_is_zero():
    assert edit_distance(list("abc"), list("abc")) == 0


def test_edit_distance_counts_substitution_insertion_deletion():
    assert edit_distance(list("abc"), list("abd")) == 1  # substitute
    assert edit_distance(list("abc"), list("abcd")) == 1  # insert
    assert edit_distance(list("abc"), list("ab")) == 1  # delete


def test_edit_distance_is_symmetric():
    a, b = list("kitten"), list("sitting")
    assert edit_distance(a, b) == edit_distance(b, a) == 3


def test_edit_distance_with_empty_sequence_is_the_other_length():
    assert edit_distance([], list("abc")) == 3
    assert edit_distance(list("abc"), []) == 3


# --------------------------------------------------------------------------
# CER / GER
# --------------------------------------------------------------------------
def test_error_rates_are_zero_for_identical_text():
    text = "កម្ពុជា"
    assert character_error_rate(text, text) == 0.0
    assert grapheme_error_rate(text, text) == 0.0


def test_grapheme_error_rate_treats_a_khmer_cluster_as_one_unit():
    """The reason GER exists: dropping the single cluster ម្ពុ is ONE
    error to a Khmer reader, but it spans 4 codepoints (ម ្ ព ុ), so CER
    scores the same mistake at nearly double the rate."""
    reference = "កម្ពុជា"  # 7 codepoints, 3 grapheme clusters
    hypothesis = "កជា"  # removed the whole ម្ពុ cluster

    assert grapheme_strings(reference) == ["ក", "ម្ពុ", "ជា"]
    assert len(reference) == 7

    # 1 grapheme deleted out of 3
    assert grapheme_error_rate(reference, hypothesis) == pytest.approx(1 / 3)
    # ...but 4 codepoints deleted out of 7
    assert character_error_rate(reference, hypothesis) == pytest.approx(4 / 7)
    assert character_error_rate(reference, hypothesis) > grapheme_error_rate(reference, hypothesis)


def test_error_rate_of_empty_reference():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "ក") == 1.0
    assert grapheme_error_rate("", "") == 0.0


def test_error_rate_can_exceed_one_for_much_longer_hypothesis():
    assert character_error_rate("ក", "កកកកក") == pytest.approx(4.0)


def test_exact_match():
    assert exact_match("កម្ពុជា", "កម្ពុជា")
    assert not exact_match("កម្ពុជា", "កម្ពុជ")


# --------------------------------------------------------------------------
# validity metrics
# --------------------------------------------------------------------------
def test_unicode_validity_rate():
    valid, invalid = "កម្ពុជា", "ា"  # orphan vowel sign
    assert unicode_validity_rate([valid, valid]) == 1.0
    assert unicode_validity_rate([valid, invalid]) == 0.5
    assert unicode_validity_rate([]) == 0.0


def test_count_validation_errors_tallies_by_code():
    counts = count_validation_errors(["ា", "ា", "កម្ពុជា"])
    assert counts["orphan-combining-mark"] == 2


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------
def test_untrained_perplexity_is_near_vocab_size():
    """A random model is ~uniform over the vocab, so its perplexity should
    be close to the vocab size. Far from it means the model or the metric
    is wrong."""
    vocab = 20
    model = KhmerGPT(GPTConfig(vocab_size=vocab, dim=8, num_layers=1, num_heads=2, max_seq_len=16), seed=0)
    data = np.random.default_rng(0).integers(0, vocab, size=200)
    result = perplexity(model, data, seq_len=8)
    assert result.perplexity == pytest.approx(vocab, rel=0.4)
    assert result.num_tokens > 0


def test_perplexity_drops_after_training():
    from khmer_language.training import train

    data = np.tile(np.arange(5), 60)
    model = KhmerGPT(GPTConfig(vocab_size=5, dim=16, num_layers=1, num_heads=2, max_seq_len=16), seed=0)
    before = perplexity(model, data, seq_len=8).perplexity
    train(model, data, steps=100, batch_size=4, seq_len=8, lr=5e-3, seed=0)
    after = perplexity(model, data, seq_len=8).perplexity

    assert after < before
    assert after < 1.5  # near-perfect on a fully predictable sequence


def test_perplexity_rejects_too_short_data():
    model = KhmerGPT(GPTConfig(vocab_size=5, dim=8, num_layers=1, num_heads=2, max_seq_len=16), seed=0)
    with pytest.raises(ValueError, match="at least"):
        perplexity(model, np.arange(3), seq_len=8)


def test_perplexity_is_deterministic():
    """Non-overlapping consecutive windows, so no seed dependence."""
    model = KhmerGPT(GPTConfig(vocab_size=7, dim=8, num_layers=1, num_heads=2, max_seq_len=16), seed=0)
    data = np.arange(7).repeat(10) % 7
    assert perplexity(model, data, seq_len=8).perplexity == perplexity(model, data, seq_len=8).perplexity
