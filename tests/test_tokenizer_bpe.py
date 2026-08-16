import pytest

from khmer_language.tokenizer.bpe import BPETokenizer
from khmer_language.tokenizer.compare import SAMPLE_CORPUS
from khmer_language.unicode.grapheme import grapheme_strings


def test_bpe_with_no_room_for_merges_matches_grapheme_units():
    # vocab_size=1 is already smaller than the base alphabet, so no merge
    # can ever be added - this isolates and proves the "starts from
    # grapheme clusters, not code points" property from the module
    # docstring.
    tok = BPETokenizer()
    tok.train(list(SAMPLE_CORPUS), vocab_size=1)
    assert tok.merges == []
    for text in SAMPLE_CORPUS:
        assert tok.tokenize(text) == grapheme_strings(text)


def test_round_trip_is_exact_for_training_sentences():
    tok = BPETokenizer()
    tok.train(list(SAMPLE_CORPUS), vocab_size=120)
    for text in SAMPLE_CORPUS:
        assert tok.decode(tok.encode(text)) == text


def test_more_merges_never_increase_sequence_length():
    small = BPETokenizer()
    small.train(list(SAMPLE_CORPUS), vocab_size=1)
    big = BPETokenizer()
    big.train(list(SAMPLE_CORPUS), vocab_size=200)
    for text in SAMPLE_CORPUS:
        assert len(big.tokenize(text)) <= len(small.tokenize(text))


def test_merge_order_is_most_frequent_pair_first():
    tok = BPETokenizer()
    tok.train(["ababab"], vocab_size=100, min_frequency=1)
    assert tok.merges[0] == ("a", "b")
    assert tok.tokenize("ababab") == ["ababab"]


def test_warns_when_vocab_cannot_exceed_the_grapheme_alphabet():
    """Khmer has ~2,600 distinct grapheme clusters in real text, against
    roughly 100 characters for English. A vocab_size below that leaves no
    room for a single merge, and BPE silently becomes the grapheme
    tokenizer - measured on real Wikipedia, asking for 2,000 produced zero
    merges and output identical to GraphemeTokenizer."""
    tok = BPETokenizer()
    with pytest.warns(UserWarning, match="no merges can be learned"):
        tok.train(list(SAMPLE_CORPUS), vocab_size=10)
    assert tok.alphabet_exceeds_vocab
    assert tok.merges == []


def test_no_warning_when_vocab_leaves_room_for_merges():
    import warnings

    tok = BPETokenizer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        tok.train(list(SAMPLE_CORPUS), vocab_size=200)
    assert not tok.alphabet_exceeds_vocab
    assert tok.merges


def test_min_frequency_stops_rare_merges():
    tok = BPETokenizer()
    tok.train(["ab"], vocab_size=100, min_frequency=5)  # "ab" pair occurs once
    assert tok.merges == []
