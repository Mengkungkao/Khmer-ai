from khmer_language.tokenizer import (
    BPETokenizer,
    CharacterTokenizer,
    GraphemeTokenizer,
    SAMPLE_CORPUS,
    compare,
    format_comparison,
)


def test_compare_runs_all_tokenizers_and_reports_stats():
    tokenizers = {
        "character": CharacterTokenizer(),
        "grapheme": GraphemeTokenizer(),
        "bpe": BPETokenizer(),
    }
    stats = compare(tokenizers, list(SAMPLE_CORPUS), vocab_size=80)
    assert [s.name for s in stats] == ["character", "grapheme", "bpe"]
    for s in stats:
        assert s.vocab_size > 0
        assert s.avg_sequence_length > 0
        assert s.compression_ratio > 0


def test_grapheme_tokenizer_compresses_at_least_as_well_as_character():
    tokenizers = {"character": CharacterTokenizer(), "grapheme": GraphemeTokenizer()}
    stats = compare(tokenizers, list(SAMPLE_CORPUS))
    by_name = {s.name: s for s in stats}
    assert by_name["grapheme"].avg_sequence_length <= by_name["character"].avg_sequence_length


def test_format_comparison_is_readable_text():
    tokenizers = {"character": CharacterTokenizer()}
    stats = compare(tokenizers, list(SAMPLE_CORPUS))
    output = format_comparison(stats)
    assert "character" in output
    assert "tokenizer" in output  # header
