import pytest

from khmer_language.lexicon.segmenter import (
    KhmerLexicon,
    LexiconEntry,
    WordSegmenter,
    _is_khmer_word,
)


@pytest.fixture
def segmenter():
    entries = [
        LexiconEntry("ប្រទេស", "noun", 1000),
        LexiconEntry("កម្ពុជា", "name", 900),
        LexiconEntry("ខ្ញុំ", "pron", 800),
        LexiconEntry("ចង់", "verb", 400),
        LexiconEntry("ទៅ", "verb", 700),
        LexiconEntry("ភ្នំពេញ", "name", 300),
        LexiconEntry("នៅ", "verb", 600),
        LexiconEntry("ជា", "prep", 500),
        LexiconEntry("រាជធានី", "noun", 200),
        LexiconEntry("ទី", "noun", 50),
        LexiconEntry("ក្រុង", "noun", 60),
    ]
    return WordSegmenter(KhmerLexicon(entries))


def test_segments_a_spaceless_run_into_words(segmenter):
    """The capability the hint-based segmenter could not provide: Khmer
    has no spaces, so this run was previously returned whole."""
    assert segmenter.segment_strings("ខ្ញុំចង់ទៅភ្នំពេញ") == ["ខ្ញុំ", "ចង់", "ទៅ", "ភ្នំពេញ"]


def test_segmentation_reassembles_to_the_original(segmenter):
    for text in ("ប្រទេសកម្ពុជា", "ខ្ញុំចង់ទៅភ្នំពេញ", "រាជធានីជាទីក្រុង"):
        assert "".join(segmenter.segment_strings(text)) == text


def test_parts_of_speech_are_attached(segmenter):
    words = segmenter.segment("ខ្ញុំចង់ទៅ")
    assert [w.pos for w in words] == ["pron", "verb", "verb"]


def test_frequency_decides_between_valid_splits(segmenter):
    """A dictionary says which strings are words; only frequency says
    which reading is likely. ទីក្រុង could split as ទី + ក្រុង, and both
    are real words, so the choice must come from counts."""
    entries = [
        LexiconEntry("ទីក្រុង", "noun", 5000),  # far more frequent as one word
        LexiconEntry("ទី", "noun", 10),
        LexiconEntry("ក្រុង", "noun", 10),
    ]
    frequent = WordSegmenter(KhmerLexicon(entries))
    assert frequent.segment_strings("ទីក្រុង") == ["ទីក្រុង"]


def test_unknown_text_still_segments(segmenter):
    """Names and loanwords absent from the dictionary must degrade to
    characters rather than making the text unsegmentable."""
    text = "ខ្ញុំzzz"
    assert "".join(segmenter.segment_strings(text)) == text


def test_unknown_words_are_flagged(segmenter):
    words = segmenter.segment("ខ្ញុំ" + "ឬ")
    assert any(not w.in_lexicon for w in words)


def test_viterbi_beats_greedy_longest_match(segmenter):
    """Greedy longest-match commits early and cannot reconsider; Viterbi
    optimizes the whole split. Here taking the longest first word would
    strand the remainder."""
    entries = [
        LexiconEntry("ការ", "noun", 100),
        LexiconEntry("ការងារ", "noun", 5),
        LexiconEntry("ងារ", "noun", 1),
        LexiconEntry("ល្អ", "adj", 500),
    ]
    seg = WordSegmenter(KhmerLexicon(entries))
    assert "".join(seg.segment_strings("ការល្អ")) == "ការល្អ"


def test_empty_text(segmenter):
    assert segmenter.segment("") == []


# --------------------------------------------------------------------------
# Coverage metric
# --------------------------------------------------------------------------
def test_coverage_is_full_for_known_text(segmenter):
    assert segmenter.coverage("ខ្ញុំចង់ទៅភ្នំពេញ") == 1.0


def test_coverage_ignores_spaces_and_latin(segmenter):
    """Regression: counting whitespace and embedded English as
    out-of-vocabulary scored a Khmer technical article at 65% purely
    because it contained spaces and the term "EV"."""
    with_latin = segmenter.coverage("ខ្ញុំ ចង់ ទៅ EV")
    without = segmenter.coverage("ខ្ញុំចង់ទៅ")
    assert with_latin == without == 1.0


def test_unknown_words_excludes_non_khmer(segmenter):
    unknown = segmenter.unknown_words("ខ្ញុំ EV ចង់")
    assert all(_is_khmer_word(w) for w in unknown)


def test_is_khmer_word():
    assert _is_khmer_word("កម្ពុជា")
    assert not _is_khmer_word("EV")
    assert not _is_khmer_word("  ")
    assert not _is_khmer_word("2026")


def test_lexicon_membership(segmenter):
    assert "កម្ពុជា" in segmenter.lexicon
    assert "ឥតមាន" not in segmenter.lexicon


def test_smoothing_gives_unseen_dictionary_words_usable_probability():
    """A word in the dictionary but absent from the corpus must still be
    selectable, otherwise the lexicon is silently reduced to whatever the
    corpus happened to contain."""
    lexicon = KhmerLexicon([LexiconEntry("ក", "noun", 0), LexiconEntry("ខ", "noun", 100)])
    assert lexicon.logprob("ក") > -25.0


def test_missing_lexicon_file_reports_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_lexicon"):
        KhmerLexicon.load(tmp_path / "absent.jsonl")
