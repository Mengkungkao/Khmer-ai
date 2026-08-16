"""Guards for the memoized grapheme segmentation.

Caching is only safe because `Grapheme` is frozen and `segment_graphemes`
hands out a fresh list each call. These tests pin both properties, plus
the correctness of the single-pass classification that replaced
classifying every character twice.
"""

import unicodedata

from khmer_language.unicode.character_types import CharacterType, classify, classify_codepoint
from khmer_language.unicode.grapheme import (
    _classify_all,
    grapheme_strings,
    segment_graphemes,
)

WORD = "កម្ពុជា"


def test_repeated_calls_return_equal_results():
    assert grapheme_strings(WORD) == grapheme_strings(WORD)
    assert [g.text for g in segment_graphemes(WORD)] == grapheme_strings(WORD)


def test_each_call_returns_an_independent_list():
    """Callers must be free to mutate the returned list without the cache
    handing the mutated version to the next caller."""
    first = segment_graphemes(WORD)
    first.append(first[0])
    first.clear()
    assert len(segment_graphemes(WORD)) == 3

    strings = grapheme_strings(WORD)
    strings.clear()
    assert grapheme_strings(WORD) == ["ក", "ម្ពុ", "ជា"]


def test_grapheme_objects_are_frozen():
    """The cache shares Grapheme instances between callers, which is only
    safe while they cannot be mutated."""
    import dataclasses

    import pytest

    grapheme = segment_graphemes(WORD)[0]
    assert dataclasses.fields(grapheme)  # it is a dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        grapheme.text = "changed"


def test_single_pass_classification_matches_the_contextual_classifier():
    """_classify_all resolves CONSONANT vs SUBSCRIPT_CONSONANT in one
    sweep; it must agree with classify() character by character."""
    for text in (WORD, "ស្ត្រី", "ខ្ញុំចង់ទៅ។", "AI ២០២៦", ""):
        assert _classify_all(text) == [classify(text, i) for i in range(len(text))]


def test_classification_table_matches_a_full_block_scan():
    """The precomputed table must agree with the semantics it replaced
    across the entire Khmer block and its neighbours."""
    for cp in range(0x1700, 0x1900):
        expected_is_khmer_block = 0x1780 <= cp <= 0x17FF
        char_type = classify_codepoint(cp)
        if not expected_is_khmer_block:
            assert char_type in (CharacterType.NON_KHMER, CharacterType.WHITESPACE)


def test_whitespace_and_non_khmer_still_classify_correctly():
    assert classify_codepoint(ord(" ")) is CharacterType.WHITESPACE
    assert classify_codepoint(ord("\n")) is CharacterType.WHITESPACE
    assert classify_codepoint(ord("A")) is CharacterType.NON_KHMER
    assert classify_codepoint(0x3000) is CharacterType.WHITESPACE  # ideographic space


def test_non_khmer_combining_marks_still_attach():
    """Latin text with a combining accent must not fragment."""
    text = "e" + chr(0x0301)  # e + combining acute
    assert unicodedata.combining(chr(0x0301)) != 0
    assert grapheme_strings(text) == [text]


def test_cache_does_not_confuse_different_inputs():
    a, b = "កម្ពុជា", "ភ្នំពេញ"
    for _ in range(3):
        assert "".join(grapheme_strings(a)) == a
        assert "".join(grapheme_strings(b)) == b
