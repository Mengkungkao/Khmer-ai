from khmer_language.unicode.character_types import (
    CharacterType,
    ZWJ,
    ZWNJ,
    ZWSP,
    classify,
    classify_codepoint,
    get_consonant_series,
)


def test_kampuchea_matches_readme_example():
    # README.md section 4's worked example:
    # ក CONSONANT, ម CONSONANT, ្ SUBSCRIPT_MARK, ព SUBSCRIPT_CONSONANT,
    # ុ VOWEL, ជ CONSONANT, ា VOWEL
    text = "កម្ពុជា"
    expected = [
        CharacterType.CONSONANT,  # ក
        CharacterType.CONSONANT,  # ម
        CharacterType.COENG,  # ្
        CharacterType.SUBSCRIPT_CONSONANT,  # ព
        CharacterType.DEPENDENT_VOWEL,  # ុ
        CharacterType.CONSONANT,  # ជ
        CharacterType.DEPENDENT_VOWEL,  # ា
    ]
    actual = [classify(text, i) for i in range(len(text))]
    assert actual == expected


def test_context_free_classification_cannot_see_subscript():
    assert classify_codepoint(0x1796) == CharacterType.CONSONANT  # ព alone


def test_zero_width_characters():
    for cp in (ZWSP, ZWNJ, ZWJ):
        assert classify_codepoint(cp) == CharacterType.ZERO_WIDTH


def test_whitespace_and_non_khmer():
    assert classify_codepoint(ord(" ")) == CharacterType.WHITESPACE
    assert classify_codepoint(ord("A")) == CharacterType.NON_KHMER


def test_khmer_digit():
    assert classify_codepoint(0x17E5) == CharacterType.DIGIT  # ៥


def test_consonant_series_lookup():
    assert get_consonant_series("ក", 0) == "a"
    assert get_consonant_series("គ", 0) == "o"
    assert get_consonant_series("ណ", 0) == "a"  # documented exception
    assert get_consonant_series("ឡ", 0) == "o"  # documented exception


def test_zwnj_between_coeng_and_consonant_still_counts_as_subscript():
    # COENG + ZWNJ + consonant: ZWNJ forces an "unjoined" rendering but the
    # consonant is still linguistically a subscript.
    text = "ក" + chr(0x17D2) + chr(ZWNJ) + "រ"
    assert classify(text, 3) == CharacterType.SUBSCRIPT_CONSONANT
