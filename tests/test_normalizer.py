from khmer_language.unicode.normalizer import (
    collapse_redundant_zwsp,
    has_zero_width,
    normalize,
    strip_zero_width,
)


def test_normalize_is_idempotent():
    text = "កម្ពុជា  ជាប្រទេស"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice


def test_normalize_collapses_ascii_whitespace_and_trims():
    assert normalize("  កម្ពុជា   ជា   ប្រទេស  ") == "កម្ពុជា ជា ប្រទេស"


def test_normalize_collapses_excess_blank_lines():
    assert normalize("មួយ\n\n\n\nពីរ") == "មួយ\n\nពីរ"


ZWSP = chr(0x200B)


def test_meaningful_zwsp_word_boundary_is_preserved():
    """ZWSP between two Khmer words IS the word boundary - Khmer does not
    space its words - so it must survive normalization."""
    assert normalize(f"ក{ZWSP}ខ") == f"ក{ZWSP}ខ"


def test_repeated_zwsp_collapses_to_one_not_zero():
    """A run of ZWSP still marks one real boundary. An earlier version
    deleted the whole run, silently destroying the word boundary."""
    assert collapse_redundant_zwsp(f"ក{ZWSP}{ZWSP}{ZWSP}ខ") == f"ក{ZWSP}ខ"


def test_zwsp_next_to_real_whitespace_is_dropped():
    """The space already marks the boundary, so the ZWSP adds nothing -
    and every ZWSP costs a token, being the most frequent token in the
    corpus at 8.2%."""
    assert collapse_redundant_zwsp(f"ក{ZWSP} ខ") == "ក ខ"
    assert collapse_redundant_zwsp(f"ក {ZWSP}ខ") == "ក ខ"
    assert collapse_redundant_zwsp(f"ក{ZWSP}\nខ") == "ក\nខ"


def test_zwsp_run_beside_whitespace_is_fully_dropped():
    assert collapse_redundant_zwsp(f"ក{ZWSP}{ZWSP} ខ") == "ក ខ"


def test_collapsing_zwsp_is_idempotent():
    text = f"ក{ZWSP}ខ{ZWSP} គ{ZWSP}{ZWSP}ឃ"
    once = collapse_redundant_zwsp(text)
    assert collapse_redundant_zwsp(once) == once


def test_text_without_zwsp_is_unchanged():
    assert collapse_redundant_zwsp("កម្ពុជា ជាប្រទេស") == "កម្ពុជា ជាប្រទេស"


def test_strip_and_detect_zero_width():
    text = "ក" + chr(0x200B) + "ខ"
    assert has_zero_width(text)
    assert not has_zero_width(strip_zero_width(text))
    assert strip_zero_width(text) == "កខ"
