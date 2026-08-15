from khmer_language.unicode.normalizer import has_zero_width, normalize, strip_zero_width


def test_normalize_is_idempotent():
    text = "កម្ពុជា  ជាប្រទេស"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice


def test_normalize_collapses_ascii_whitespace_and_trims():
    assert normalize("  កម្ពុជា   ជា   ប្រទេស  ") == "កម្ពុជា ជា ប្រទេស"


def test_normalize_collapses_excess_blank_lines():
    assert normalize("មួយ\n\n\n\nពីរ") == "មួយ\n\nពីរ"


def test_strip_and_detect_zero_width():
    text = "ក" + chr(0x200B) + "ខ"
    assert has_zero_width(text)
    assert not has_zero_width(strip_zero_width(text))
    assert strip_zero_width(text) == "កខ"
