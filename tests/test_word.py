from khmer_language.unicode.word import word_strings

ZWSP = chr(0x200B)


def test_splits_on_zwsp():
    text = f"ខ្ញុំ{ZWSP}ចូលចិត្ត{ZWSP}កម្ពុជា"
    assert word_strings(text) == ["ខ្ញុំ", "ចូលចិត្ត", "កម្ពុជា"]


def test_splits_on_whitespace():
    assert word_strings("កម្ពុជា ជាប្រទេស") == ["កម្ពុជា", "ជាប្រទេស"]


def test_splits_on_punctuation_without_space():
    khan = chr(0x17D4)
    assert word_strings(f"ខ្ញុំចង់ទៅ{khan}អ្នកទៅណា") == ["ខ្ញុំចង់ទៅ", "អ្នកទៅណា"]


def test_unsegmented_run_is_returned_whole():
    # Documented v1 limitation: no dictionary yet, so a boundary-free run
    # of Khmer text is one "word" even though it contains several.
    assert word_strings("កម្ពុជាជាប្រទេស") == ["កម្ពុជាជាប្រទេស"]


def test_empty_and_whitespace_only():
    assert word_strings("") == []
    assert word_strings("   ") == []
