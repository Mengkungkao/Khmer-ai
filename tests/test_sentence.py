from khmer_language.unicode.sentence import sentence_strings

KHAN = chr(0x17D4)


def test_splits_on_khan():
    text = f"ខ្ញុំចង់ទៅ{KHAN} អ្នកទៅណា{KHAN}"
    assert sentence_strings(text) == ["ខ្ញុំចង់ទៅ" + KHAN, "អ្នកទៅណា" + KHAN]


def test_no_terminator_is_still_one_sentence():
    assert sentence_strings("ខ្ញុំចង់ទៅ") == ["ខ្ញុំចង់ទៅ"]


def test_empty_text_has_no_sentences():
    assert sentence_strings("") == []
    assert sentence_strings("   ") == []


def test_ascii_terminators_also_split():
    assert sentence_strings("Hello! How are you?") == ["Hello!", "How are you?"]


def test_repeated_terminators_do_not_create_empty_sentences():
    assert sentence_strings(f"មួយ{KHAN}{KHAN} ពីរ") == [f"មួយ{KHAN}{KHAN}", "ពីរ"]
