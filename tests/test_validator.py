from khmer_language.unicode.validator import is_valid, validate

COENG = chr(0x17D2)


def test_valid_word_has_no_issues():
    assert validate("កម្ពុជា") == []
    assert is_valid("កម្ពុជា")


def test_orphan_vowel_sign_at_start():
    issues = validate("ា")
    assert any(i.code == "orphan-combining-mark" and i.severity == "error" for i in issues)
    assert not is_valid("ា")


def test_coeng_at_end_of_text():
    issues = validate("ក" + COENG)
    assert len(issues) == 1
    assert issues[0].code == "coeng-no-consonant"
    assert issues[0].severity == "error"
    assert not is_valid("ក" + COENG)


def test_double_coeng_is_flagged_once():
    text = "ក" + COENG + COENG + "រ"
    issues = validate(text)
    assert [i.code for i in issues] == ["coeng-no-consonant"]


def test_unassigned_codepoint_in_khmer_block():
    text = chr(0x17DE)
    issues = validate(text)
    assert any(i.code == "unassigned-codepoint" for i in issues)
    assert not is_valid(text)


def test_obsolete_consonant_is_warning_not_error():
    issues = validate("ឝ")
    assert [(i.code, i.severity) for i in issues] == [("obsolete-consonant", "warning")]
    assert is_valid("ឝ")  # warnings don't affect validity


def test_legacy_independent_vowel_is_warning():
    issues = validate(chr(0x17A3))
    assert [(i.code, i.severity) for i in issues] == [("legacy-independent-vowel", "warning")]
    assert is_valid(chr(0x17A3))


def test_multiple_vowel_signs_warning():
    issues = validate("ក" + "ា" + "ិ")
    assert any(i.code == "multiple-vowel-signs" and i.severity == "warning" for i in issues)
    assert is_valid("ក" + "ា" + "ិ")
