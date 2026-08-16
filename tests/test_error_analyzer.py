from khmer_language.evaluation.error_analyzer import (
    FAIL,
    PASS,
    UNAVAILABLE,
    analyze_output,
    format_report,
)


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_valid_khmer_passes_every_implemented_check():
    report = analyze_output("កម្ពុជាជាប្រទេសនៅអាស៊ីអាគ្នេយ៍។")
    assert _check(report, "Unicode").status == PASS
    assert _check(report, "Khmer script").status == PASS
    assert _check(report, "Repetition").status == PASS
    assert report.passed


def test_structurally_invalid_khmer_fails_the_unicode_check():
    report = analyze_output("ា")  # orphan vowel sign, no base
    unicode_check = _check(report, "Unicode")
    assert unicode_check.status == FAIL
    assert "orphan-combining-mark" in unicode_check.detail
    assert not report.passed


def test_mostly_latin_output_fails_the_script_check():
    report = analyze_output("Hello world this is not Khmer at all")
    assert _check(report, "Khmer script").status == FAIL
    assert not report.passed


def test_degenerate_repetition_is_detected():
    """The classic undertrained-LM failure: emitting one token forever."""
    report = analyze_output("ក" * 20)
    repetition = _check(report, "Repetition")
    assert repetition.status == FAIL
    assert repetition.score == 20.0


def test_mild_repetition_is_allowed():
    report = analyze_output("កកកខគ")
    assert _check(report, "Repetition").status == PASS


def test_unimplemented_checks_are_reported_as_unavailable_with_a_reason():
    """These must never silently report PASS - a fabricated score would
    make the model look evaluated when it is not.

    Spelling used to be on this list. It became implementable once the
    Khmer lexicon existed, which is the intended direction of travel:
    checks graduate off this list by being built, never by being faked.
    """
    report = analyze_output("កម្ពុជា")
    for name in ("Grammar", "Meaning", "Naturalness"):
        check = _check(report, name)
        assert check.status == UNAVAILABLE
        assert check.detail  # must explain what is missing


def test_unavailable_checks_do_not_count_as_passes_in_coverage():
    """Coverage counts what is actually measured. It read 3/7 before the
    lexicon existed and 4/7 once spelling could be checked for real."""
    from khmer_language.lexicon import DEFAULT_LEXICON

    report = analyze_output("កម្ពុជា")
    expected = "4/7" if DEFAULT_LEXICON.exists() else "3/7"
    assert report.coverage == f"{expected} checks implemented"


def test_khmer_ratio_score_is_reported():
    report = analyze_output("កម្ពុជា")
    assert _check(report, "Khmer script").score == 1.0


def test_zero_width_word_separators_do_not_count_against_khmer_ratio():
    """ZWSP is the conventional Khmer word boundary (Khmer has no spaces),
    so it is everywhere in real Khmer text. Because "\\u200b".isspace() is
    False, a naive whitespace filter let it through and scored genuine
    Khmer as only 87% Khmer."""
    zwsp = chr(0x200B)
    text = f"ក្រុង{zwsp}ដូន{zwsp}ខ្មែរ"
    assert _check(analyze_output(text), "Khmer script").score == 1.0


def test_error_analyzer_and_language_id_agree_on_khmer_ratio():
    """Two modules computing the same quantity must not disagree."""
    from khmer_language.corpus.language_id import identify

    zwsp = chr(0x200B)
    for text in ("កម្ពុជា", f"ក្រុង{zwsp}ដូន", "កម្ពុជា Cambodia", "Hello"):
        analyzer_score = _check(analyze_output(text), "Khmer script").score
        assert analyzer_score == identify(text).khmer_ratio, f"disagreement on {text!r}"


def test_empty_text_does_not_crash():
    report = analyze_output("")
    assert _check(report, "Khmer script").score == 0.0


def test_format_report_shows_statuses_and_coverage():
    output = format_report(analyze_output("កម្ពុជា"))
    assert "Unicode:" in output
    assert "UNAVAILABLE" in output
    assert "checks implemented" in output
