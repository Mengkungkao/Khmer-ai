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
    make the model look evaluated when it is not."""
    report = analyze_output("កម្ពុជា")
    for name in ("Spelling", "Grammar", "Meaning", "Naturalness"):
        check = _check(report, name)
        assert check.status == UNAVAILABLE
        assert check.detail  # must explain what is missing


def test_unavailable_checks_do_not_count_as_passes_in_coverage():
    report = analyze_output("កម្ពុជា")
    assert report.coverage == "3/7 checks implemented"


def test_khmer_ratio_score_is_reported():
    report = analyze_output("កម្ពុជា")
    assert _check(report, "Khmer script").score == 1.0


def test_empty_text_does_not_crash():
    report = analyze_output("")
    assert _check(report, "Khmer script").score == 0.0


def test_format_report_shows_statuses_and_coverage():
    output = format_report(analyze_output("កម្ពុជា"))
    assert "Unicode:" in output
    assert "UNAVAILABLE" in output
    assert "3/7 checks implemented" in output
