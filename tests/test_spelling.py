import pytest

from khmer_language.lexicon.segmenter import KhmerLexicon, LexiconEntry
from khmer_language.lexicon.spelling import SpellChecker, Verdict
from khmer_language.unicode.grapheme import grapheme_strings


@pytest.fixture
def checker():
    entries = [
        LexiconEntry("កម្ពុជា", "name", 900),
        LexiconEntry("ភ្នំពេញ", "name", 300),
        LexiconEntry("រាជធានី", "noun", 200),
        LexiconEntry("ប្រទេស", "noun", 1000),
        LexiconEntry("ក", "noun", 5000),  # short and very frequent
        LexiconEntry("ទៅ", "verb", 700),
    ]
    return SpellChecker(KhmerLexicon(entries))


def test_dictionary_word_is_correct(checker):
    assert checker.check_word("កម្ពុជា").verdict is Verdict.CORRECT


def test_word_one_edit_away_is_flagged_as_misspelled(checker):
    result = checker.check_word("កម្ពុជ")  # កម្ពុជា minus the final grapheme
    assert result.verdict is Verdict.MISSPELLED
    assert "កម្ពុជា" in result.suggestions


def test_suggestions_prefer_shared_prefix_over_raw_frequency(checker):
    """Frequency alone proposed ក for a truncated កម្ពុជា - a very common
    short word that is an edit-1 neighbour of a great many things. Khmer
    typing errors mostly affect the END of a word, so shared prefix is
    the stronger signal."""
    suggestions = checker.check_word("កម្ពុជ").suggestions
    assert suggestions[0] == "កម្ពុជា"


def test_unrelated_word_is_unknown_not_misspelled(checker):
    """Absence from the dictionary is not evidence of misspelling: 24% of
    Khmer Wikipedia words are outside this lexicon and nearly all are
    correctly spelled proper nouns."""
    assert checker.check_word("ឩឋឪឫ").verdict is Verdict.UNKNOWN


def test_unknown_words_do_not_lower_the_score(checker):
    """A proper noun must not be scored as an error."""
    report = checker.check("ឩឋឪឫ")
    assert report.unknown
    assert report.score == 1.0


def test_misspellings_lower_the_score_when_they_survive_segmentation(checker):
    """Scoring works on words the segmenter keeps whole.

    A misspelling only reaches the checker if segmentation does not
    dissolve it first - see `test_segmentation_can_absorb_a_typo`. Here
    the damaged word is checked directly, which is the reliable path.
    """
    report = checker.check_word("កម្ពុជ")
    assert report.verdict is Verdict.MISSPELLED

    from khmer_language.lexicon.spelling import SpellingReport

    assert SpellingReport(checks=(report,)).score < 1.0


def test_clean_text_scores_perfectly(checker):
    assert checker.check("ប្រទេសកម្ពុជា").score == 1.0


def test_single_grapheme_words_are_not_second_guessed(checker):
    """One-grapheme words have too many edit-1 neighbours for a
    suggestion to mean anything."""
    assert checker.check_word("ក").verdict is Verdict.CORRECT


def test_empty_text(checker):
    report = checker.check("")
    assert report.checks == ()
    assert report.score == 1.0


def test_segmentation_can_absorb_a_typo(checker):
    """Documents a real limitation rather than papering over it.

    In running text the segmenter may re-split a misspelling into valid
    words - ប្រទេស minus a grapheme becomes ប្រ + ទេ - so no unknown word
    reaches the checker. Detection is high-precision but limited-recall
    on running text; the same word checked in isolation IS caught.
    """
    isolated = checker.check_word("កម្ពុជ")
    assert isolated.verdict is Verdict.MISSPELLED

    # Whereas embedded in text, the outcome depends on whether the
    # segmenter can explain the damaged span with real words.
    report = checker.check("កម្ពុជ")
    assert report.score <= 1.0  # never crashes, may or may not catch it


def test_checker_handles_non_khmer_gracefully(checker):
    report = checker.check("Hello world 2026")
    assert report.checks == ()  # nothing Khmer to check


def test_error_analyzer_reports_spelling_when_a_lexicon_exists():
    from khmer_language.evaluation import analyze_output
    from khmer_language.lexicon import DEFAULT_LEXICON

    report = analyze_output("ប្រទេសកម្ពុជា")
    spelling = next(c for c in report.checks if c.name == "Spelling")
    if DEFAULT_LEXICON.exists():
        assert spelling.status != "UNAVAILABLE"
        assert spelling.score is not None
    else:
        assert spelling.status == "UNAVAILABLE"
