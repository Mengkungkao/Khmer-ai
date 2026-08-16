from khmer_language.grammar import (
    ALL_FUNCTION_WORDS,
    WordClass,
    analyze_sentence,
    analyze_text,
    by_class,
    format_analysis,
    lookup,
)


# --------------------------------------------------------------------------
# Function word database
# --------------------------------------------------------------------------
def test_lookup_returns_grammatical_role():
    assert lookup("មិន").word_class is WordClass.NEGATION
    assert lookup("កំពុង").word_class is WordClass.ASPECT
    assert lookup("តើ").word_class is WordClass.QUESTION
    assert lookup("ខ្ញុំ").word_class is WordClass.PRONOUN


def test_unknown_word_returns_none():
    assert lookup("កម្ពុជា") is None  # a content word, not a function word


def test_every_entry_has_a_gloss():
    assert all(w.gloss for w in ALL_FUNCTION_WORDS)


def test_classes_are_populated():
    for word_class in (WordClass.NEGATION, WordClass.ASPECT, WordClass.MODAL,
                       WordClass.QUESTION, WordClass.PRONOUN):
        assert by_class(word_class)


# --------------------------------------------------------------------------
# Negation - a circumfix, not a single word
# --------------------------------------------------------------------------
def test_detects_negation():
    assert analyze_sentence("ខ្ញុំមិនទៅទេ។").is_negated


def test_clause_final_particle_alone_is_not_negation():
    """ទេ also forms polar questions, so negation requires the pre-verbal
    marker as well - otherwise every question reads as negated."""
    assert not analyze_sentence("អ្នកទៅទេ។").is_negated


def test_colloquial_negator_recognized():
    assert analyze_sentence("ខ្ញុំឥតដឹង។").is_negated


# --------------------------------------------------------------------------
# The substring problem - Khmer has no spaces
# --------------------------------------------------------------------------
def test_particle_inside_a_content_word_is_not_matched():
    """Regression: ទេ occurs inside ប្រទេស ("country"), so a plain
    substring search reported the negation/question particle in any
    sentence mentioning a country. ទេ is only that particle
    clause-finally, which rules the false positive out."""
    analysis = analyze_sentence("ប្រទេសកម្ពុជាស្ថិតនៅអាស៊ី។")
    assert not any(m.word.word == "ទេ" for m in analysis.matches)
    assert not analysis.is_negated


def test_clause_final_particle_is_still_matched():
    analysis = analyze_sentence("ខ្ញុំមិនដឹងទេ។")
    assert any(m.word.word == "ទេ" for m in analysis.matches)


def test_longer_question_word_wins_over_its_substring():
    analysis = analyze_sentence("ហេតុអ្វីបានជាគេទៅ?")
    words = [m.word.word for m in analysis.matches]
    assert "ហេតុអ្វី" in words
    assert "អ្វី" not in words


# --------------------------------------------------------------------------
# Tense and aspect - carried by words, since verbs never inflect
# --------------------------------------------------------------------------
def test_future_marker():
    assert analyze_sentence("ខ្ញុំនឹងទៅ។").tense.startswith("future")


def test_past_marker():
    assert analyze_sentence("ខ្ញុំបានទៅ។").tense == "past"


def test_unmarked_tense():
    assert analyze_sentence("ខ្ញុំទៅផ្សារ។").tense == "unmarked"


def test_continuous_aspect():
    assert analyze_sentence("គេកំពុងហែល។").aspect == "continuous"


def test_perfective_aspect():
    assert analyze_sentence("គេទៅហើយ។").aspect == "perfective"


def test_ambiguous_marker_is_flagged_not_guessed():
    """នឹង marks future before a verb but is the preposition "with" after
    one. "I do not agree WITH this" was being reported as future tense;
    resolving it needs a parser, so it is reported as ambiguous."""
    analysis = analyze_sentence("ខ្ញុំមិនយល់ស្របនឹងការនេះទេ។")
    assert analysis.ambiguous
    assert analysis.tense.endswith("?")


# --------------------------------------------------------------------------
# Sentence type - Khmer marks questions lexically, not by word order
# --------------------------------------------------------------------------
def test_question_opener_marks_a_question():
    assert analyze_sentence("តើអ្នកទៅណា?").is_question


def test_question_word_marks_a_question():
    assert analyze_sentence("គេទៅណា").is_question


def test_plain_statement_is_not_a_question():
    assert not analyze_sentence("ខ្ញុំទៅផ្សារ។").is_question


def test_modals_are_listed():
    assert "អាច" in analyze_sentence("ខ្ញុំអាចទៅបាន។").modals


# --------------------------------------------------------------------------
# Whole-text analysis and reporting
# --------------------------------------------------------------------------
def test_analyze_text_splits_into_sentences():
    assert len(analyze_text("ខ្ញុំទៅ។ អ្នកមកទេ។")) == 2


def test_confidence_is_lower_for_short_markers():
    """Short markers are far likelier to be spurious substrings in a
    script without spaces."""
    short = analyze_sentence("ខ្ញុំជាគេ។")
    long = analyze_sentence("ហេតុអ្វីបានជាប្រសិនបើគេទៅ?")
    assert short.confidence <= long.confidence


def test_text_with_no_function_words_is_fully_confident():
    assert analyze_sentence("កម្ពុជា").confidence == 1.0


def test_format_analysis_is_readable():
    output = format_analysis(analyze_sentence("តើអ្នកនឹងទៅទេ?"))
    assert "type:" in output and "tense:" in output


def test_empty_text_does_not_crash():
    analysis = analyze_sentence("")
    assert analysis.matches == ()
    assert not analysis.is_negated
