from khmer_language import analyze, format_analysis
from khmer_language.unicode.character_types import CharacterType


def test_analyze_kampuchea_end_to_end():
    result = analyze("កម្ពុជា")

    assert [c.type for c in result.characters] == [
        CharacterType.CONSONANT,
        CharacterType.CONSONANT,
        CharacterType.COENG,
        CharacterType.SUBSCRIPT_CONSONANT,
        CharacterType.DEPENDENT_VOWEL,
        CharacterType.CONSONANT,
        CharacterType.DEPENDENT_VOWEL,
    ]
    assert [g.text for g in result.graphemes] == ["ក", "ម្ពុ", "ជា"]
    assert [s.text for s in result.syllables] == ["ក", "ម្ពុ", "ជា"]
    assert [w.text for w in result.words] == ["កម្ពុជា"]
    assert [s.text for s in result.sentences] == ["កម្ពុជា"]
    assert result.transliteration == "kâmpŭchéa"
    assert result.is_valid
    assert result.issues == ()


def test_format_analysis_is_readable_text():
    output = format_analysis(analyze("កម្ពុជា"))
    assert "Graphemes (3)" in output
    assert "Validity: PASS" in output


def test_analyze_flags_invalid_text():
    result = analyze("ា")
    assert not result.is_valid
    assert any(issue.code == "orphan-combining-mark" for issue in result.issues)
