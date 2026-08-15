from khmer_language.unicode.syllable import syllable_strings


def test_syllables_match_graphemes_for_kampuchea():
    assert syllable_strings("កម្ពុជា") == ["ក", "ម្ពុ", "ជា"]
