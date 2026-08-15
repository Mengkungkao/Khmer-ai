from khmer_language.unicode.transliterator import transliterate


def test_kampuchea():
    # ក(a-series, no vowel -> inherent "â") + ម្ពុ(o-series base, "ŭ") +
    # ជា(o-series base, "éa") -> matches the classic romanization
    # "Kampuchea" almost exactly, which is a nice independent sanity check.
    assert transliterate("កម្ពុជា") == "kâmpŭchéa"


def test_transliterate_is_deterministic():
    text = "សួស្តី"
    assert transliterate(text) == transliterate(text)


def test_transliterate_digits():
    assert transliterate("១២៣") == "123"


def test_transliterate_passes_through_non_khmer():
    assert transliterate("Hello") == "Hello"


def test_toandakhiat_silences_inherent_vowel():
    # ន + TOANDAKHIAT: consonant with cancelled inherent vowel -> no
    # trailing "ô" the way a bare ន alone would produce.
    plain = transliterate("ន")
    silenced = transliterate("ន" + chr(0x17CD))
    assert plain == "nô"
    assert silenced == "n"
