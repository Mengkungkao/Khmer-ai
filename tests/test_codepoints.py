from khmer_language.unicode import codepoints as cp_db


def test_consonant_count():
    assert len(cp_db.CONSONANTS) == 35
    assert sum(1 for c in cp_db.CONSONANTS if not c.obsolete) == 33


def test_independent_vowel_count():
    assert len(cp_db.INDEPENDENT_VOWELS) == 17


def test_dependent_vowel_count():
    assert len(cp_db.DEPENDENT_VOWELS) == 16


def test_digit_count_and_order():
    assert len(cp_db.DIGITS) == 10
    assert [d.value for d in cp_db.DIGITS] == list(range(10))
    assert cp_db.DIGITS[0].char == "០"
    assert cp_db.DIGITS[5].char == "៥"


def test_anchor_glyphs_derive_correctly_from_codepoints():
    # Guards against transcription mistakes: glyphs are always derived via
    # chr(codepoint), never hand-typed, but pin a couple of well-known
    # anchors so a data-table typo (wrong codepoint) still gets caught.
    assert chr(0x1780) == "ក"  # KA
    assert cp_db.CONSONANTS_BY_CODEPOINT[0x1780].name == "KA"
    assert cp_db.CONSONANTS_BY_CODEPOINT[0x17A2].name == "QA"
    assert cp_db.COENG_CODEPOINT == 0x17D2


def test_known_register_exceptions():
    # NNO and LA are the two well-documented exceptions to the otherwise
    # regular a,a,o,o,o per-row consonant register pattern.
    nno = cp_db.CONSONANTS_BY_CODEPOINT[0x178E]
    la = cp_db.CONSONANTS_BY_CODEPOINT[0x17A1]
    assert nno.series == "a"
    assert la.series == "o"


def test_export_json_round_trips(tmp_path):
    import json

    out = tmp_path / "characters.json"
    cp_db.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["consonants"]) == 35
    assert data["consonants"][0]["char"] == chr(0x1780)
