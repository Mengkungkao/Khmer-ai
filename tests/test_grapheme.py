from khmer_language.unicode.character_types import ZWNJ, ZWSP
from khmer_language.unicode.grapheme import grapheme_strings, segment_graphemes


def test_kampuchea_clusters_into_three_graphemes():
    assert grapheme_strings("កម្ពុជា") == ["ក", "ម្ពុ", "ជា"]


def test_stacked_subscripts_stay_in_one_cluster():
    # ស្ត្រី (srey-ish): SA + COENG+TA + COENG+RO + II, all one cluster.
    text = "ស" + chr(0x17D2) + "ត" + chr(0x17D2) + "រ" + "ី"
    assert grapheme_strings(text) == [text]


def test_zwsp_is_its_own_cluster_and_breaks_neighbors():
    text = "ក" + chr(ZWSP) + "ខ"
    assert grapheme_strings(text) == ["ក", chr(ZWSP), "ខ"]


def test_zwnj_stays_attached_to_current_cluster():
    text = "ក" + chr(0x17D2) + chr(ZWNJ) + "រ"
    clusters = grapheme_strings(text)
    assert clusters == [text]


def test_empty_string():
    assert segment_graphemes("") == []


def test_plain_ascii_is_one_grapheme_per_character():
    assert grapheme_strings("AI") == ["A", "I"]


def test_digits_do_not_merge():
    assert grapheme_strings("១២៣") == ["១", "២", "៣"]


def test_grapheme_offsets_cover_the_whole_string():
    text = "កម្ពុជា"
    clusters = segment_graphemes(text)
    assert clusters[0].start == 0
    assert clusters[-1].end == len(text)
    for a, b in zip(clusters, clusters[1:]):
        assert a.end == b.start
