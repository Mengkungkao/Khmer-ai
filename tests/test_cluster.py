from khmer_language.unicode.cluster import analyze_cluster
from khmer_language.unicode.grapheme import segment_graphemes


def test_analyze_cluster_with_one_subscript():
    graphemes = segment_graphemes("កម្ពុជា")
    middle = analyze_cluster(graphemes[1])  # ម្ពុ
    assert middle.base == "ម"
    assert middle.subscripts == ("ព",)
    assert middle.vowel == "ុ"
    assert middle.base_series == "o"


def test_analyze_cluster_with_two_stacked_subscripts():
    text = "ស" + chr(0x17D2) + "ត" + chr(0x17D2) + "រ" + "ី"
    grapheme = segment_graphemes(text)[0]
    cluster = analyze_cluster(grapheme)
    assert cluster.base == "ស"
    assert cluster.subscripts == ("ត", "រ")
    assert cluster.vowel == "ី"


def test_analyze_cluster_plain_consonant_no_vowel():
    grapheme = segment_graphemes("កម្ពុជា")[0]  # ក alone
    cluster = analyze_cluster(grapheme)
    assert cluster.base == "ក"
    assert cluster.subscripts == ()
    assert cluster.vowel is None
    assert cluster.base_series == "a"
