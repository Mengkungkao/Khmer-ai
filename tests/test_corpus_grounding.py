import pytest

from khmer_language.evaluation.corpus_grounding import CorpusGrounding

REFERENCE = [
    "ភ្នំពេញជារាជធានីនៃប្រទេសកម្ពុជា។",
    "ប្រទេសកម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។",
    "អង្គរវត្តជាកេរ្តិ៍ដំណែលពិភពលោក។",
]


@pytest.fixture
def grounding():
    return CorpusGrounding(REFERENCE)


def test_text_from_the_reference_scores_perfectly(grounding):
    for text in REFERENCE:
        assert grounding.score(text, 3).ratio == 1.0


def test_gibberish_scores_near_zero(grounding):
    """The property machine translation could not provide: nonsense must
    be distinguishable from real Khmer."""
    assert grounding.score("ខគជធឆលបមណតទផពហអ", 3).ratio < 0.2


def test_real_khmer_outscores_gibberish(grounding):
    real = grounding.score(REFERENCE[0], 3).ratio
    fake = grounding.score("ខគជធឆលបមណតទផពហអ", 3).ratio
    assert real > fake


def test_longer_ngrams_are_stricter(grounding):
    """A partially-real string should score lower as n grows, since longer
    spans are harder to match by accident."""
    mixed = "ភ្នំពេញខគជធឆលបម"
    scores = grounding.score_all(mixed)
    assert scores[2].ratio >= scores[4].ratio


def test_index_sizes_are_reported(grounding):
    sizes = grounding.sizes()
    assert set(sizes) == {2, 3, 4}
    assert all(v > 0 for v in sizes.values())


def test_text_shorter_than_n_yields_no_ngrams(grounding):
    score = grounding.score("ក", 4)
    assert score.total == 0
    assert score.ratio == 0.0


def test_empty_text(grounding):
    assert grounding.score("", 3).ratio == 0.0


def test_unbuilt_order_is_rejected(grounding):
    with pytest.raises(ValueError, match="not built for n=7"):
        grounding.score("កម្ពុជា", 7)


def test_custom_orders():
    grounding = CorpusGrounding(REFERENCE, orders=(2, 5))
    assert set(grounding.sizes()) == {2, 5}
    assert grounding.score(REFERENCE[0], 5).ratio == 1.0


def test_summary_is_readable(grounding):
    summary = grounding.summary(REFERENCE[0])
    assert "2-gram" in summary and "100%" in summary


def test_score_counts_are_consistent(grounding):
    score = grounding.score(REFERENCE[0], 3)
    assert score.attested <= score.total
    assert score.ratio == score.attested / score.total


def test_empty_reference_corpus_scores_everything_zero():
    grounding = CorpusGrounding([])
    assert grounding.score("កម្ពុជា", 2).ratio == 0.0
