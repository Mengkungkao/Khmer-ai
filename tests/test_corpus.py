import numpy as np
import pytest

from khmer_language.corpus import (
    Document,
    MinHasher,
    content_hash,
    deduplicate,
    identify,
    is_khmer,
    jaccard,
    read_jsonl,
    run_pipeline,
    score_document,
    shingles,
    to_sentences,
    write_jsonl,
)

KHMER = "កម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។ រាជធានីគឺទីក្រុងភ្នំពេញ។"


def _doc(id_: str, text: str, **kwargs) -> Document:
    base = dict(source="test", license="CC-BY-4.0")
    base.update(kwargs)
    return Document(id=id_, text=text, **base)


# --------------------------------------------------------------------------
# Document / JSONL
# --------------------------------------------------------------------------
def test_document_round_trips_through_jsonl(tmp_path):
    docs = [_doc("a", KHMER, domain="geography"), _doc("b", "ខ្ញុំចង់ទៅភ្នំពេញ។")]
    path = tmp_path / "corpus.jsonl"
    write_jsonl(docs, path)

    loaded = list(read_jsonl(path))
    assert [d.id for d in loaded] == ["a", "b"]
    assert loaded[0].text == KHMER
    assert loaded[0].domain == "geography"
    assert loaded[0].license == "CC-BY-4.0"


def test_document_from_dict_preserves_unknown_fields_in_metadata():
    doc = Document.from_dict(
        {"id": "x", "text": "ក", "source": "s", "license": "l", "scraped_at": "2026-01-01"}
    )
    assert doc.metadata["scraped_at"] == "2026-01-01"


# --------------------------------------------------------------------------
# Language ID
# --------------------------------------------------------------------------
def test_pure_khmer_scores_one():
    assert identify(KHMER).khmer_ratio == 1.0
    assert is_khmer(KHMER)


def test_pure_english_scores_zero():
    assert identify("Hello world").khmer_ratio == 0.0
    assert not is_khmer("Hello world")


def test_mixed_language_scores_in_between():
    score = identify("កម្ពុជា Cambodia")
    assert 0.0 < score.khmer_ratio < 1.0


def test_punctuation_and_digits_are_script_neutral():
    """ASCII digits and punctuation appear in good Khmer text, so they
    must not count against the Khmer ratio."""
    assert identify("កម្ពុជា, 2026! (ជា)").khmer_ratio == 1.0


def test_empty_text_is_not_khmer():
    assert identify("").khmer_ratio == 0.0


# --------------------------------------------------------------------------
# Quality
# --------------------------------------------------------------------------
def test_clean_khmer_scores_high():
    assert score_document(KHMER).score > 0.9


def test_non_khmer_text_scores_zero_despite_clean_components():
    """Regression test for a real design bug: under an arithmetic mean,
    English scored 0.71, because unicode/repetition/markup are all
    vacuously perfect when there is no Khmer to be malformed."""
    report = score_document("This is entirely English text with no Khmer at all.")
    assert report.components["script"] == 0.0
    assert report.components["unicode"] == 1.0  # vacuously clean
    assert report.components["markup"] == 1.0  # vacuously clean
    assert report.score == 0.0  # ...but the document is worthless as Khmer data


def test_repetitive_spam_is_rejected_despite_clean_components():
    """Second regression of the same class: one word repeated 80 times
    scored 0.605 under an arithmetic mean and passed a 0.6 filter, because
    script/unicode/markup were all 1.0. The geometric mean is dominated by
    the catastrophic repetition component instead."""
    report = score_document("កម្ពុជា" * 80)
    assert report.components["repetition"] < 0.05
    assert report.components["script"] == 1.0  # perfectly good Khmer characters
    assert report.score < 0.4  # ...but unusable as training data


def test_score_is_dominated_by_its_worst_component():
    """The defining property of the geometric mean here: a document is
    only as good as its weakest dimension."""
    report = score_document("កម្ពុជា" * 80)
    assert report.score < min(
        v for k, v in report.components.items() if k != "repetition"
    )


def test_partially_khmer_text_is_scored_proportionally():
    mixed = score_document("កម្ពុជា Cambodia is a country in Southeast Asia")
    assert 0.0 < mixed.score < score_document(KHMER).score


def test_repetitive_document_is_penalized():
    repetitive = score_document("កម្ពុជា" * 50)
    normal = score_document(KHMER)
    assert repetitive.components["repetition"] < normal.components["repetition"]


def test_html_markup_is_penalized():
    with_markup = score_document(f"<div class='x'>{KHMER}</div><br/>&nbsp;")
    assert with_markup.components["markup"] < 1.0


def test_structurally_invalid_khmer_lowers_unicode_component():
    assert score_document("ា" * 30).components["unicode"] < 1.0


def test_unavailable_components_are_declared_not_faked():
    """Grammar and readability must never be silently scored."""
    report = score_document(KHMER)
    assert "grammar" not in report.components
    assert "readability" not in report.components
    assert any("grammar" in u for u in report.unavailable)
    assert any("readability" in u for u in report.unavailable)


def test_score_always_lies_in_zero_to_one():
    for text in [KHMER, "", "abc", "កម្ពុជា" * 100, "<html></html>"]:
        assert 0.0 <= score_document(text).score <= 1.0


def test_source_score_is_included_when_supplied():
    without = score_document(KHMER).score
    with_low_source = score_document(KHMER, source_score=0.0).score
    assert with_low_source < without


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------
def test_content_hash_ignores_whitespace_and_zero_width_differences():
    a = "កម្ពុជា ជា"
    b = "កម្ពុជា   ជា"  # extra spaces
    c = "កម្ពុជា" + chr(0x200B) + " ជា"  # zero-width space
    assert content_hash(a) == content_hash(b) == content_hash(c)


def test_content_hash_differs_for_different_text():
    assert content_hash("កម្ពុជា") != content_hash("ភ្នំពេញ")


def test_shingles_are_grapheme_ngrams():
    assert shingles("កម្ពុជា", n=2) == {"កម្ពុ", "ម្ពុជា"}


def test_jaccard_bounds():
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_minhash_estimates_jaccard_similarity():
    """The core MinHash property: signature agreement approximates the
    true Jaccard similarity of the shingle sets."""
    a = KHMER
    b = KHMER + " ប្រទេសនេះមានប្រជាជនច្រើន។"

    hasher = MinHasher(num_hashes=256, seed=0)
    estimated = hasher.similarity(hasher.signature(a), hasher.signature(b))
    true = jaccard(shingles(a), shingles(b))
    assert abs(estimated - true) < 0.15


def test_minhash_signature_of_identical_text_matches_exactly():
    hasher = MinHasher(num_hashes=64, seed=0)
    assert np.array_equal(hasher.signature(KHMER), hasher.signature(KHMER))


def test_exact_duplicates_are_removed():
    docs = [_doc("a", KHMER), _doc("b", KHMER), _doc("c", "ភ្នំពេញជារាជធានី។")]
    result = deduplicate(docs)
    assert len(result.kept) == 2
    assert result.exact_duplicates == 1
    assert result.kept[0].id == "a"  # first occurrence kept


def test_near_duplicates_are_removed():
    """The real-world case exact hashing misses: a long article republished
    with a changed headline/date. Most of the text is shared, so MinHash
    similarity stays high even though the bytes differ."""
    body = (
        "ប្រទេសកម្ពុជាមានប្រវត្តិសាស្ត្រយូរលង់ណាស់។ "
        "អង្គរវត្តជាកេរ្តិ៍ដំណែលពិភពលោកដ៏ល្បីល្បាញ។ "
        "ភ្នំពេញជារាជធានីនៃប្រទេសកម្ពុជា។ "
        "ប្រជាជនខ្មែរនិយាយភាសាខ្មែរជាភាសាកំណើត។ "
    ) * 2
    original = "ព័ត៌មានថ្ងៃទី១។ " + body
    reposted = "ព័ត៌មានថ្ងៃទី២។ " + body  # same article, different date line

    result = deduplicate([_doc("a", original), _doc("b", reposted)], near_duplicate_threshold=0.8)
    assert result.exact_duplicates == 0  # bytes differ, so exact hashing misses it
    assert result.near_duplicates == 1
    assert len(result.kept) == 1


def test_substantially_edited_document_is_kept():
    """A genuinely different document must survive dedup: MinHash measured
    ~0.54 similarity for this pair, correctly below the threshold."""
    edited = KHMER.replace("អាស៊ីអាគ្នេយ៍", "អាស៊ី")
    result = deduplicate([_doc("a", KHMER), _doc("b", edited)], near_duplicate_threshold=0.8)
    assert result.near_duplicates == 0
    assert len(result.kept) == 2


def test_distinct_documents_are_not_deduplicated():
    docs = [_doc("a", KHMER), _doc("b", "ខ្ញុំចូលចិត្តញ៉ាំបាយនិងសម្លរម្ជូរនៅផ្ទះ។")]
    result = deduplicate(docs, near_duplicate_threshold=0.8)
    assert len(result.kept) == 2
    assert result.removed == 0


def test_threshold_of_one_skips_near_duplicate_detection():
    edited = KHMER.replace("អាស៊ីអាគ្នេយ៍", "អាស៊ី")
    result = deduplicate([_doc("a", KHMER), _doc("b", edited)], near_duplicate_threshold=1.0)
    assert result.near_duplicates == 0
    assert len(result.kept) == 2


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def test_pipeline_keeps_good_khmer_documents():
    long_khmer = KHMER + " ប្រទេសនេះមានវប្បធម៌យូរលង់ណាស់។"
    result = run_pipeline([_doc("a", long_khmer)])
    assert result.stats.output_documents == 1
    assert result.documents[0].quality is not None


def test_pipeline_rejects_unlicensed_documents():
    result = run_pipeline([_doc("a", KHMER, license="")])
    assert result.stats.missing_license == 1
    assert result.stats.output_documents == 0


def test_pipeline_can_be_told_not_to_require_a_license():
    long_khmer = KHMER + " ប្រទេសនេះមានវប្បធម៌យូរលង់ណាស់។"
    result = run_pipeline([_doc("a", long_khmer, license="")], require_license=False)
    assert result.stats.output_documents == 1


def test_pipeline_rejects_non_khmer():
    english = "This is a long English document with plenty of words in it, but no Khmer."
    result = run_pipeline([_doc("a", english)])
    assert result.stats.wrong_language == 1
    assert result.stats.output_documents == 0


def test_pipeline_rejects_too_short_documents():
    result = run_pipeline([_doc("a", "កម្ពុជា")])
    assert result.stats.too_short == 1


def test_pipeline_deduplicates():
    long_khmer = KHMER + " ប្រទេសនេះមានវប្បធម៌យូរលង់ណាស់។"
    result = run_pipeline([_doc("a", long_khmer), _doc("b", long_khmer)])
    assert result.stats.exact_duplicates == 1
    assert result.stats.output_documents == 1


def test_pipeline_stats_account_for_every_input_document():
    """Every dropped document must be attributed to a reason - otherwise a
    shrinking corpus is impossible to debug."""
    docs = [
        _doc("good", KHMER + " ប្រទេសនេះមានវប្បធម៌យូរលង់ណាស់។"),
        _doc("dupe", KHMER + " ប្រទេសនេះមានវប្បធម៌យូរលង់ណាស់។"),
        _doc("nolicense", KHMER, license=""),
        _doc("english", "This is a long English document with plenty of words but no Khmer."),
        _doc("short", "ក"),
    ]
    s = run_pipeline(docs).stats
    accounted = (
        s.missing_license
        + s.exact_duplicates
        + s.near_duplicates
        + s.wrong_language
        + s.too_short
        + s.low_quality
        + s.output_documents
    )
    assert accounted == s.input_documents == 5


def test_pipeline_normalizes_text():
    messy = "កម្ពុជាស្ថិតនៅអាស៊ីអាគ្នេយ៍។    រាជធានីគឺទីក្រុងភ្នំពេញ។   "
    result = run_pipeline([_doc("a", messy)])
    assert "    " not in result.documents[0].text
    assert not result.documents[0].text.endswith(" ")


def test_pipeline_stats_are_printable():
    output = str(run_pipeline([_doc("a", KHMER)]).stats)
    assert "input:" in output
    assert "output:" in output


def test_to_sentences_splits_documents_into_sentences():
    docs = [_doc("a", KHMER)]
    sentences = to_sentences(docs)
    assert len(sentences) == 2
    assert all(s.strip() for s in sentences)


def test_empty_corpus_does_not_crash():
    result = run_pipeline([])
    assert result.documents == []
    assert result.stats.input_documents == 0
