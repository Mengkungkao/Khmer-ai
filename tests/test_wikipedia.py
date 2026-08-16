import bz2

import pytest

from khmer_language.corpus.wikipedia import (
    WIKIPEDIA_LICENSE,
    clean_wikitext,
    iter_pages,
    load_documents,
)

DUMP_TEMPLATE = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>កម្ពុជា</title>
    <ns>0</ns>
    <revision><text>ប្រទេស'''កម្ពុជា''' ស្ថិតនៅអាស៊ី។</text></revision>
  </page>
  <page>
    <title>Talk:កម្ពុជា</title>
    <ns>1</ns>
    <revision><text>discussion page</text></revision>
  </page>
  <page>
    <title>ខ្មែរ</title>
    <ns>0</ns>
    <revision><text>#REDIRECT [[កម្ពុជា]]</text></revision>
  </page>
  <page>
    <title>ភ្នំពេញ</title>
    <ns>0</ns>
    <revision><text>ភ្នំពេញជា[[រាជធានី]]។</text></revision>
  </page>
</mediawiki>
"""


@pytest.fixture
def dump(tmp_path):
    path = tmp_path / "dump.xml.bz2"
    path.write_bytes(bz2.compress(DUMP_TEMPLATE.encode("utf-8")))
    return path


# --------------------------------------------------------------------------
# Wikitext cleaning
# --------------------------------------------------------------------------
def test_removes_nested_templates():
    """Templates nest, so this cannot be done with a regular expression -
    the reason the cleaner scans brace depth instead."""
    assert clean_wikitext("A{{outer|x{{inner|y}}z}}B") == "AB"


def test_unbalanced_template_degrades_gracefully():
    """Real dumps contain broken markup; it must not raise."""
    assert clean_wikitext("text {{unclosed") == "text"


def test_wikilink_keeps_display_text():
    assert clean_wikitext("[[កម្ពុជា|ស្រុកខ្មែរ]]") == "ស្រុកខ្មែរ"
    assert clean_wikitext("[[ភ្នំពេញ]]") == "ភ្នំពេញ"


def test_file_and_category_links_are_dropped():
    assert clean_wikitext("[[File:a.jpg|thumb|caption]]សួស្តី") == "សួស្តី"
    assert clean_wikitext("[[ឯកសារ:a.jpg|caption]]សួស្តី") == "សួស្តី"


def test_references_and_comments_are_removed():
    assert "note" not in clean_wikitext("text<ref>note</ref>more")
    assert "hidden" not in clean_wikitext("text<!--hidden-->more")


def test_headings_keep_their_text():
    assert clean_wikitext("== ប្រវត្តិ ==") == "ប្រវត្តិ"


def test_bold_and_italic_markers_are_stripped():
    assert clean_wikitext("'''ខ្មែរ''' and ''italic''") == "ខ្មែរ and italic"


def test_external_links_keep_the_label():
    assert clean_wikitext("[http://example.com Example] tail") == "Example tail"


def test_tables_are_removed():
    assert clean_wikitext("{|class=x\n|cell\n|}after") == "after"


def test_magic_words_are_removed():
    """__NOTOC__ and friends are rendering directives, not prose - they
    appear verbatim on pages like the main page."""
    assert clean_wikitext("__NOTOC__ការតាំង__NOEDITSECTION__") == "ការតាំង"


# --------------------------------------------------------------------------
# Dump parsing
# --------------------------------------------------------------------------
def test_only_article_namespace_pages_are_yielded(dump):
    titles = [p.title for p in iter_pages(dump)]
    assert "Talk:កម្ពុជា" not in titles


def test_redirects_are_skipped(dump):
    titles = [p.title for p in iter_pages(dump)]
    assert "ខ្មែរ" not in titles  # the redirect page
    assert titles == ["កម្ពុជា", "ភ្នំពេញ"]


def test_load_documents_carries_wikipedia_provenance(dump):
    docs = list(load_documents(dump))
    assert len(docs) == 2
    assert all(d.license == WIKIPEDIA_LICENSE for d in docs)
    assert all(d.source == "wikipedia:km" for d in docs)
    assert docs[0].metadata["title"] == "កម្ពុជា"


def test_load_documents_returns_cleaned_text(dump):
    docs = list(load_documents(dump))
    assert "'''" not in docs[0].text
    assert docs[0].text == "ប្រទេសកម្ពុជា ស្ថិតនៅអាស៊ី។"
    assert docs[1].text == "ភ្នំពេញជារាជធានី។"


def test_limit_caps_the_number_of_documents(dump):
    assert len(list(load_documents(dump, limit=1))) == 1


def test_documents_pass_the_license_check(dump):
    """Wikipedia's licence is the reason it was chosen as the source: every
    document must survive the pipeline's licence gate."""
    from khmer_language.corpus import run_pipeline

    docs = list(load_documents(dump))
    assert run_pipeline(docs, min_graphemes=1).stats.missing_license == 0
