"""Khmer Wikipedia dump ingestion (README section 6, "Collection").

This is the one collection source the project has chosen, and it is here
rather than as a general scraper for a specific reason: Wikipedia dumps
come with an unambiguous licence (CC BY-SA 4.0), a stable citable
source, and no robots.txt or terms-of-service question. Every document
produced carries that provenance, satisfying the licence check the
pipeline enforces.

Dumps are read as a **stream** - bz2 decompression feeding
`iterparse`, with each element released after use - so a multi-gigabyte
dump never has to fit in memory. Khmer's is far smaller than that, but
the same code then works unchanged for larger wikis.

Wikitext cleaning is necessarily heuristic. MediaWiki markup is not a
regular language: templates nest, so `{{a{{b}}c}}` cannot be stripped
with a regular expression, and this module scans brace depth instead.
What survives is a best-effort plain-text rendering, not a faithful
parse. Documents that come out mangled are caught downstream by the
quality filters rather than being assumed clean here.
"""

from __future__ import annotations

import bz2
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from .document import Document

WIKIPEDIA_LICENSE = "CC-BY-SA-4.0"
KHMER_WIKIPEDIA_DUMP_URL = (
    "https://dumps.wikimedia.org/kmwiki/latest/kmwiki-latest-pages-articles.xml.bz2"
)

_REDIRECT_RE = re.compile(r"^\s*#(REDIRECT|ប្តូរទីតាំង)", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
_LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# MediaWiki "magic words" (__NOTOC__, __NOEDITSECTION__, ...) are rendering
# directives, not prose; they show up verbatim on pages like the main page.
_MAGIC_WORD_RE = re.compile(r"__[A-Z_]+__")

# Namespaced link prefixes whose contents are captions/metadata, not prose.
_NON_PROSE_LINK_PREFIXES = (
    "file:",
    "image:",
    "category:",
    "media:",
    "ឯកសារ:",  # "File" in Khmer
    "រូបភាព:",  # "Image" in Khmer
    "ចំណាត់ថ្នាក់ក្រុម:",  # "Category" in Khmer
)


def _strip_nested(text: str, open_token: str, close_token: str) -> str:
    """Remove balanced nested spans, which regex cannot do.

    Used for `{{templates}}` and `{|tables|}`. Unbalanced markup (common
    in real dumps) degrades to dropping the remainder of the span rather
    than raising.
    """
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(open_token, i):
            depth += 1
            i += len(open_token)
        elif text.startswith(close_token, i):
            depth = max(0, depth - 1)
            i += len(close_token)
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _clean_links(text: str) -> str:
    """`[[target|display]]` -> `display`, dropping file/category links."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("[[", i):
            end = text.find("]]", i)
            if end == -1:
                out.append(text[i:])
                break
            inner = text[i + 2 : end]
            lowered = inner.lower()
            if not any(lowered.startswith(p) for p in _NON_PROSE_LINK_PREFIXES):
                # Keep the display text (after the last pipe), else the target.
                out.append(inner.split("|")[-1])
            i = end + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def clean_wikitext(text: str) -> str:
    """Best-effort conversion of MediaWiki markup to plain text."""
    text = _COMMENT_RE.sub(" ", text)
    text = _REF_RE.sub(" ", text)
    text = _strip_nested(text, "{{", "}}")
    text = _strip_nested(text, "{|", "|}")
    text = _clean_links(text)
    text = _EXTERNAL_LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    text = _MAGIC_WORD_RE.sub("", text)
    text = _LIST_PREFIX_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _local_name(tag: str) -> str:
    """Strip the MediaWiki XML namespace from an element tag."""
    return tag.rsplit("}", 1)[-1]


@dataclass(frozen=True)
class WikiPage:
    title: str
    text: str


def iter_pages(dump_path: str | Path, namespace: str = "0") -> Iterator[WikiPage]:
    """Stream article pages out of a MediaWiki dump.

    Only `namespace` 0 (main/article space) is yielded by default, which
    excludes Talk, User, Template and Category pages. Redirects are
    skipped: they carry no prose.
    """
    path = Path(dump_path)
    opener = bz2.open if path.suffix == ".bz2" else open

    with opener(path, "rb") as handle:
        title: str | None = None
        page_ns: str | None = None
        body: str | None = None

        for event, element in ElementTree.iterparse(handle, events=("end",)):
            tag = _local_name(element.tag)

            if tag == "title":
                title = element.text
            elif tag == "ns":
                page_ns = element.text
            elif tag == "text":
                body = element.text
            elif tag == "page":
                if page_ns == namespace and title and body and not _REDIRECT_RE.match(body):
                    yield WikiPage(title=title, text=body)
                title = page_ns = body = None
                # Releasing the finished page keeps memory flat regardless
                # of dump size.
                element.clear()


def load_documents(
    dump_path: str | Path,
    limit: int | None = None,
    domain: str = "wikipedia",
) -> Iterator[Document]:
    """Stream cleaned Wikipedia articles as pipeline-ready `Document`s."""
    for index, page in enumerate(iter_pages(dump_path)):
        if limit is not None and index >= limit:
            break
        cleaned = clean_wikitext(page.text)
        if not cleaned:
            continue
        yield Document(
            id=f"kmwiki_{index:07d}",
            text=cleaned,
            source="wikipedia:km",
            license=WIKIPEDIA_LICENSE,
            domain=domain,
            language="km",
            metadata={"title": page.title},
        )
