"""Khmer text normalization.

Deliberately conservative for v1: this module only performs transforms
that are unambiguously safe (standard Unicode NFC, whitespace cleanup,
optional zero-width stripping). It does NOT attempt to silently reorder
combining-mark sequences into a "canonical" input order (e.g. fixing a
COENG cluster typed in the wrong order relative to a vowel sign). That
class of fix is well known in Khmer NLP tooling (e.g. the community
`khnormal` project), but the exact rule table needs a dedicated reference
to get right, so for now `validator.py` only *reports* suspicious
orderings rather than this module silently rewriting them. Silently
"fixing" text you are not fully certain about is worse than leaving it
alone and flagging it.
"""

from __future__ import annotations

import re
import unicodedata

from .character_types import ZWJ, ZWNJ, ZWSP

_ZERO_WIDTH_CHARS = "".join(chr(cp) for cp in (ZWSP, ZWNJ, ZWJ))
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")
_ASCII_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")


def normalize(text: str, *, form: str = "NFC") -> str:
    """Apply standard Unicode normalization plus safe whitespace cleanup."""
    text = unicodedata.normalize(form, text)
    text = _ASCII_WHITESPACE_RUN_RE.sub(" ", text)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_zero_width(text: str) -> str:
    """Remove ZWSP/ZWNJ/ZWJ. Useful for display/comparison; do not use
    before tokenization if you want to preserve ZWSP word-boundary hints.
    """
    return _ZERO_WIDTH_RE.sub("", text)


def has_zero_width(text: str) -> bool:
    return bool(_ZERO_WIDTH_RE.search(text))
