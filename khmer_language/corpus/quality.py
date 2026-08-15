"""Document quality scoring (README section 7).

Section 7 proposes:

    quality_score = language + unicode + grammar + duplication
                  + source + readability

Four of those six are honestly computable with what this repo has; two
are not, and this module says so rather than inventing them - the same
stance `evaluation/error_analyzer.py` takes:

  IMPLEMENTED
    script      - Khmer script ratio (language_id.py; reliable, since the
                  Khmer block is used essentially only by Khmer)
    unicode     - structural validity (Project 1's validator)
    repetition  - internal duplication, catching the "extremely repetitive
                  documents" and "meaningless character sequences" that
                  section 7 lists for removal
    markup      - leftover HTML/markup, section 7's "broken HTML"

  NOT IMPLEMENTED (never silently scored)
    grammar     - needs a parser or trained grammar model
    readability - needs a dictionary and word-frequency data (section 8)

`source_score` is not computed at all: trust in a source is a judgement
the corpus builder makes, so it is passed in as a weight rather than
guessed at.

Scoring is a weighted **geometric** mean, not the sum section 7 sketches
and not an arithmetic mean. Both alternatives were tried and both failed
on real inputs:

  - A *sum* silently changes range whenever a component is added or
    removed, so any threshold tuned against it breaks.

  - An *arithmetic mean* lets one catastrophic component be averaged away
    by unrelated good ones. Two concrete failures caught while testing:
    English text scored **0.71** (script was correctly 0.0, but `unicode`
    and `markup` are *vacuously* 1.0 when there is no Khmer to be
    malformed), and spam consisting of one word repeated 80 times scored
    **0.605** and passed the filter despite `repetition` = 0.013.

The geometric mean fixes both with one mechanism, because it is dominated
by its smallest term: any component at 0 forces the whole score to 0, and
a component at 0.01 drags the result down hard instead of being diluted.
That matches what quality filtering actually needs - a document is only
as good as its worst dimension, since one fatal flaw makes it unusable
regardless of how clean everything else is.

Components are still reported individually, so *why* a document scored
badly is always visible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..unicode.grapheme import grapheme_strings
from ..unicode.validator import validate
from .language_id import identify

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,5});")
_URL_RE = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class QualityReport:
    score: float
    components: dict[str, float]
    unavailable: tuple[str, ...] = field(default=())

    def passes(self, threshold: float) -> bool:
        return self.score >= threshold


def _script_score(text: str) -> float:
    return identify(text).khmer_ratio


def _unicode_score(text: str) -> float:
    """1.0 for structurally clean Khmer, falling as errors accumulate.

    Scaled per grapheme cluster so a long document is not punished more
    than a short one for the same error density.
    """
    units = max(len(grapheme_strings(text)), 1)
    errors = sum(1 for i in validate(text) if i.severity == "error")
    return max(0.0, 1.0 - (errors / units))


def _repetition_score(text: str, window: int = 5) -> float:
    """1.0 = no excess repetition, 0.0 = fully repetitive.

    Measured as the ratio of distinct to total grapheme n-grams. Spam and
    degenerate machine output repeat n-grams heavily; natural prose does
    not.
    """
    units = grapheme_strings(text)
    if len(units) < window + 1:
        return 1.0
    ngrams = [tuple(units[i : i + window]) for i in range(len(units) - window + 1)]
    return len(set(ngrams)) / len(ngrams)


def _markup_score(text: str) -> float:
    """1.0 = clean text, lower when HTML tags/entities/URLs dominate."""
    if not text:
        return 1.0
    markup_chars = sum(
        len(m.group()) for pattern in (_HTML_TAG_RE, _HTML_ENTITY_RE, _URL_RE) for m in pattern.finditer(text)
    )
    return max(0.0, 1.0 - markup_chars / len(text))


DEFAULT_WEIGHTS: dict[str, float] = {
    "script": 1.0,
    "unicode": 1.0,
    "repetition": 1.0,
    "markup": 0.5,
}

UNAVAILABLE_COMPONENTS: tuple[str, ...] = (
    "grammar (needs a parser or trained grammar model)",
    "readability (needs the Khmer dictionary and word frequencies, section 8)",
)


def score_document(
    text: str,
    weights: dict[str, float] | None = None,
    source_score: float | None = None,
) -> QualityReport:
    """Score `text` in [0, 1] as Khmer training data.

    Uses a weighted geometric mean, so the score is dominated by the worst
    component (see the module docstring). `source_score`, if given, is a
    caller-supplied trust weight rather than anything this code infers.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    components = {
        "script": _script_score(text),
        "unicode": _unicode_score(text),
        "repetition": _repetition_score(text),
        "markup": _markup_score(text),
    }
    if source_score is not None:
        components["source"] = source_score
        weights.setdefault("source", 1.0)

    weighted = [(v, weights.get(name, 0.0)) for name, v in components.items()]
    total_weight = sum(w for _, w in weighted)
    if total_weight == 0:
        return QualityReport(0.0, components, UNAVAILABLE_COMPONENTS)

    # Any weighted component at exactly 0 makes the geometric mean 0;
    # short-circuit rather than evaluating log(0).
    if any(v <= 0.0 and w > 0 for v, w in weighted):
        return QualityReport(0.0, components, UNAVAILABLE_COMPONENTS)

    log_mean = sum(w * math.log(v) for v, w in weighted if w > 0) / total_weight
    return QualityReport(
        score=math.exp(log_mean), components=components, unavailable=UNAVAILABLE_COMPONENTS
    )


def format_quality(report: QualityReport) -> str:
    lines = [f"quality: {report.score:.3f}", "  components:"]
    for name, value in sorted(report.components.items()):
        lines.append(f"    {name:<12} {value:.3f}")
    lines.append("  not scored:")
    for name in report.unavailable:
        lines.append(f"    {name}")
    return "\n".join(lines)
