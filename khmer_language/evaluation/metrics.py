"""Evaluation metrics for Khmer text and Khmer language models.

README section 17 insists evaluation must not be "loss = 2.31". This
module implements the metrics that are genuinely measurable with what
exists in this repo today; `error_analyzer.py` is explicit about the ones
that are not yet measurable (spelling, grammar, naturalness all need a
dictionary or reference model that Project 3/8 have not built).

The Khmer-specific point here is **Grapheme Error Rate**. Character Error
Rate is the standard metric, but on Khmer a codepoint-level CER badly
misrepresents errors: getting the single grapheme cluster ម្ពុ wrong
counts as 3 codepoint errors (ម, COENG, ព plus the vowel), while a
different single-codepoint mistake counts as 1. Measuring over grapheme
clusters instead - the units a Khmer reader actually perceives - gives a
number that reflects how wrong the output really is. Both are provided,
since CER remains the comparable-to-other-work number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..unicode.grapheme import grapheme_strings
from ..unicode.validator import is_valid, validate


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance between two sequences of units.

    Uses the two-row dynamic programme rather than the full matrix: only
    the previous row is ever needed, which keeps memory O(min(n,m))
    instead of O(n*m) for long documents.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_unit in enumerate(reference, start=1):
        current = [i]
        for j, hyp_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_unit != hyp_unit),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Codepoint-level CER = edit_distance / len(reference)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)


def grapheme_error_rate(reference: str, hypothesis: str) -> float:
    """Khmer-aware error rate over grapheme clusters rather than codepoints.

    See the module docstring: this reflects perceived Khmer errors far
    better than CER, because a Khmer grapheme cluster is one unit to a
    reader even when it spans several codepoints.
    """
    ref_units = grapheme_strings(reference)
    if not ref_units:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(ref_units, grapheme_strings(hypothesis)) / len(ref_units)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


def unicode_validity_rate(texts: list[str]) -> float:
    """Fraction of texts with no structural Khmer Unicode errors.

    This is README section 29's Level 1 milestone, made measurable.
    """
    if not texts:
        return 0.0
    return sum(1 for t in texts if is_valid(t)) / len(texts)


def count_validation_errors(texts: list[str]) -> dict[str, int]:
    """Tally structural error codes across texts, for debugging *what*
    kind of invalid Khmer a model produces rather than just how much."""
    counts: dict[str, int] = {}
    for text in texts:
        for issue in validate(text):
            if issue.severity == "error":
                counts[issue.code] = counts.get(issue.code, 0) + 1
    return counts


@dataclass(frozen=True)
class PerplexityResult:
    perplexity: float
    mean_loss: float
    num_tokens: int


def perplexity(model, data: np.ndarray, seq_len: int | None = None, batch_size: int = 8) -> PerplexityResult:
    """Perplexity of `model` on a held-out id stream.

    Perplexity = exp(mean cross-entropy). Read it as "on average the model
    is as uncertain as if choosing uniformly among this many tokens", so a
    perplexity equal to the vocabulary size means the model has learned
    nothing at all, and lower is better.

    Evaluated on consecutive non-overlapping windows so every token is
    counted exactly once - random sampling would make the number depend on
    the seed and be incomparable between runs.
    """
    from ..models.from_scratch.layers import cross_entropy_loss

    seq_len = seq_len or min(model.config.max_seq_len, 32)
    if len(data) < seq_len + 1:
        raise ValueError(f"need at least {seq_len + 1} tokens to evaluate, got {len(data)}")

    windows = [
        (data[s : s + seq_len], data[s + 1 : s + seq_len + 1])
        for s in range(0, len(data) - seq_len - 1, seq_len)
    ]

    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        x = np.stack([w[0] for w in chunk])
        y = np.stack([w[1] for w in chunk])
        loss, _ = cross_entropy_loss(model.forward(x), y)
        total_loss += loss * y.size
        total_tokens += y.size

    mean_loss = total_loss / total_tokens
    return PerplexityResult(
        perplexity=float(np.exp(mean_loss)), mean_loss=mean_loss, num_tokens=total_tokens
    )
