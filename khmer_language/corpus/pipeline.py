"""The corpus pipeline (README section 6).

Implements the stated flow, minus collection itself:

    Collection            <- caller's job; see note below
    License check         <- enforced: documents must declare a licence
    Deduplication         <- dedup.py (exact + near-duplicate)
    Unicode normalization <- unicode/normalizer.py
    Quality filtering     <- quality.py
    Language ID           <- language_id.py
    Sentence segmentation <- unicode/sentence.py
    Training dataset      <- JSONL out

**Collection is deliberately not implemented here.** Fetching Khmer text
from the web involves licensing, robots.txt, terms of service and
provenance decisions that belong to the project owner, not to a
general-purpose function. This module takes documents that already exist
and processes them, so it works identically whether they came from
Wikipedia dumps, an existing dataset, or hand-typed text.

Every stage records what it dropped and why, because "my corpus went from
100k to 3k documents" is otherwise impossible to debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..unicode.normalizer import normalize
from ..unicode.sentence import sentence_strings
from .dedup import deduplicate
from .document import Document
from .language_id import identify
from .quality import score_document


@dataclass
class PipelineStats:
    input_documents: int = 0
    missing_license: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    wrong_language: int = 0
    low_quality: int = 0
    too_short: int = 0
    output_documents: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        return self.input_documents - self.output_documents

    def __str__(self) -> str:
        lines = [
            f"input:              {self.input_documents}",
            f"  missing license:  {self.missing_license}",
            f"  exact duplicates: {self.exact_duplicates}",
            f"  near duplicates:  {self.near_duplicates}",
            f"  wrong language:   {self.wrong_language}",
            f"  too short:        {self.too_short}",
            f"  low quality:      {self.low_quality}",
            f"output:             {self.output_documents}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class PipelineResult:
    documents: list[Document]
    stats: PipelineStats


def run_pipeline(
    documents: list[Document],
    *,
    min_quality: float = 0.6,
    min_khmer_ratio: float = 0.5,
    min_graphemes: int = 20,
    near_duplicate_threshold: float = 0.8,
    require_license: bool = True,
) -> PipelineResult:
    """Clean and filter a document list into a training-ready corpus."""
    stats = PipelineStats(input_documents=len(documents))

    # 1. Licence check, before anything else touches the text.
    licensed = []
    for doc in documents:
        if require_license and not doc.license.strip():
            stats.missing_license += 1
            continue
        licensed.append(doc)

    # 2. Normalize before deduplicating, so that documents differing only
    #    in whitespace or zero-width characters collapse together.
    for doc in licensed:
        doc.text = normalize(doc.text)

    # 3. Deduplicate.
    dedup = deduplicate(licensed, near_duplicate_threshold=near_duplicate_threshold)
    stats.exact_duplicates = dedup.exact_duplicates
    stats.near_duplicates = dedup.near_duplicates

    # 4. Language ID, length, quality.
    kept: list[Document] = []
    for doc in dedup.kept:
        if identify(doc.text).khmer_ratio < min_khmer_ratio:
            stats.wrong_language += 1
            continue

        from ..unicode.grapheme import grapheme_strings

        if len(grapheme_strings(doc.text)) < min_graphemes:
            stats.too_short += 1
            continue

        report = score_document(doc.text)
        if report.score < min_quality:
            stats.low_quality += 1
            continue

        doc.quality = round(report.score, 4)
        kept.append(doc)

    stats.output_documents = len(kept)
    return PipelineResult(documents=kept, stats=stats)


def to_sentences(documents: list[Document]) -> list[str]:
    """Flatten a corpus into sentences (README section 6's segmentation
    step), which is the form the tokenizer and LM training loop consume."""
    sentences: list[str] = []
    for doc in documents:
        sentences.extend(sentence_strings(doc.text))
    return sentences
