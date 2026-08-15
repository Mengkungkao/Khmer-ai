"""Khmer corpus building pipeline (README.md sections 6-7).

Collection itself is intentionally not implemented - see `pipeline.py`.
This package processes documents that already exist, regardless of source.
"""

from .dedup import DedupResult, MinHasher, content_hash, deduplicate, jaccard, shingles
from .document import Document, read_jsonl, write_jsonl
from .language_id import LanguageScore, identify, is_khmer
from .pipeline import PipelineResult, PipelineStats, run_pipeline, to_sentences
from .quality import QualityReport, format_quality, score_document

__all__ = [
    "DedupResult",
    "MinHasher",
    "content_hash",
    "deduplicate",
    "jaccard",
    "shingles",
    "Document",
    "read_jsonl",
    "write_jsonl",
    "LanguageScore",
    "identify",
    "is_khmer",
    "PipelineResult",
    "PipelineStats",
    "run_pipeline",
    "to_sentences",
    "QualityReport",
    "format_quality",
    "score_document",
]
