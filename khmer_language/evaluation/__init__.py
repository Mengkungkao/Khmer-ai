"""Khmer evaluation system (README.md sections 17-18)."""

from .benchmark import (
    STRUCTURAL_CASES,
    BenchmarkCase,
    BenchmarkResult,
    CaseResult,
    CategoryScore,
    format_benchmark,
    run_benchmark,
)
from .error_analyzer import (
    FAIL,
    PASS,
    UNAVAILABLE,
    Check,
    ErrorReport,
    analyze_output,
    format_report,
)
from .metrics import (
    PerplexityResult,
    character_error_rate,
    count_validation_errors,
    edit_distance,
    exact_match,
    grapheme_error_rate,
    perplexity,
    unicode_validity_rate,
)

__all__ = [
    "STRUCTURAL_CASES",
    "BenchmarkCase",
    "BenchmarkResult",
    "CaseResult",
    "CategoryScore",
    "format_benchmark",
    "run_benchmark",
    "PASS",
    "FAIL",
    "UNAVAILABLE",
    "Check",
    "ErrorReport",
    "analyze_output",
    "format_report",
    "PerplexityResult",
    "character_error_rate",
    "count_validation_errors",
    "edit_distance",
    "exact_match",
    "grapheme_error_rate",
    "perplexity",
    "unicode_validity_rate",
]
