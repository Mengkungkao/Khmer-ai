"""KhmerAI Benchmark harness (README section 17).

A benchmark is a list of `BenchmarkCase`s, each with a category, an input,
and an expected output. Running it reports per-category and overall
scores using the metrics in `metrics.py`.

The harness is real and usable; what does not exist yet is a *populated*
benchmark. Section 17's categories (comprehension, knowledge, translation,
reasoning) need authored Khmer test data, which is a content task
requiring native-speaker judgement rather than something to auto-generate
- auto-generating it would produce a benchmark that measures nothing.
`STRUCTURAL_CASES` therefore contains only cases that can be checked
mechanically and objectively, and the harness is ready for real cases to
be added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .metrics import character_error_rate, exact_match, grapheme_error_rate


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    input: str
    expected: str
    note: str = ""


@dataclass(frozen=True)
class CaseResult:
    case: BenchmarkCase
    predicted: str
    exact: bool
    cer: float
    ger: float


@dataclass(frozen=True)
class CategoryScore:
    category: str
    num_cases: int
    exact_match_rate: float
    mean_cer: float
    mean_ger: float


@dataclass(frozen=True)
class BenchmarkResult:
    results: tuple[CaseResult, ...]

    @property
    def categories(self) -> list[CategoryScore]:
        by_category: dict[str, list[CaseResult]] = {}
        for r in self.results:
            by_category.setdefault(r.case.category, []).append(r)

        scores = []
        for category, rows in sorted(by_category.items()):
            n = len(rows)
            scores.append(
                CategoryScore(
                    category=category,
                    num_cases=n,
                    exact_match_rate=sum(r.exact for r in rows) / n,
                    mean_cer=sum(r.cer for r in rows) / n,
                    mean_ger=sum(r.ger for r in rows) / n,
                )
            )
        return scores

    @property
    def overall_exact_match(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.exact for r in self.results) / len(self.results)


def run_benchmark(cases: list[BenchmarkCase], predict: Callable[[str], str]) -> BenchmarkResult:
    """Run `predict` over every case and score the outputs."""
    results = []
    for case in cases:
        predicted = predict(case.input)
        results.append(
            CaseResult(
                case=case,
                predicted=predicted,
                exact=exact_match(case.expected, predicted),
                cer=character_error_rate(case.expected, predicted),
                ger=grapheme_error_rate(case.expected, predicted),
            )
        )
    return BenchmarkResult(results=tuple(results))


def format_benchmark(result: BenchmarkResult) -> str:
    header = f"{'category':<20} {'n':>4} {'exact':>8} {'CER':>8} {'GER':>8}"
    lines = [header, "-" * len(header)]
    for score in result.categories:
        lines.append(
            f"{score.category:<20} {score.num_cases:>4} {score.exact_match_rate:>7.0%} "
            f"{score.mean_cer:>8.3f} {score.mean_ger:>8.3f}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'OVERALL':<20} {len(result.results):>4} {result.overall_exact_match:>7.0%}")
    return "\n".join(lines)


# Mechanically checkable cases only - see the module docstring for why
# there are no comprehension/knowledge/translation cases here yet.
STRUCTURAL_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("normalization", "  កម្ពុជា   ជា  ", "កម្ពុជា ជា", "collapse whitespace"),
    BenchmarkCase("normalization", "មួយ\n\n\n\nពីរ", "មួយ\n\nពីរ", "collapse blank lines"),
    BenchmarkCase("transliteration", "កម្ពុជា", "kâmpŭchéa", "citation romanization"),
    BenchmarkCase("transliteration", "១២៣", "123", "Khmer digits"),
)
