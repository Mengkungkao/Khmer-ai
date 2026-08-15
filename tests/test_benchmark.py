import pytest

from khmer_language.evaluation.benchmark import (
    STRUCTURAL_CASES,
    BenchmarkCase,
    format_benchmark,
    run_benchmark,
)
from khmer_language.unicode.normalizer import normalize
from khmer_language.unicode.transliterator import transliterate


def test_perfect_predictor_scores_full_marks():
    cases = [BenchmarkCase("demo", "in", "expected")]
    result = run_benchmark(cases, lambda _: "expected")
    assert result.overall_exact_match == 1.0
    assert result.results[0].cer == 0.0
    assert result.results[0].ger == 0.0


def test_wrong_predictor_scores_zero_exact_match():
    cases = [BenchmarkCase("demo", "in", "កម្ពុជា")]
    result = run_benchmark(cases, lambda _: "ខុស")
    assert result.overall_exact_match == 0.0
    assert result.results[0].cer > 0


def test_scores_are_grouped_by_category():
    cases = [
        BenchmarkCase("a", "1", "1"),
        BenchmarkCase("a", "2", "2"),
        BenchmarkCase("b", "3", "wrong"),
    ]
    result = run_benchmark(cases, lambda x: x)
    by_name = {c.category: c for c in result.categories}
    assert by_name["a"].num_cases == 2
    assert by_name["a"].exact_match_rate == 1.0
    assert by_name["b"].exact_match_rate == 0.0


def test_structural_cases_pass_against_the_real_implementations():
    """The bundled cases must actually be satisfiable by this repo's own
    normalizer and transliterator - otherwise the benchmark encodes wrong
    expectations."""

    def predict(text: str) -> str:
        case = next(c for c in STRUCTURAL_CASES if c.input == text)
        return normalize(text) if case.category == "normalization" else transliterate(text)

    result = run_benchmark(list(STRUCTURAL_CASES), predict)
    assert result.overall_exact_match == 1.0


def test_empty_benchmark_does_not_crash():
    result = run_benchmark([], lambda x: x)
    assert result.overall_exact_match == 0.0
    assert result.categories == []


def test_format_benchmark_includes_categories_and_overall():
    result = run_benchmark([BenchmarkCase("demo", "a", "a")], lambda x: x)
    output = format_benchmark(result)
    assert "demo" in output
    assert "OVERALL" in output
