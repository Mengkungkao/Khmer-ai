import pytest

from khmer_language.evaluation.cases import (
    BenchmarkCaseError,
    DEFAULT_CASES_PATH,
    case_counts,
    load_cases,
)


def _write(tmp_path, text):
    path = tmp_path / "cases.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_cases(tmp_path):
    path = _write(
        tmp_path,
        '{"category": "knowledge", "input": "តើរាជធានី?", "expected": "ភ្នំពេញ"}\n'
        '{"category": "translation", "input": "Cambodia", "expected": "កម្ពុជា"}\n',
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].category == "knowledge"
    assert cases[0].expected == "ភ្នំពេញ"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    path = _write(
        tmp_path,
        "# a comment explaining the group\n"
        "\n"
        '{"category": "knowledge", "input": "x", "expected": "y"}\n'
        "\n",
    )
    assert len(load_cases(path)) == 1


def test_optional_note_is_preserved(tmp_path):
    path = _write(
        tmp_path, '{"category": "k", "input": "x", "expected": "y", "note": "check me"}\n'
    )
    assert load_cases(path)[0].note == "check me"


def test_missing_file_returns_no_cases(tmp_path):
    assert load_cases(tmp_path / "absent.jsonl") == []


def test_missing_required_field_is_rejected(tmp_path):
    path = _write(tmp_path, '{"category": "k", "input": "x"}\n')
    with pytest.raises(BenchmarkCaseError, match="missing field"):
        load_cases(path)


def test_empty_expected_answer_is_rejected(tmp_path):
    """A case with no expected answer cannot be scored, so loading it
    would silently produce misleading benchmark numbers."""
    path = _write(tmp_path, '{"category": "k", "input": "x", "expected": "  "}\n')
    with pytest.raises(BenchmarkCaseError, match="cannot be scored"):
        load_cases(path)


def test_invalid_json_reports_the_line_number(tmp_path):
    path = _write(tmp_path, '{"category": "k", "input": "x", "expected": "y"}\n{bad json\n')
    with pytest.raises(BenchmarkCaseError, match="line 2"):
        load_cases(path)


def test_non_string_field_is_rejected(tmp_path):
    path = _write(tmp_path, '{"category": "k", "input": 42, "expected": "y"}\n')
    with pytest.raises(BenchmarkCaseError, match="must be a string"):
        load_cases(path)


def test_case_counts_groups_by_category():
    from khmer_language.evaluation.benchmark import BenchmarkCase

    cases = [
        BenchmarkCase("knowledge", "a", "b"),
        BenchmarkCase("knowledge", "c", "d"),
        BenchmarkCase("translation", "e", "f"),
    ]
    assert case_counts(cases) == {"knowledge": 2, "translation": 1}


def test_bundled_case_file_is_valid():
    """The template shipped in data/benchmark/ must actually parse -
    otherwise the first thing a contributor does is hit an error."""
    cases = load_cases(DEFAULT_CASES_PATH)
    assert len(cases) >= 1
    assert all(c.expected.strip() for c in cases)
