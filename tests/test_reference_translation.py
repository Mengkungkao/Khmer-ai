"""Tests for the reference translator.

Deliberately offline: a test suite that needs a third-party web service
fails for reasons that have nothing to do with this code. Network
behaviour is exercised by injecting a fake `_fetch`.
"""

import json

import pytest

from khmer_language.evaluation.reference_translation import ReferenceTranslator, Translation


class FakeTranslator(ReferenceTranslator):
    """Records calls so cache behaviour is observable without a network."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = responses
        self.calls = []

    def _fetch(self, text, langpair):
        self.calls.append((text, langpair))
        return self.responses.get(text)


def test_translate_returns_result(tmp_path):
    t = FakeTranslator(
        {"កម្ពុជា": {"translated": "Cambodia", "match": 0.99}},
        cache_path=tmp_path / "c.json",
    )
    result = t.translate("កម្ពុជា")
    assert result.translated == "Cambodia"
    assert result.match == 0.99
    assert not result.from_cache


def test_second_call_is_served_from_cache(tmp_path):
    t = FakeTranslator(
        {"កម្ពុជា": {"translated": "Cambodia", "match": 0.99}},
        cache_path=tmp_path / "c.json",
    )
    t.translate("កម្ពុជា")
    second = t.translate("កម្ពុជា")
    assert second.from_cache
    assert len(t.calls) == 1  # not fetched twice


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    first = FakeTranslator({"ក": {"translated": "ka", "match": 0.9}}, cache_path=path)
    first.translate("ក")

    second = FakeTranslator({}, cache_path=path)  # would fail to fetch
    result = second.translate("ក")
    assert result is not None
    assert result.translated == "ka"
    assert second.calls == []


def test_unreachable_service_returns_none_rather_than_raising(tmp_path):
    """Evaluation must keep working when the service is down."""
    t = FakeTranslator({}, cache_path=tmp_path / "c.json")
    assert t.translate("កម្ពុជា") is None


def test_offline_mode_never_calls_out(tmp_path):
    t = FakeTranslator(
        {"កម្ពុជា": {"translated": "Cambodia", "match": 0.99}},
        cache_path=tmp_path / "c.json",
        offline=True,
    )
    assert t.translate("កម្ពុជា") is None
    assert t.calls == []


def test_offline_mode_still_uses_the_cache(tmp_path):
    path = tmp_path / "c.json"
    warm = FakeTranslator({"ក": {"translated": "ka", "match": 0.9}}, cache_path=path)
    warm.translate("ក")

    offline = FakeTranslator({}, cache_path=path, offline=True)
    assert offline.translate("ក").translated == "ka"


def test_empty_text_is_not_sent(tmp_path):
    t = FakeTranslator({}, cache_path=tmp_path / "c.json")
    assert t.translate("   ") is None
    assert t.calls == []


def test_corrupt_cache_file_is_survived(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not valid json", encoding="utf-8")
    t = FakeTranslator({"ក": {"translated": "ka", "match": 0.9}}, cache_path=path)
    assert t.translate("ក").translated == "ka"


def test_language_pair_is_part_of_the_cache_key(tmp_path):
    t = FakeTranslator(
        {"ក": {"translated": "ka", "match": 0.9}}, cache_path=tmp_path / "c.json"
    )
    t.translate("ក", "km", "en")
    t.translate("ក", "km", "fr")
    assert len(t.calls) == 2  # different pairs must not collide


def test_low_confidence_flag():
    assert Translation("x", "y", 0.2, False).low_confidence
    assert not Translation("x", "y", 0.9, False).low_confidence


def test_round_trip_translates_twice(tmp_path):
    t = FakeTranslator(
        {
            "កម្ពុជា": {"translated": "Cambodia", "match": 0.99},
            "Cambodia": {"translated": "កម្ពុជា", "match": 0.99},
        },
        cache_path=tmp_path / "c.json",
    )
    assert t.round_trip("កម្ពុជា").translated == "កម្ពុជា"
    assert [c[1] for c in t.calls] == ["km|en", "en|km"]


def test_round_trip_returns_none_if_the_first_leg_fails(tmp_path):
    t = FakeTranslator({}, cache_path=tmp_path / "c.json")
    assert t.round_trip("កម្ពុជា") is None


def test_cache_file_is_valid_json(tmp_path):
    path = tmp_path / "c.json"
    t = FakeTranslator({"ក": {"translated": "ka", "match": 0.9}}, cache_path=path)
    t.translate("ក")
    assert json.loads(path.read_text(encoding="utf-8"))
