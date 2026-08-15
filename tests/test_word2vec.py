import numpy as np
import pytest

from khmer_language.embeddings import Word2Vec
from khmer_language.tokenizer import GraphemeTokenizer
from khmer_language.tokenizer.compare import SAMPLE_CORPUS


def _model(**kwargs):
    defaults = dict(dim=16, window=2, negative_samples=3, seed=0)
    defaults.update(kwargs)
    return Word2Vec(GraphemeTokenizer(), **defaults)


def test_build_vocab_sets_matrix_shapes():
    m = _model()
    m.build_vocab(list(SAMPLE_CORPUS))
    assert m.W_in.shape == (len(m.vocab), 16)
    assert m.W_out.shape == (len(m.vocab), 16)


def test_noise_probs_are_a_valid_distribution_excluding_specials():
    m = _model()
    m.build_vocab(list(SAMPLE_CORPUS))
    assert m._noise_probs == pytest.approx(m._noise_probs)  # no nan
    assert m._noise_probs.sum() == pytest.approx(1.0)
    for special in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
        assert m._noise_probs[m.vocab.token_to_id[special]] == 0.0


def test_min_count_filters_rare_tokens():
    m = _model(min_count=3)
    m.build_vocab(["កកក ខ"])  # ក appears 3x, ខ once
    assert "ក" in m.vocab
    assert "ខ" not in m.vocab


def test_training_reduces_loss():
    m = _model()
    losses = m.train(list(SAMPLE_CORPUS), epochs=5, learning_rate=0.05)
    assert len(losses) == 5
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_training_produces_no_nan_weights():
    m = _model()
    m.train(list(SAMPLE_CORPUS), epochs=3)
    assert np.all(np.isfinite(m.W_in))
    assert np.all(np.isfinite(m.W_out))


def test_similarity_of_token_with_itself_is_one():
    m = _model()
    m.train(list(SAMPLE_CORPUS), epochs=2)
    token = next(t for t in m.vocab.id_to_token if t not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"))
    assert m.similarity(token, token) == pytest.approx(1.0)


def test_similarity_and_vector_are_none_for_unknown_token():
    m = _model()
    m.train(list(SAMPLE_CORPUS), epochs=1)
    assert m.get_vector("ZZZ") is None
    assert m.similarity("ZZZ", "ក") is None


def test_most_similar_excludes_self_and_special_tokens():
    m = _model()
    m.train(list(SAMPLE_CORPUS), epochs=2)
    token = next(t for t in m.vocab.id_to_token if t not in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"))
    results = m.most_similar(token, top_n=5)
    names = [t for t, _ in results]
    assert token not in names
    assert not ({"<PAD>", "<UNK>", "<BOS>", "<EOS>"} & set(names))
    assert len(results) <= 5


def test_deterministic_given_same_seed():
    a, b = _model(seed=42), _model(seed=42)
    a.train(list(SAMPLE_CORPUS), epochs=2)
    b.train(list(SAMPLE_CORPUS), epochs=2)
    assert np.allclose(a.W_in, b.W_in)


def test_empty_corpus_does_not_crash():
    m = _model()
    assert m.train([""], epochs=2) == []


def test_learns_that_tokens_in_shared_contexts_are_related():
    # README section 11's stated expectation: ថៃ / វៀតណាម / ឡាវ (Thailand,
    # Vietnam, Laos) should become related because they occur in the same
    # contexts. Here ថៃ and ឡាវ are perfectly interchangeable in the
    # corpus while ខ្ញុំ never shares their contexts, so ថៃ must end up
    # closer to ឡាវ's first grapheme than to ខ្ញុំ.
    corpus = [
        "ប្រទេស ថៃ នៅ ជិត កម្ពុជា",
        "ប្រទេស ឡាវ នៅ ជិត កម្ពុជា",
        "ខ្ញុំ ទៅ ផ្សារ",
    ] * 6

    # Seeded for reproducibility, but the effect is not seed-luck: the
    # margin was verified positive (+0.30 to +0.61) across seeds 0-5.
    m = _model(dim=32, negative_samples=5, seed=1)
    m.train(corpus, epochs=8, learning_rate=0.05)

    assert m.similarity("ថៃ", "ឡា") > m.similarity("ថៃ", "ខ្ញុំ")


def test_save_writes_loadable_json(tmp_path):
    import json

    m = _model()
    m.train(list(SAMPLE_CORPUS), epochs=1)
    out = tmp_path / "vectors.json"
    m.save(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["dim"] == 16
    assert len(data["W_in"]) == len(m.vocab)
