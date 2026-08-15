import numpy as np
import pytest

from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.tokenizer import GraphemeTokenizer
from khmer_language.tokenizer.compare import SAMPLE_CORPUS
from khmer_language.training import encode_corpus, make_batch, train
from khmer_language.unicode.validator import is_valid


def test_encode_corpus_flattens_to_one_id_stream():
    tok = GraphemeTokenizer()
    tok.train(list(SAMPLE_CORPUS))
    data = encode_corpus(tok, list(SAMPLE_CORPUS))
    assert data.ndim == 1
    assert len(data) == sum(len(tok.encode(t)) for t in SAMPLE_CORPUS)


def test_make_batch_targets_are_inputs_shifted_by_one():
    data = np.arange(50)
    x, y = make_batch(data, batch_size=4, seq_len=5, rng=np.random.default_rng(0))
    assert x.shape == (4, 5)
    assert y.shape == (4, 5)
    assert np.all(y == x + 1)  # data is 0,1,2,... so the shift is visible


def test_make_batch_rejects_corpus_shorter_than_window():
    with pytest.raises(ValueError, match="too short"):
        make_batch(np.arange(4), batch_size=2, seq_len=8, rng=np.random.default_rng(0))


def test_training_reduces_loss_on_khmer_text():
    tok = GraphemeTokenizer()
    tok.train(list(SAMPLE_CORPUS))
    data = encode_corpus(tok, list(SAMPLE_CORPUS))

    model = KhmerGPT(
        GPTConfig(vocab_size=len(tok.vocab), dim=16, num_layers=1, num_heads=2, max_seq_len=16),
        seed=0,
    )
    report = train(model, data, steps=40, batch_size=4, seq_len=8, lr=3e-3, seed=0)

    assert report.improved
    assert np.all(np.isfinite(report.losses))
    early = np.mean(report.losses[:5])
    late = np.mean(report.losses[-5:])
    assert late < early


def test_model_can_memorize_a_repeating_pattern():
    """The definitive end-to-end check that forward, backward and the
    optimizer are all correct together: on a perfectly predictable
    repeating sequence the model must drive loss near zero. A subtly wrong
    gradient anywhere would leave it stuck well above that."""
    data = np.tile(np.arange(5), 40)  # 0,1,2,3,4,0,1,2,3,4,...
    model = KhmerGPT(
        GPTConfig(vocab_size=5, dim=16, num_layers=1, num_heads=2, max_seq_len=16), seed=0
    )
    report = train(model, data, steps=120, batch_size=4, seq_len=8, lr=5e-3, seed=0)
    assert report.final_loss < 0.1


def test_gradient_norms_are_recorded_and_finite():
    data = np.tile(np.arange(5), 30)
    model = KhmerGPT(GPTConfig(vocab_size=5, dim=8, num_layers=1, num_heads=2, max_seq_len=8), seed=0)
    report = train(model, data, steps=10, batch_size=2, seq_len=4, seed=0)
    assert len(report.grad_norms) == 10
    assert np.all(np.isfinite(report.grad_norms))


def test_generated_text_is_structurally_valid_khmer():
    """Because the grapheme tokenizer can only ever emit whole Khmer
    grapheme clusters, structural Unicode validity (README section 29,
    Levels 1-2) holds by construction rather than having to be learned -
    even before the model is trained. Worth asserting so a future
    tokenizer change that breaks the guarantee is caught."""
    tok = GraphemeTokenizer()
    tok.train(list(SAMPLE_CORPUS))
    data = encode_corpus(tok, list(SAMPLE_CORPUS))

    model = KhmerGPT(
        GPTConfig(vocab_size=len(tok.vocab), dim=16, num_layers=1, num_heads=2, max_seq_len=16),
        seed=0,
    )
    text = tok.decode(
        model.generate(list(data[:3]), max_new_tokens=20, rng=np.random.default_rng(0))
    )
    assert is_valid(text)
