import numpy as np
import pytest

from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.models.from_scratch.gradcheck import gradients_match, numerical_gradient


def _tiny(**kwargs):
    defaults = dict(vocab_size=11, dim=8, num_layers=1, num_heads=2, max_seq_len=8)
    defaults.update(kwargs)
    return KhmerGPT(GPTConfig(**defaults), seed=0)


def test_config_rejects_dim_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(vocab_size=5, dim=9, num_heads=2)


def test_forward_shape_is_batch_time_vocab():
    model = _tiny()
    ids = np.array([[1, 2, 3], [4, 5, 6]])
    assert model.forward(ids).shape == (2, 3, 11)


def test_forward_accepts_unbatched_ids():
    model = _tiny()
    assert model.forward(np.array([1, 2, 3])).shape == (1, 3, 11)


def test_rejects_sequence_longer_than_max_seq_len():
    model = _tiny(max_seq_len=4)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward(np.arange(5)[None, :])


def test_untrained_loss_is_about_log_vocab_size():
    """A randomly initialized model should be near-uniform over the vocab,
    so cross-entropy should start close to ln(vocab_size). A very
    different starting loss means the initialization is wrong."""
    model = _tiny(vocab_size=20)
    ids = np.array([[1, 2, 3, 4]])
    loss, _ = model.loss(ids, np.array([[2, 3, 4, 5]]))
    assert loss == pytest.approx(np.log(20), rel=0.25)


def test_num_parameters_counts_every_parameter_array():
    model = _tiny()
    assert model.num_parameters() == sum(int(np.prod(p.shape)) for p in model.parameters())
    assert model.num_parameters() > 0


def test_model_is_causal_end_to_end():
    """Changing a later token must not alter logits at earlier positions."""
    model = _tiny(max_seq_len=8)
    ids = np.array([[1, 2, 3, 4, 5]])
    out_a = model.forward(ids).copy()

    ids_b = ids.copy()
    ids_b[0, 3:] = [9, 10]
    out_b = model.forward(ids_b)

    assert np.allclose(out_a[0, :3], out_b[0, :3])
    assert not np.allclose(out_a[0, 3:], out_b[0, 3:])


def test_full_model_gradient_check():
    """End-to-end gradient check through embeddings, blocks, norm and head
    - this is what proves the assembled model trains correctly, not just
    the individual layers."""
    model = _tiny(vocab_size=7, dim=8, num_layers=2, num_heads=2, max_seq_len=6, tie_embeddings=False)
    ids = np.array([[1, 2, 3]])
    targets = np.array([[2, 3, 4]])

    for param in (model.pos_emb, model.token_emb.weight, model.head.W):
        for p in model.parameters():
            p.grad.fill(0.0)
        _, dlogits = model.loss(ids, targets)
        model.backward(dlogits)
        analytic = param.grad.copy()

        numeric = numerical_gradient(lambda: model.loss(ids, targets)[0], param.value)
        assert gradients_match(analytic, numeric), f"gradient mismatch for {param.shape}"


def test_positional_embedding_gradient_only_touches_used_positions():
    model = _tiny(max_seq_len=8)
    for p in model.parameters():
        p.grad.fill(0.0)
    _, dlogits = model.loss(np.array([[1, 2, 3]]), np.array([[2, 3, 4]]))
    model.backward(dlogits)
    assert np.any(model.pos_emb.grad[:3] != 0)
    assert np.all(model.pos_emb.grad[3:] == 0)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def test_generate_returns_prompt_plus_requested_tokens():
    model = _tiny()
    out = model.generate([1, 2], max_new_tokens=4, rng=np.random.default_rng(0))
    assert len(out) == 6
    assert out[:2] == [1, 2]


def test_generate_ids_are_all_within_vocab():
    model = _tiny(vocab_size=11)
    out = model.generate([1], max_new_tokens=10, rng=np.random.default_rng(1))
    assert all(0 <= i < 11 for i in out)


def test_greedy_generation_is_deterministic():
    model = _tiny()
    a = model.generate([1, 2], max_new_tokens=5, temperature=0.0)
    b = model.generate([1, 2], max_new_tokens=5, temperature=0.0)
    assert a == b


def test_sampling_is_reproducible_given_the_same_seed():
    model = _tiny()
    a = model.generate([1], max_new_tokens=6, rng=np.random.default_rng(3))
    b = model.generate([1], max_new_tokens=6, rng=np.random.default_rng(3))
    assert a == b


def test_top_k_restricts_sampling_to_the_k_most_likely_tokens():
    model = _tiny(vocab_size=11)
    logits = model.forward(np.array([[1]]))[0, -1]
    allowed = set(np.argsort(logits)[-2:].tolist())
    out = model.generate([1], max_new_tokens=1, top_k=2, rng=np.random.default_rng(4))
    assert out[-1] in allowed


def test_generation_past_context_window_does_not_crash():
    """Prompt longer than max_seq_len must be truncated to the most recent
    window rather than indexing past the positional embeddings."""
    model = _tiny(max_seq_len=4)
    out = model.generate([1, 2, 3, 4, 5, 6], max_new_tokens=3, rng=np.random.default_rng(5))
    assert len(out) == 9
