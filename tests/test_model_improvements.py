"""Weight tying, residual init scaling, LR schedule, and nucleus sampling."""

import numpy as np
import pytest

from khmer_language.models.from_scratch.gpt import GPTConfig, KhmerGPT
from khmer_language.models.from_scratch.gradcheck import gradients_match, numerical_gradient
from khmer_language.models.from_scratch.layers import Embedding
from khmer_language.models.from_scratch.optimizer import cosine_lr


def _model(tie=True, vocab=20, **kw):
    cfg = dict(vocab_size=vocab, dim=16, num_layers=2, num_heads=2, max_seq_len=16,
               tie_embeddings=tie)
    cfg.update(kw)
    return KhmerGPT(GPTConfig(**cfg), seed=0)


# --------------------------------------------------------------------------
# Weight tying
# --------------------------------------------------------------------------
def test_tying_removes_the_separate_head_matrix():
    tied, untied = _model(tie=True), _model(tie=False)
    assert tied.num_parameters() < untied.num_parameters()
    # the saving is exactly the head matrix: vocab * dim
    assert untied.num_parameters() - tied.num_parameters() == 20 * 16


def test_tied_head_shares_storage_with_the_embedding():
    model = _model(tie=True)
    assert model.head.weight is model.token_emb.weight


def test_shared_parameter_is_reported_exactly_once():
    """Reporting it twice would make the optimizer apply every update to
    it twice, silently doubling its effective learning rate."""
    model = _model(tie=True)
    params = model.parameters()
    assert len({id(p) for p in params}) == len(params)


def test_tied_embedding_gradient_matches_numerical():
    """The shared matrix receives gradient from both the input embedding
    lookup and the output projection; both contributions must be there."""
    model = _model(tie=True, vocab=7, dim=8, max_seq_len=6)
    ids, targets = np.array([[1, 2, 3]]), np.array([[2, 3, 4]])

    for p in model.parameters():
        p.grad.fill(0.0)
    _, dlogits = model.loss(ids, targets)
    model.backward(dlogits)
    analytic = model.token_emb.weight.grad.copy()

    numeric = numerical_gradient(lambda: model.loss(ids, targets)[0], model.token_emb.weight.value)
    assert gradients_match(analytic, numeric)


def test_tied_model_trains():
    from khmer_language.training import train

    model = _model(tie=True, vocab=5)
    data = np.tile(np.arange(5), 40)
    report = train(model, data, steps=60, batch_size=4, seq_len=8, lr=5e-3, seed=0)
    assert report.final_loss < report.losses[0]


# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------
def test_untrained_loss_is_near_uniform():
    """A model that knows nothing should predict near-uniformly. With a
    tied head the embedding scale sets initial confidence, so too large a
    scale makes it start confidently wrong - worse than random."""
    vocab = 50
    model = _model(vocab=vocab)
    ids = np.array([[1, 2, 3, 4]])
    loss, _ = model.loss(ids, np.array([[2, 3, 4, 5]]))
    assert loss == pytest.approx(np.log(vocab), rel=0.1)


def test_residual_projections_are_scaled_by_depth():
    """Each block adds twice into the residual stream, so deeper stacks
    need smaller writes to keep activations well-scaled."""
    shallow, deep = _model(num_layers=1), _model(num_layers=8)
    shallow_std = np.std(shallow.blocks[0].attn.proj.W.value)
    deep_std = np.std(deep.blocks[0].attn.proj.W.value)
    assert deep_std < shallow_std


def test_embedding_uses_gpt2_init_scale():
    rng = np.random.default_rng(0)
    weight = Embedding(500, 64, rng).weight.value
    assert np.std(weight) == pytest.approx(Embedding.DEFAULT_INIT_STD, rel=0.1)


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------
def test_warmup_ramps_up_linearly_from_near_zero():
    lrs = [cosine_lr(s, 100, base_lr=1.0, warmup=10) for s in range(10)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[-1] == pytest.approx(1.0)
    assert lrs == sorted(lrs)


def test_rate_decays_after_warmup():
    after_warmup = [cosine_lr(s, 100, base_lr=1.0, warmup=10) for s in range(10, 100, 10)]
    assert after_warmup == sorted(after_warmup, reverse=True)


def test_final_rate_respects_the_floor():
    assert cosine_lr(100, 100, base_lr=1.0, warmup=10, min_ratio=0.1) == pytest.approx(0.1)


def test_schedule_never_exceeds_the_base_rate():
    for step in range(0, 200):
        assert cosine_lr(step, 100, base_lr=2e-3, warmup=10) <= 2e-3 + 1e-12


def test_training_with_schedule_still_reduces_loss():
    from khmer_language.training import train

    model = _model(vocab=5)
    data = np.tile(np.arange(5), 40)
    report = train(model, data, steps=80, batch_size=4, seq_len=8, lr=5e-3, seed=0, schedule=True)
    assert report.final_loss < report.losses[0]


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def test_top_p_keeps_at_least_one_token():
    """A p smaller than the largest single probability must not empty the
    candidate set."""
    model = _model(vocab=20)
    out = model.generate([1], max_new_tokens=3, top_p=1e-6, rng=np.random.default_rng(0))
    assert len(out) == 4


def test_top_p_restricts_the_candidate_set():
    model = _model(vocab=20)
    logits = model.forward(np.array([[1]]))[0, -1]
    from khmer_language.models.from_scratch.layers import softmax

    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    nucleus = set(order[: int(np.searchsorted(np.cumsum(probs[order]), 0.5) + 1)].tolist())

    sampled = {
        model.generate([1], max_new_tokens=1, top_p=0.5, rng=np.random.default_rng(s))[-1]
        for s in range(30)
    }
    assert sampled <= nucleus


def test_repetition_penalty_reduces_repeats():
    """Undertrained models loop; the penalty should measurably reduce that."""
    model = _model(vocab=10)
    prompt = [3, 3, 3, 3]

    def repeats(penalty):
        counts = []
        for seed in range(12):
            out = model.generate(
                prompt, max_new_tokens=20, repetition_penalty=penalty,
                rng=np.random.default_rng(seed),
            )
            counts.append(out[len(prompt):].count(3))
        return sum(counts)

    assert repeats(2.0) <= repeats(1.0)


def test_repetition_penalty_of_one_changes_nothing():
    model = _model(vocab=10)
    a = model.generate([1, 2], max_new_tokens=8, repetition_penalty=1.0, rng=np.random.default_rng(5))
    b = model.generate([1, 2], max_new_tokens=8, rng=np.random.default_rng(5))
    assert a == b


def test_sampling_options_combine_without_error():
    model = _model(vocab=30)
    out = model.generate(
        [1, 2], max_new_tokens=10, temperature=0.8, top_k=10, top_p=0.9,
        repetition_penalty=1.2, rng=np.random.default_rng(0),
    )
    assert len(out) == 12
    assert all(0 <= i < 30 for i in out)
