"""Every backward pass here is verified against a numerical gradient.

A wrong hand-derived gradient does not crash - it just trains badly - so
these checks are the actual correctness bar for the from-scratch layers.
"""

import numpy as np
import pytest

from khmer_language.models.from_scratch.gradcheck import gradients_match, numerical_gradient
from khmer_language.models.from_scratch.layers import (
    GELU,
    Embedding,
    LayerNorm,
    Linear,
    cross_entropy_loss,
    softmax,
)

def _check_input_grad(layer, x, seed=0):
    """Analytic vs numerical gradient w.r.t. the layer input."""
    rng = np.random.default_rng(seed)
    dout = rng.normal(size=layer.forward(x).shape)

    layer.forward(x)
    analytic = layer.backward(dout)

    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), x)
    return gradients_match(analytic, numeric)


def _check_param_grad(layer, x, param, seed=0):
    """Analytic vs numerical gradient w.r.t. one parameter array."""
    rng = np.random.default_rng(seed)
    dout = rng.normal(size=layer.forward(x).shape)

    layer.zero_grad()
    layer.forward(x)
    layer.backward(dout)
    analytic = param.grad.copy()

    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), param.value)
    return gradients_match(analytic, numeric)


# --------------------------------------------------------------------------
# Linear
# --------------------------------------------------------------------------
def test_linear_forward_shape_and_value():
    rng = np.random.default_rng(0)
    layer = Linear(3, 2, rng)
    x = rng.normal(size=(4, 3))
    expected = x @ layer.W.value + layer.b.value
    assert np.allclose(layer.forward(x), expected)


def test_linear_input_gradient():
    rng = np.random.default_rng(1)
    layer = Linear(4, 3, rng)
    assert _check_input_grad(layer, rng.normal(size=(5, 4)))


def test_linear_weight_and_bias_gradients():
    rng = np.random.default_rng(2)
    layer = Linear(4, 3, rng)
    x = rng.normal(size=(5, 4))
    assert _check_param_grad(layer, x, layer.W)
    assert _check_param_grad(layer, x, layer.b)


def test_linear_handles_extra_batch_dimensions():
    rng = np.random.default_rng(3)
    layer = Linear(4, 3, rng)
    x = rng.normal(size=(2, 5, 4))  # (batch, time, features)
    assert layer.forward(x).shape == (2, 5, 3)
    assert _check_input_grad(layer, x)


def test_linear_without_bias_has_one_parameter():
    rng = np.random.default_rng(4)
    layer = Linear(3, 2, rng, bias=False)
    assert len(layer.parameters()) == 1
    assert _check_input_grad(layer, rng.normal(size=(4, 3)))


# --------------------------------------------------------------------------
# LayerNorm
# --------------------------------------------------------------------------
def test_layernorm_normalizes_last_axis():
    rng = np.random.default_rng(5)
    layer = LayerNorm(6)
    out = layer.forward(rng.normal(size=(4, 6)) * 10 + 3)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_layernorm_input_gradient():
    rng = np.random.default_rng(6)
    layer = LayerNorm(5)
    assert _check_input_grad(layer, rng.normal(size=(4, 5)))


def test_layernorm_gamma_and_beta_gradients():
    rng = np.random.default_rng(7)
    layer = LayerNorm(5)
    x = rng.normal(size=(4, 5))
    assert _check_param_grad(layer, x, layer.gamma)
    assert _check_param_grad(layer, x, layer.beta)


def test_layernorm_input_gradient_with_batch_dims():
    rng = np.random.default_rng(8)
    layer = LayerNorm(4)
    assert _check_input_grad(layer, rng.normal(size=(2, 3, 4)))


# --------------------------------------------------------------------------
# GELU
# --------------------------------------------------------------------------
def test_gelu_is_near_zero_for_large_negative_and_identity_for_large_positive():
    layer = GELU()
    out = layer.forward(np.array([-10.0, 10.0]))
    assert out[0] == pytest.approx(0.0, abs=1e-4)
    assert out[1] == pytest.approx(10.0, abs=1e-4)


def test_gelu_input_gradient():
    rng = np.random.default_rng(9)
    layer = GELU()
    assert _check_input_grad(layer, rng.normal(size=(4, 5)))


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------
def test_embedding_forward_is_a_row_lookup():
    rng = np.random.default_rng(10)
    layer = Embedding(6, 3, rng)
    ids = np.array([0, 4, 2])
    assert np.allclose(layer.forward(ids), layer.weight.value[ids])


def test_embedding_gradient_accumulates_for_repeated_ids():
    rng = np.random.default_rng(11)
    layer = Embedding(5, 3, rng)
    ids = np.array([2, 2, 2])  # same row three times
    layer.zero_grad()
    layer.forward(ids)
    layer.backward(np.ones((3, 3)))
    assert np.allclose(layer.weight.grad[2], 3.0)
    assert np.allclose(layer.weight.grad[0], 0.0)


def test_embedding_weight_gradient_matches_numerical():
    rng = np.random.default_rng(12)
    layer = Embedding(5, 3, rng)
    ids = np.array([1, 3, 1])
    dout = rng.normal(size=(3, 3))

    layer.zero_grad()
    layer.forward(ids)
    layer.backward(dout)
    analytic = layer.weight.grad.copy()

    numeric = numerical_gradient(lambda: float((layer.forward(ids) * dout).sum()), layer.weight.value)
    assert gradients_match(analytic, numeric)


# --------------------------------------------------------------------------
# softmax / cross-entropy
# --------------------------------------------------------------------------
def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(13)
    probs = softmax(rng.normal(size=(4, 7)))
    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.all(probs > 0)


def test_softmax_is_stable_for_huge_logits():
    probs = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0)


def test_softmax_is_shift_invariant():
    rng = np.random.default_rng(14)
    x = rng.normal(size=(3, 5))
    assert np.allclose(softmax(x), softmax(x + 100.0))


def test_cross_entropy_loss_of_confident_correct_prediction_is_near_zero():
    logits = np.array([[0.0, 50.0, 0.0]])
    loss, _ = cross_entropy_loss(logits, np.array([1]))
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_cross_entropy_loss_of_uniform_prediction_is_log_vocab():
    logits = np.zeros((1, 8))
    loss, _ = cross_entropy_loss(logits, np.array([3]))
    assert loss == pytest.approx(np.log(8))


def test_cross_entropy_gradient_matches_numerical():
    rng = np.random.default_rng(15)
    logits = rng.normal(size=(4, 6))
    targets = rng.integers(0, 6, size=4)

    _, analytic = cross_entropy_loss(logits, targets)
    numeric = numerical_gradient(lambda: cross_entropy_loss(logits, targets)[0], logits)
    assert gradients_match(analytic, numeric)


def test_cross_entropy_handles_batch_time_shape():
    rng = np.random.default_rng(16)
    logits = rng.normal(size=(2, 3, 5))
    targets = rng.integers(0, 5, size=(2, 3))
    loss, dlogits = cross_entropy_loss(logits, targets)
    assert np.isfinite(loss)
    assert dlogits.shape == logits.shape
