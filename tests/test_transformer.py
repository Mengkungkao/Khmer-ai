import numpy as np
import pytest

from khmer_language.models.from_scratch.gradcheck import gradients_match, numerical_gradient
from khmer_language.models.from_scratch.transformer import FeedForward, TransformerBlock


def _grad_check_input(layer, x, seed=0):
    rng = np.random.default_rng(seed)
    dout = rng.normal(size=layer.forward(x).shape)
    layer.zero_grad()
    layer.forward(x)
    analytic = layer.backward(dout)
    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), x)
    return gradients_match(analytic, numeric)


def _grad_check_param(layer, x, param, seed=0):
    rng = np.random.default_rng(seed)
    dout = rng.normal(size=layer.forward(x).shape)
    layer.zero_grad()
    layer.forward(x)
    layer.backward(dout)
    analytic = param.grad.copy()
    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), param.value)
    return gradients_match(analytic, numeric)


# --------------------------------------------------------------------------
# FeedForward
# --------------------------------------------------------------------------
def test_feedforward_preserves_shape():
    rng = np.random.default_rng(0)
    ffn = FeedForward(6, rng)
    x = rng.normal(size=(2, 3, 6))
    assert ffn.forward(x).shape == x.shape


def test_feedforward_widens_by_expansion_factor():
    rng = np.random.default_rng(1)
    ffn = FeedForward(6, rng, expansion=4)
    assert ffn.fc1.W.shape == (6, 24)
    assert ffn.fc2.W.shape == (24, 6)


def test_feedforward_input_gradient():
    rng = np.random.default_rng(2)
    ffn = FeedForward(4, rng)
    assert _grad_check_input(ffn, rng.normal(size=(2, 3, 4)))


@pytest.mark.parametrize("param_index", range(4))
def test_feedforward_parameter_gradients(param_index):
    rng = np.random.default_rng(3)
    ffn = FeedForward(4, rng)
    x = rng.normal(size=(1, 3, 4))
    assert _grad_check_param(ffn, x, ffn.parameters()[param_index])


# --------------------------------------------------------------------------
# TransformerBlock
# --------------------------------------------------------------------------
def test_block_preserves_shape():
    rng = np.random.default_rng(4)
    block = TransformerBlock(8, 2, rng)
    x = rng.normal(size=(2, 5, 8))
    assert block.forward(x).shape == x.shape


def test_block_exposes_all_sublayer_parameters():
    rng = np.random.default_rng(5)
    block = TransformerBlock(8, 2, rng)
    # norm1(2) + attn(4) + norm2(2) + ffn(4)
    assert len(block.parameters()) == 12


def test_block_input_gradient():
    rng = np.random.default_rng(6)
    block = TransformerBlock(8, 2, rng)
    assert _grad_check_input(block, rng.normal(size=(1, 4, 8)))


@pytest.mark.parametrize("param_index", range(12))
def test_block_parameter_gradients(param_index):
    rng = np.random.default_rng(7)
    block = TransformerBlock(8, 2, rng)
    x = rng.normal(size=(1, 3, 8))
    assert _grad_check_param(block, x, block.parameters()[param_index])


def test_block_is_causal_end_to_end():
    """The residual/norm/ffn wrapping must not leak future information."""
    rng = np.random.default_rng(8)
    block = TransformerBlock(8, 2, rng)
    x = rng.normal(size=(1, 6, 8))
    out_a = block.forward(x).copy()

    x2 = x.copy()
    x2[0, 4:] = rng.normal(size=(2, 8))
    out_b = block.forward(x2)

    assert np.allclose(out_a[0, :4], out_b[0, :4])


def test_stacked_blocks_gradient_check():
    """Gradients must flow correctly through a depth-2 stack, which is
    where a residual-branch mistake would show up."""
    rng = np.random.default_rng(9)
    b1 = TransformerBlock(8, 2, rng)
    b2 = TransformerBlock(8, 2, rng)
    x = rng.normal(size=(1, 3, 8))
    dout = rng.normal(size=(1, 3, 8))

    def f():
        return float((b2.forward(b1.forward(x)) * dout).sum())

    b1.zero_grad()
    b2.zero_grad()
    b2.forward(b1.forward(x))
    analytic = b1.backward(b2.backward(dout))
    numeric = numerical_gradient(f, x)
    assert gradients_match(analytic, numeric)
