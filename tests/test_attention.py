import numpy as np
import pytest

from khmer_language.models.from_scratch.attention import MultiHeadCausalSelfAttention
from khmer_language.models.from_scratch.gradcheck import gradients_match, numerical_gradient

def _layer(dim=8, heads=2, seed=0):
    return MultiHeadCausalSelfAttention(dim, heads, np.random.default_rng(seed))


def test_rejects_dim_not_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadCausalSelfAttention(10, 4, np.random.default_rng(0))


def test_output_shape_matches_input():
    rng = np.random.default_rng(1)
    layer = _layer()
    x = rng.normal(size=(2, 5, 8))
    assert layer.forward(x).shape == x.shape


def test_attention_rows_sum_to_one():
    rng = np.random.default_rng(2)
    layer = _layer()
    layer.forward(rng.normal(size=(2, 6, 8)))
    attn = layer.attention_weights()
    assert np.allclose(attn.sum(axis=-1), 1.0)


def test_causal_mask_gives_exactly_zero_weight_to_future_positions():
    rng = np.random.default_rng(3)
    layer = _layer()
    T = 6
    layer.forward(rng.normal(size=(1, T, 8)))
    attn = layer.attention_weights()
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    assert np.all(attn[..., future] == 0.0)


def test_first_position_attends_only_to_itself():
    rng = np.random.default_rng(4)
    layer = _layer()
    layer.forward(rng.normal(size=(1, 5, 8)))
    attn = layer.attention_weights()
    assert attn[0, :, 0, 0] == pytest.approx(1.0)


def test_future_tokens_cannot_change_earlier_outputs():
    """The defining property of causal attention: editing position t must
    leave outputs at positions < t bit-for-bit identical."""
    rng = np.random.default_rng(5)
    layer = _layer()
    x = rng.normal(size=(1, 6, 8))
    out_a = layer.forward(x).copy()

    x_modified = x.copy()
    x_modified[0, 4:] = rng.normal(size=(2, 8))  # scramble positions 4,5
    out_b = layer.forward(x_modified)

    assert np.allclose(out_a[0, :4], out_b[0, :4])
    assert not np.allclose(out_a[0, 4:], out_b[0, 4:])


def test_input_gradient_matches_numerical():
    rng = np.random.default_rng(6)
    layer = _layer()
    x = rng.normal(size=(2, 4, 8))
    dout = rng.normal(size=(2, 4, 8))

    layer.zero_grad()
    layer.forward(x)
    analytic = layer.backward(dout)

    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), x)
    assert gradients_match(analytic, numeric)


@pytest.mark.parametrize("param_index", range(4))
def test_parameter_gradients_match_numerical(param_index):
    rng = np.random.default_rng(7)
    layer = _layer()
    x = rng.normal(size=(1, 4, 8))
    dout = rng.normal(size=(1, 4, 8))
    param = layer.parameters()[param_index]

    layer.zero_grad()
    layer.forward(x)
    layer.backward(dout)
    analytic = param.grad.copy()

    numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), param.value)
    assert gradients_match(analytic, numeric)


def test_single_head_and_multi_head_both_gradient_check():
    for heads in (1, 4):
        rng = np.random.default_rng(8)
        layer = MultiHeadCausalSelfAttention(8, heads, rng)
        x = rng.normal(size=(1, 4, 8))
        dout = rng.normal(size=(1, 4, 8))

        layer.zero_grad()
        layer.forward(x)
        analytic = layer.backward(dout)
        numeric = numerical_gradient(lambda: float((layer.forward(x) * dout).sum()), x)
        assert gradients_match(analytic, numeric), f"failed for heads={heads}"


def test_key_bias_provably_cannot_affect_the_output():
    """The key bias is mathematically redundant in attention.

    scores[i,j] = q_i . (k_j + b_k) = q_i.k_j + q_i.b_k, and the q_i.b_k
    term is identical across all j within row i. Softmax is shift-
    invariant along the axis it normalizes, so b_k cancels exactly and
    its gradient is exactly zero.

    This is why some implementations drop the key bias entirely. It is
    also why gradient checking here needs an absolute tolerance: the
    finite-difference estimate of an exactly-zero gradient is pure
    cancellation noise (~1e-11), which no relative test can accept.
    """
    rng = np.random.default_rng(0)
    layer = _layer()
    x = rng.normal(size=(1, 4, 8))

    before = layer.forward(x).copy()
    layer.qkv.b.value[8:16] += rng.normal(size=8) * 5.0  # middle third = K bias
    after = layer.forward(x)
    assert np.allclose(before, after, atol=1e-12)

    # ...and the query bias, by contrast, genuinely does matter.
    layer.qkv.b.value[0:8] += rng.normal(size=8) * 5.0
    assert not np.allclose(before, layer.forward(x))


def test_sequence_length_one_works():
    rng = np.random.default_rng(9)
    layer = _layer()
    x = rng.normal(size=(1, 1, 8))
    assert layer.forward(x).shape == (1, 1, 8)
    assert layer.attention_weights()[0, 0, 0, 0] == pytest.approx(1.0)
