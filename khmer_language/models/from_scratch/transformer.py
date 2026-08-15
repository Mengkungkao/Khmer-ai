"""Feed-forward network and transformer block (README section 13).

The block uses the **pre-norm** arrangement (LayerNorm before each
sublayer, as in GPT-2), not the post-norm of the original 2017 paper:

    x = x + attention(norm1(x))
    x = x + feed_forward(norm2(x))

Pre-norm keeps a clean identity path from input to output, so gradients
reach early layers without passing through a LayerNorm at every step.
That is what makes deep stacks trainable without a learning-rate warmup
schedule - worth knowing rather than copying, since it is the single
detail most responsible for "my deep transformer won't converge".

Residual backward: because the forward is `x + sublayer(x)`, an incoming
gradient flows down *both* branches and the two contributions are summed.
"""

from __future__ import annotations

import numpy as np

from .attention import MultiHeadCausalSelfAttention
from .layers import GELU, Layer, LayerNorm, Linear, Parameter


class FeedForward(Layer):
    """Linear -> GELU -> Linear, widening by `expansion` in the middle."""

    def __init__(self, dim: int, rng: np.random.Generator, expansion: int = 4):
        hidden = dim * expansion
        self.fc1 = Linear(dim, hidden, rng)
        self.act = GELU()
        self.fc2 = Linear(hidden, dim, rng)

    def parameters(self) -> list[Parameter]:
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.act.forward(self.fc1.forward(x)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.fc1.backward(self.act.backward(self.fc2.backward(dout)))


class TransformerBlock(Layer):
    def __init__(self, dim: int, num_heads: int, rng: np.random.Generator, expansion: int = 4):
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadCausalSelfAttention(dim, num_heads, rng)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, rng, expansion=expansion)

    def parameters(self) -> list[Parameter]:
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Second residual: the "+ x" branch passes dout through unchanged,
        # the sublayer branch goes back through ffn then norm2.
        d = dout + self.norm2.backward(self.ffn.backward(dout))
        # First residual, same structure.
        d = d + self.norm1.backward(self.attn.backward(d))
        return d
