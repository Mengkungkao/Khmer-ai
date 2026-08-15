"""Multi-head causal self-attention with a hand-derived backward pass.

The core of the transformer (README section 13). Forward pass, for each
attention head:

    scores = (Q K^T) / sqrt(head_dim)     # (T, T) - how much each
                                          # position attends to each other
    scores = mask(scores)                 # causal: position i may not
                                          # look at any j > i
    attn   = softmax(scores)
    out    = attn V

The 1/sqrt(head_dim) scaling matters: without it, dot products grow with
dimension, softmax saturates, and gradients vanish.

Causal masking is what makes this a *language model* rather than an
encoder - each position may only use earlier context, so the model can be
trained on every position of a sequence in parallel while never letting a
position see its own answer.

Masking is done with a large finite negative (-1e9) rather than -inf:
after softmax those entries are exactly 0, and because the softmax
backward formula is `attn * (...)`, their gradients are automatically 0
too - whereas -inf risks inf-times-0 producing nan.
"""

from __future__ import annotations

import numpy as np

from .layers import Layer, Linear, Parameter, softmax


class MultiHeadCausalSelfAttention(Layer):
    def __init__(self, dim: int, num_heads: int, rng: np.random.Generator):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # One fused projection produces Q, K and V together, then it is
        # split - fewer, larger matmuls than three separate Linears.
        self.qkv = Linear(dim, 3 * dim, rng)
        self.proj = Linear(dim, dim, rng)
        self._cache: tuple | None = None

    def parameters(self) -> list[Parameter]:
        return self.qkv.parameters() + self.proj.parameters()

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, C) -> (B, H, T, head_dim)"""
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, H, T, head_dim) -> (B, T, C)"""
        B, H, T, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * hd)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        qkv = self.qkv.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal_mask, -1e9, scores)

        attn = softmax(scores, axis=-1)
        head_out = attn @ v
        merged = self._merge_heads(head_out)

        self._cache = (q, k, v, attn, merged)
        return self.proj.forward(merged)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None, "forward() must be called before backward()"
        q, k, v, attn, _ = self._cache

        d_merged = self.proj.backward(dout)
        d_head_out = self._split_heads(d_merged)

        # out = attn @ v
        d_attn = d_head_out @ v.transpose(0, 1, 3, 2)
        d_v = attn.transpose(0, 1, 3, 2) @ d_head_out

        # softmax Jacobian, applied row-wise without materializing it:
        # ds_i = p_i * (dp_i - sum_j dp_j p_j)
        d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
        d_scores *= self.scale

        d_q = d_scores @ k
        d_k = d_scores.transpose(0, 1, 3, 2) @ q

        d_qkv = np.concatenate(
            [self._merge_heads(d_q), self._merge_heads(d_k), self._merge_heads(d_v)], axis=-1
        )
        return self.qkv.backward(d_qkv)

    def attention_weights(self) -> np.ndarray | None:
        """The most recent forward pass's attention matrix, (B, H, T, T).

        Kept accessible because being able to inspect what the model
        attends to is the point of an auditable stack (README section 1),
        and it is what makes the causal structure visible.
        """
        return None if self._cache is None else self._cache[3]
