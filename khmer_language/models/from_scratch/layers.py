"""Neural network primitives with hand-derived backward passes.

README section 13 asks for the transformer to be built without hiding it
behind a framework, so there is no autograd here: every layer implements
`forward` (caching what the backward pass needs) and `backward` (applying
the chain rule by hand, returning the gradient w.r.t. its input and
accumulating gradients into its parameters).

The correctness bar for "I derived this by hand" is not "it runs" - it is
that each analytic gradient matches a numerical finite-difference
gradient. `tests/test_layers.py` checks exactly that for every layer
here, which is what makes the hand-derived math trustworthy.

Convention: `x` has shape (..., features); all layers act on the last
axis and broadcast over any number of leading batch/time axes.
"""

from __future__ import annotations

import numpy as np


class Parameter:
    """A learnable array plus its accumulated gradient."""

    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = np.zeros_like(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape


class Layer:
    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad.fill(0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Linear(Layer):
    """y = x @ W + b"""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, bias: bool = True):
        # Scaled ("Xavier"-style) init: keeps activation variance roughly
        # stable through depth instead of exploding or vanishing.
        scale = 1.0 / np.sqrt(in_features)
        self.W = Parameter(rng.normal(0.0, scale, size=(in_features, out_features)))
        self.b = Parameter(np.zeros(out_features)) if bias else None
        self._x: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        return [self.W] if self.b is None else [self.W, self.b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.W.value
        if self.b is not None:
            y = y + self.b.value
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        assert x is not None, "forward() must be called before backward()"
        # Flatten all leading axes so the parameter gradient is a single
        # matmul regardless of how many batch/time dimensions there are.
        x_flat = x.reshape(-1, x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        self.W.grad += x_flat.T @ dout_flat
        if self.b is not None:
            self.b.grad += dout_flat.sum(axis=0)
        return dout @ self.W.value.T


class LayerNorm(Layer):
    """Normalize over the last axis, then scale and shift."""

    def __init__(self, features: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones(features))
        self.beta = Parameter(np.zeros(features))
        self.eps = eps
        self._cache: tuple | None = None

    def parameters(self) -> list[Parameter]:
        return [self.gamma, self.beta]

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        self._cache = (x_hat, inv_std)
        return self.gamma.value * x_hat + self.beta.value

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None, "forward() must be called before backward()"
        x_hat, inv_std = self._cache
        features = x_hat.shape[-1]

        axes = tuple(range(dout.ndim - 1))
        self.gamma.grad += (dout * x_hat).sum(axis=axes)
        self.beta.grad += dout.sum(axis=axes)

        dx_hat = dout * self.gamma.value
        # Both correction terms come from mean/variance depending on every
        # element of the row, so each element's gradient feeds back through
        # the row's mean (2nd term) and variance (3rd term).
        dx = inv_std * (
            dx_hat
            - dx_hat.mean(axis=-1, keepdims=True)
            - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
        )
        return dx


class GELU(Layer):
    """Gaussian Error Linear Unit, tanh approximation (as used by GPT-2)."""

    _C = np.sqrt(2.0 / np.pi)

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        inner = self._C * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + np.tanh(inner))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        assert x is not None, "forward() must be called before backward()"
        inner = self._C * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        d_inner = self._C * (1.0 + 3 * 0.044715 * x**2)
        dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * d_inner
        return dout * dgelu


class Embedding(Layer):
    """Integer token ids -> dense vectors (a differentiable row lookup)."""

    # GPT-2's initialization scale. Deliberately smaller than the
    # 1/sqrt(dim) used for Linear layers, and it matters most when the
    # embedding is tied to the output head: logits are then x @ E.T, so
    # the embedding scale sets how confident the model is before it has
    # learned anything. At 1/sqrt(dim) the untrained logits have unit
    # variance and the model starts out confidently wrong - measurably
    # worse than uniform. At 0.02 predictions start near-uniform, which
    # is the correct prior for a model that knows nothing.
    DEFAULT_INIT_STD = 0.02

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        rng: np.random.Generator,
        std: float = DEFAULT_INIT_STD,
    ):
        self.weight = Parameter(rng.normal(0.0, std, size=(num_embeddings, dim)))
        self._ids: np.ndarray | None = None

    def parameters(self) -> list[Parameter]:
        return [self.weight]

    def forward(self, ids: np.ndarray) -> np.ndarray:
        self._ids = ids
        return self.weight.value[ids]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        ids = self._ids
        assert ids is not None, "forward() must be called before backward()"
        # A token appearing several times must accumulate all its
        # gradients, so this needs scatter-add, not plain assignment.
        np.add.at(self.weight.grad, ids, dout)
        return np.zeros(0)  # ids are integers; no gradient flows to them


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exponentiating,
    otherwise large logits overflow to inf)."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over a batch, plus the gradient w.r.t. `logits`.

    `logits` is (..., vocab), `targets` holds the correct id per position.
    Returns (loss, dlogits). The softmax and the cross-entropy are fused
    because their composed gradient simplifies exactly to (probs - onehot),
    which is both faster and more numerically stable than backpropagating
    through a separate softmax layer.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

    probs = softmax(flat_logits, axis=-1)
    correct = probs[np.arange(n), flat_targets]
    loss = float(-np.log(np.clip(correct, 1e-12, None)).mean())

    dlogits = probs.copy()
    dlogits[np.arange(n), flat_targets] -= 1.0
    dlogits /= n
    return loss, dlogits.reshape(logits.shape)
