"""Optimizers, from scratch (README section 13 lists Adam explicitly).

Adam (Kingma & Ba, 2015) keeps two running averages per parameter:

    m = beta1*m + (1-beta1)*g        # mean of recent gradients
    v = beta2*v + (1-beta2)*g^2      # mean of recent squared gradients
    p -= lr * m_hat / (sqrt(v_hat) + eps)

Dividing by sqrt(v) gives each parameter its own effective step size, so
parameters with consistently small gradients (common in embeddings for
rare tokens) still move, while noisy ones are damped.

The bias correction (`m_hat`, `v_hat`) is not optional bookkeeping. Both
averages start at zero, so early steps are biased hard toward zero;
without correction the first updates are far too small and training
appears to stall. Dividing by (1 - beta^t) exactly cancels that bias.
"""

from __future__ import annotations

import numpy as np

from .layers import Parameter


class Optimizer:
    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad.fill(0.0)

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Plain stochastic gradient descent - a baseline to compare Adam to."""

    def __init__(self, parameters: list[Parameter], lr: float = 0.1):
        super().__init__(parameters)
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            p.value -= self.lr * p.grad


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.value) for p in parameters]
        self.v = [np.zeros_like(p.value) for p in parameters]

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for i, p in enumerate(self.parameters):
            grad = p.grad
            if self.weight_decay:
                # Decoupled (AdamW-style): applied to the parameter, not
                # folded into the gradient, so it is not rescaled by v.
                p.value -= self.lr * self.weight_decay * p.value

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad**2

            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(parameters: list[Parameter], max_norm: float) -> float:
    """Rescale all gradients so their combined L2 norm is <= `max_norm`.

    Transformers occasionally produce a single huge gradient (a rare token,
    an unlucky batch) that would take one destructive step and undo many
    good ones. Clipping the *global* norm - not per-parameter - preserves
    the gradient's direction while bounding its length.

    Returns the norm before clipping, which is worth logging: a sudden
    spike is the earliest visible sign of a diverging run.
    """
    total = np.sqrt(sum(float((p.grad**2).sum()) for p in parameters))
    if total > max_norm and total > 0:
        scale = max_norm / total
        for p in parameters:
            p.grad *= scale
    return total
