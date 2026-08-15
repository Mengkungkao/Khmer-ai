"""From-scratch neural network components (README.md section 13).

NumPy only, no autograd: every layer's backward pass is derived by hand
and verified against numerical gradients (`gradcheck.py`).
"""

from .layers import (
    GELU,
    Embedding,
    Layer,
    LayerNorm,
    Linear,
    Parameter,
    cross_entropy_loss,
    softmax,
)

__all__ = [
    "GELU",
    "Embedding",
    "Layer",
    "LayerNorm",
    "Linear",
    "Parameter",
    "cross_entropy_loss",
    "softmax",
]
