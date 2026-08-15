"""Numerical gradient checking.

The whole point of hand-deriving backward passes (README section 13) is
undermined if the derivation is silently wrong, and a wrong gradient does
not raise an error - it just trains badly. So every backward pass in this
package is checked against a central finite difference:

    df/dx  ~=  (f(x + h) - f(x - h)) / 2h

The central difference is used rather than the one-sided
`(f(x+h) - f(x))/h` because its error is O(h^2) instead of O(h), which
matters at the small h needed to avoid float64 cancellation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Finite-difference gradient of `f` w.r.t. the array `x`.

    `f` must read `x` in place (it takes no arguments), so this works
    equally for layer inputs and for parameter arrays.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + h
        plus = f()
        x[idx] = original - h
        minus = f()
        x[idx] = original

        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-invariant gradient comparison.

    Plain absolute difference is meaningless when gradients are tiny or
    huge, so compare relative to magnitude. Below ~1e-7 means the analytic
    gradient matches; above ~1e-4 means the derivation is wrong.

    Careful: this metric is *undefined in practice* when both values sit
    in the finite-difference noise floor - comparing an exactly-zero
    analytic gradient against 4e-11 of numerical noise yields a relative
    error of 1.0 despite both being "zero". Use `gradients_match` for
    assertions; use this for reporting a single headline number.
    """
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def gradients_match(
    analytic: np.ndarray,
    numeric: np.ndarray,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> bool:
    """Whether an analytic gradient matches a numerical one, element-wise.

    An element passes if EITHER test passes:

    - relative:  |a-b| / (|a|+|b|)  <= rel_tol
    - absolute:  |a-b|              <= abs_tol

    Both criteria are needed. The relative test alone gives false failures
    for genuinely-zero gradients, where the numerical estimate is pure
    cancellation noise: with f ~ O(1) and h = 1e-5, float64 cancellation
    puts that noise at roughly eps*|f| / 2h ~ 1e-11, which is exactly the
    magnitude observed for gradients that are mathematically exactly zero
    (e.g. an attention key bias - see `test_attention.py`).

    `abs_tol` is set two orders of magnitude above that noise floor but
    far below any gradient that could affect training, so this admits
    noise without hiding real errors.
    """
    diff = np.abs(analytic - numeric)
    denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-12)
    return bool(np.all((diff <= abs_tol) | (diff / denom <= rel_tol)))
