import numpy as np
import pytest

from khmer_language.models.from_scratch.layers import Parameter
from khmer_language.models.from_scratch.optimizer import SGD, Adam, clip_grad_norm


def test_sgd_steps_downhill():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([2.0])
    SGD([p], lr=0.1).step()
    assert p.value[0] == pytest.approx(0.8)


def test_adam_minimizes_a_quadratic():
    """Minimize f(x) = x^2, whose gradient is 2x; the minimum is 0."""
    p = Parameter(np.array([5.0]))
    opt = Adam([p], lr=0.1)
    for _ in range(300):
        p.grad = 2 * p.value
        opt.step()
    assert abs(p.value[0]) < 0.05


def test_adam_bias_correction_makes_the_first_step_full_sized():
    """Without bias correction the first step would be ~lr*(1-beta1)
    instead of ~lr, i.e. 10x too small."""
    p = Parameter(np.array([0.0]))
    p.grad = np.array([1.0])
    Adam([p], lr=0.1).step()
    assert abs(p.value[0]) == pytest.approx(0.1, rel=1e-3)


def test_adam_step_size_is_scale_invariant():
    """Adam normalizes by sqrt(v), so a constant gradient of 1 and of 1000
    produce the same first step - that is the whole point of the method."""
    small, big = Parameter(np.array([0.0])), Parameter(np.array([0.0]))
    small.grad, big.grad = np.array([1.0]), np.array([1000.0])
    Adam([small], lr=0.1).step()
    Adam([big], lr=0.1).step()
    assert small.value[0] == pytest.approx(big.value[0], rel=1e-6)


def test_adam_zero_grad_clears_gradients():
    p = Parameter(np.ones(3))
    p.grad = np.ones(3)
    opt = Adam([p])
    opt.zero_grad()
    assert np.all(p.grad == 0)


def test_weight_decay_shrinks_parameters_when_gradient_is_zero():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([0.0])
    Adam([p], lr=0.1, weight_decay=0.1).step()
    assert p.value[0] < 1.0


def test_clip_grad_norm_rescales_only_when_over_the_limit():
    p = Parameter(np.zeros(2))
    p.grad = np.array([3.0, 4.0])  # norm 5
    returned = clip_grad_norm([p], max_norm=1.0)
    assert returned == pytest.approx(5.0)
    assert np.linalg.norm(p.grad) == pytest.approx(1.0)
    # direction preserved
    assert p.grad[1] / p.grad[0] == pytest.approx(4.0 / 3.0)


def test_clip_grad_norm_leaves_small_gradients_untouched():
    p = Parameter(np.zeros(2))
    p.grad = np.array([0.3, 0.4])  # norm 0.5
    clip_grad_norm([p], max_norm=1.0)
    assert np.allclose(p.grad, [0.3, 0.4])


def test_clip_grad_norm_is_global_across_parameters():
    a, b = Parameter(np.zeros(1)), Parameter(np.zeros(1))
    a.grad, b.grad = np.array([3.0]), np.array([4.0])
    assert clip_grad_norm([a, b], max_norm=1.0) == pytest.approx(5.0)
    combined = np.sqrt(a.grad[0] ** 2 + b.grad[0] ** 2)
    assert combined == pytest.approx(1.0)
